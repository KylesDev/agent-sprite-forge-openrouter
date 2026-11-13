"""OpenRouter adapter (media_providers.OpenRouterAdapter): image-to-video through OpenRouter's normalized video API
under one OPENROUTER_API_KEY: request shapes, token pricing, the poll, the keyed download, error codes and gating.

Offline: the real generate_media.Transport talks to MockHTTPS (test_media_providers_common). The shapes follow
OpenRouter's video-generation guide and the Video Models API as read on 2026-11-13; nothing here proves a live call.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from test_media_providers_common import (KEYS, assert_contracts, assert_no_secret, http, ledger_lines, media_mp4,  # noqa: F401
                                         mp4, no_network, png, project, run_media, run_route, surfaces)

API = "https://openrouter.ai/api/v1"
KEY = KEYS["OPENROUTER_API_KEY"]


@pytest.fixture
def key(monkeypatch, project):
    """OPENROUTER_API_KEY, plus still.png: a 512 px square master."""
    monkeypatch.setenv("OPENROUTER_API_KEY", KEY)
    (project / "still.png").write_bytes(png((512, 512)))


def data_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def test_seedance_mini_pins_both_frames_square_silent_and_priced_by_tokens(project, http, key, capsys):
    http.on("POST", API + "/videos", (202, {"id": "vid_abc123", "polling_url": API + "/videos/vid_abc123",
                                            "status": "pending"}))
    http.on("GET", API + "/videos/vid_abc123", (200, {"id": "vid_abc123", "status": "pending"}),
            (200, {"id": "vid_abc123", "status": "in_progress"}),
            (200, {"id": "vid_abc123", "status": "completed",
                   "unsigned_urls": [API + "/videos/vid_abc123/content?index=0"],
                   "usage": {"cost": 0.1358, "is_byok": False}}), exact=True)
    http.on("GET", API + "/videos/vid_abc123/content", (200, mp4()))
    code, result, err = run_route(capsys, "video", "--prompt-file", "prompt.txt", "--route", "openrouter",
                                  "--reference", "still.png", "--last-frame", "still.png", "--resolution", "480p",
                                  "--duration", "4", "--out-dir", "out/idle")
    assert code == 0, err
    assert (result["route"], result["model"], result["lastFrameUsed"]) == ("api:openrouter",
                                                                           "bytedance/seedance-2.0-mini", True)
    post = http.calls("POST")[0]
    assert post.headers["authorization"] == "Bearer " + KEY
    body = post.json()
    first = data_uri(project / "still.png")
    assert body == {"model": "bytedance/seedance-2.0-mini", "prompt": body["prompt"], "duration": 4,
                    "resolution": "480p", "aspect_ratio": "1:1", "generate_audio": False, "frame_images": [
                        {"type": "image_url", "image_url": {"url": first}, "frame_type": "first_frame"},
                        {"type": "image_url", "image_url": {"url": first}, "frame_type": "last_frame"}]}
    download = http.calls("GET")[-1]
    assert download.url == API + "/videos/vid_abc123/content?index=0"
    assert download.headers["authorization"] == "Bearer " + KEY  # an OpenRouter-hosted file on the API host
    folder = project / "out" / "idle"
    assert media_mp4.scan((folder / "generated.mp4").read_bytes())["tracks"] == ["vide"]
    job = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    assert job["requestId"] == "vid_abc123" and job["options"]["frames"] == ["first_frame", "last_frame"]
    assert result["estimateUsd"] == pytest.approx(0.1358)  # 640x640 x 97 frames / 1024 = 38800 tokens (live 2026-11-13)
    assert job["costUsd"] == 0.1358 and "video tokens" in job["estimate"]["basis"]
    assert_contracts(job, project)
    line = ledger_lines(project)[-1]
    assert (line["status"], line["jobId"], line["actualUsd"]) == ("done", "vid_abc123", 0.1358)
    assert_no_secret(*surfaces(project, "out/idle", json.dumps(result), err))


def test_tiers_draft_drops_the_last_frame_and_hero_is_720p(project, http, key, capsys):
    code, result, err = run_route(capsys, "video", "--prompt-file", "prompt.txt", "--route", "openrouter", "--tier",
                                  "draft", "--reference", "still.png", "--last-frame", "still.png", "--resolution",
                                  "480p", "--duration", "4", "--out-dir", "out/draft", "--dry-run")
    assert code == 0, err
    assert result["model"] == "x-ai/grok-imagine-video-1.5-lite" and result["lastFrameUsed"] is False
    assert result["estimateUsd"] == pytest.approx(4 * 0.02 + 0.01)  # plus one input image
    code, result, err = run_route(capsys, "video", "--prompt-file", "prompt.txt", "--route", "openrouter", "--tier",
                                  "hero", "--reference", "still.png", "--last-frame", "still.png", "--duration", "5",
                                  "--out-dir", "out/hero", "--dry-run")
    assert code == 0, err
    assert result["model"] == "kwaivgi/kling-v3.0-std" and result["estimateUsd"] == pytest.approx(5 * 0.084)
    assert not http.requests


def test_veo_lite_pads_a_square_still_onto_16_9(project, http, key, capsys):
    http.on("POST", API + "/videos", (202, {"id": "vid_veo", "status": "pending"}))
    http.on("GET", API + "/videos/vid_veo", (200, {"status": "completed"}), exact=True)
    http.on("GET", API + "/videos/vid_veo/content", (200, mp4(audio=False)))
    code, out, err = run_media(capsys, "video", "--provider", "openrouter", "--model", "google/veo-3.1-lite",
                               "--prompt-file", "prompt.txt", "--reference", "still.png", "--duration", "4",
                               "--out-dir", "out/veo", "--project-dir", ".", "--execute")
    assert code == 0, err
    body = http.calls("POST")[0].json()
    assert (body["resolution"], body["aspect_ratio"], body["generate_audio"]) == ("720p", "16:9", False)
    assert http.calls("GET")[-1].url == API + "/videos/vid_veo/content?index=0"  # no URL reported: the documented one
    job = json.loads((project / "out" / "veo" / "job.json").read_text(encoding="utf-8"))
    assert job["crop"]["width"] == pytest.approx(512 / 912, abs=1e-3)


@pytest.mark.parametrize("status, message, code", [
    (401, "No auth credentials found", "auth"),
    (402, "Insufficient credits", "quota"),
    (403, "Input was flagged by moderation", "moderation"),
    (404, "No endpoints found for this model", "entitlement"),
    (429, "Rate limit exceeded", "rate_limit"),
    (400, "Invalid duration", "invalid_request"),
    (502, "Upstream error", "provider_error"),
])
def test_error_codes_map_to_asf_codes(project, http, key, capsys, status, message, code):
    http.on("POST", API + "/videos", (status, {"error": {"code": status, "message": message}}))
    rc, out, err = run_media(capsys, "video", "--provider", "openrouter", "--model", "bytedance/seedance-2.0-mini",
                             "--prompt-file", "prompt.txt", "--reference", "still.png", "--resolution", "480p",
                             "--duration", "4", "--out-dir", "out/err", "--project-dir", ".", "--execute")
    assert rc == 1 and len(http.requests) == 1
    job = json.loads((project / "out" / "err" / "job.json").read_text(encoding="utf-8"))
    assert job["receipt"]["outcomeCode"] == code


@pytest.mark.parametrize("task, status, code", [
    ({"status": "failed", "error": "Output was blocked by the safety filter"}, "failed", "moderation"),
    ({"status": "failed", "error": {"message": "upstream timeout"}}, "failed", "failed"),
    ({"status": "expired"}, "expired", "expired"),
    ({"status": "cancelled"}, "failed", "failed"),
])
def test_a_failed_job_is_final(project, http, key, capsys, task, status, code):
    http.on("POST", API + "/videos", (202, {"id": "vid_fail", "status": "pending"}))
    http.on("GET", API + "/videos/vid_fail", (200, task), exact=True)
    rc, out, err = run_route(capsys, "video", "--prompt-file", "prompt.txt", "--route", "openrouter", "--reference",
                             "still.png", "--resolution", "480p", "--out-dir", "out/fail")
    assert rc == 1 and err.startswith(f"error: api:openrouter: {code}:") and len(http.calls("POST")) == 1
    job = json.loads((project / "out" / "fail" / "job.json").read_text(encoding="utf-8"))
    assert job["status"] == status


def test_gating_resolutions_and_unknown_models(project, http, key, capsys):
    code, out, err = run_media(capsys, "video", "--provider", "openrouter", "--model", "bytedance/seedance-2.0-mini",
                               "--prompt-file", "prompt.txt", "--reference", "still.png", "--resolution", "1080p",
                               "--out-dir", "out/x", "--project-dir", ".", "--execute")
    assert code == 1 and "renders 480p, 720p" in err
    code, out, err = run_media(capsys, "video", "--provider", "openrouter", "--model", "someone/unknown-video",
                               "--prompt-file", "prompt.txt", "--reference", "still.png", "--out-dir", "out/y",
                               "--project-dir", ".", "--execute")
    assert code == 1 and "Unsupported video model" in err
    assert not http.requests
