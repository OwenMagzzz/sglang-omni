# SPDX-License-Identifier: Apache-2.0
"""Check long-recording references, scoring and transport error handling."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from sglang_omni_mlx.native.ci.long_audio_client import (
    MemorySampler,
    score,
    transcribe_http,
)
from sglang_omni_mlx.native.ci.long_audio_corpus import (
    Clip,
    build_recording,
    load_clips,
)


def test_recording_keeps_whole_clips_and_matching_references(tmp_path: Path) -> None:
    """Truncating the last clip would score words that were never sent."""
    (tmp_path / "clips").mkdir()
    records = []
    for index in range(3):
        clip_id = f"{index:04d}_en_mid"
        with wave.open(str(tmp_path / "clips" / f"{clip_id}.wav"), "wb") as writer:
            writer.setnchannels(1)
            writer.setsampwidth(2)
            writer.setframerate(16000)
            writer.writeframes(bytes([index, 0]) * 16000)
        records.append(
            {
                "id": clip_id,
                "stratum": "en_mid",
                "lang": "en",
                "reference": f"word{index}",
            }
        )
    (tmp_path / "manifest.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records)
    )
    clips = load_clips(tmp_path, "en")
    result = build_recording(clips, 7.5, 2888, tmp_path / "long.wav")
    reference_by_id = {clip.clip_id: clip.reference for clip in clips}
    assert result["audio_seconds"] == 8
    assert result["reference"] == " ".join(
        reference_by_id[clip_id] for clip_id in result["clip_ids"]
    )
    assert len(set(result["clip_ids"][:3])) == 3
    assert all(
        left != right for left, right in zip(result["clip_ids"], result["clip_ids"][1:])
    )
    with wave.open(str(tmp_path / "long.wav"), "rb") as reader:
        assert reader.getnframes() == 8 * 16000
        expected = b"".join(
            bytes([int(clip_id[:4]), 0]) * 16000 for clip_id in result["clip_ids"]
        )
        assert reader.readframes(reader.getnframes()) == expected
    repeated = build_recording(clips, 7.5, 2888, tmp_path / "again.wav")
    assert repeated == result
    assert (
        result["wav_sha256"]
        == hashlib.sha256((tmp_path / "long.wav").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize(
    ("language", "reference", "hypothesis", "errors", "reference_tokens"),
    [
        ("en", "It's FIVE words right here.", "it's five words", 2, 5),
        ("zh", "今天天气好。", "今天天气好好好", 2, 5),
    ],
)
def test_score_counts_dropped_tails_and_repeated_text(
    language: str, reference: str, hypothesis: str, errors: int, reference_tokens: int
) -> None:
    result = score(reference, hypothesis, language)
    assert result["errors"] == errors
    assert result["reference_tokens"] == reference_tokens
    assert result["error_rate"] == errors / reference_tokens


def test_empty_reference_is_not_a_perfect_score() -> None:
    with pytest.raises(ValueError, match="empty reference"):
        score("", "hallucinated text", "en")


@pytest.mark.skipif(platform.system() != "Darwin", reason="macOS process accounting")
@pytest.mark.parametrize("limit_bytes,limit_seconds", [(1, 60), (2**40, 0)])
def test_resource_guard_stops_its_child(
    tmp_path: Path, limit_bytes: int, limit_seconds: float
) -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "input()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    sampler = MemorySampler(
        process, tmp_path / "memory.jsonl", limit_bytes, limit_seconds
    )
    try:
        sampler.thread.start()
        assert process.wait(timeout=5) != 0
        sampler.thread.join(timeout=5)
        assert not sampler.thread.is_alive()
        if limit_seconds == 0:
            assert sampler.error == "Case wall-clock deadline exceeded"
        else:
            assert sampler.exceeded
        assert sampler.samples[0]["footprint_bytes"] > 0
    finally:
        sampler.done.set()
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        else:
            pass


@pytest.mark.parametrize("seconds", [0, -1, float("nan"), float("inf")])
def test_invalid_duration_cannot_create_an_unbounded_recording(
    tmp_path: Path, seconds: float
) -> None:
    clips = [
        Clip(
            clip_id=str(index),
            path=tmp_path / f"{index}.wav",
            reference="hello",
            frames=16000,
            sha256="unused",
        )
        for index in range(2)
    ]
    with pytest.raises(ValueError):
        build_recording(clips, seconds, 2888, tmp_path / "invalid.wav")
    assert not (tmp_path / "invalid.wav").exists()


@pytest.mark.parametrize(
    "reply",
    [
        b'data: {"type":"transcript.text.done","text":"actual transcript"}\n\n',
        b'data: {"type":"error","error":"failed"}\n\n',
        b":\n\n",
    ],
)
def test_http_requires_a_successful_final_event(reply: bytes) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.end_headers()
            self.wfile.write(reply)

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.handle_request)
        thread.start()
        try:
            if b"transcript.text.done" in reply:
                assert (
                    transcribe_http(server.server_port, b"audio", "en", 2)["text"]
                    == "actual transcript"
                )
            else:
                with pytest.raises(RuntimeError):
                    transcribe_http(server.server_port, b"audio", "en", 2)
        finally:
            thread.join(timeout=5)
