# Qwen3-ASR long-recording measurements

The opt-in benchmark measures the native Qwen3-ASR server through HTTP or its realtime WebSocket. It reports transcription quality alongside memory and elapsed time. Finishing a request does not establish acceptable accuracy, and a tested duration is not the maximum supported duration.

## Run

Build the native runtime with sglang_omni_mlx/native/scripts/build_runtime.sh, install ci/requirements.lock into a Python 3.12 environment, and provision the frozen corpus and Qwen3-ASR-0.6B-4bit using ci/provision.py. From the repository root, run python -m sglang_omni_mlx.native.ci.benchmark_long_audio --help for the required paths.

```bash
python -m sglang_omni_mlx.native.ci.benchmark_long_audio \
  --server "$NATIVE_RUNTIME_BIN/qwen3_asr_server" \
  --model-directory "$CI_DATA_ROOT/models/mlx-community_Qwen3-ASR-0.6B-4bit" \
  --corpus-directory "$CI_DATA_ROOT/corpus/v1" \
  --output "$ASR_RESULTS_DIRECTORY" \
  --mode realtime --minutes 5 10 30 60 --languages en zh --speed 1
```

Supply --server, --model-directory, --corpus-directory and a new --output directory. Use --mode realtime --minutes 5 10 30 60 --languages en zh --speed 1 for live pacing. This matrix contains 210 minutes of audio. A speed greater than one is an accelerated input stress measurement: refresh decodes may be coalesced, and its accuracy and finalization latency are not equivalent to live dictation. Use --mode http separately to measure one whole-file request; this deliberately does not implement Voxt's client-side chunking. Start with five minutes before testing longer whole-file requests.

Each case starts a dedicated server, warms it with a short clip, sends the long recording, then tests SSE disconnection on a separate five-minute recording and transcribes the warmup clip again. An observed running request is required before calling cancellation conclusive. Natural completion of that same five-minute request is also timed: a cancellation that is not materially faster cannot prove that the server stopped early. The post-cancellation transcript must match warmup exactly. Cancellation here covers HTTP SSE, not WebSocket cancellation.

The benchmark polls the dedicated server's macOS physical footprint and resident memory every 250 ms. By default it kills only that server if its physical footprint exceeds 8 GiB; change --max-footprint-gib explicitly to test another budget. Report a limit-triggered run as a failure under that memory budget, not the model's absolute maximum duration. Sampling can miss shorter peaks.

## Input and scoring

The recording concatenates complete short and medium clips of one language from the frozen corpus. A seeded permutation uses every clip before reusing the pool, avoiding the single repeated utterance that can make a long-input cancellation test finish early. Longer recordings necessarily reuse clips; the manifest records their order, unique count, hashes and reuse count. These are synthetic concatenations, not natural hour-long meetings.

The final clip is kept whole, so actual duration is at least the requested duration and may exceed it by one clip. Every reference word remains aligned with audio actually sent. The same seed gives nested recording prefixes across durations. English WER and Chinese CER use the frozen-corpus checker's normalization; edit distance uses RapidFuzz to avoid quadratic Python execution time on hour-long transcripts. Both missing tails and repeated text count as errors. Chinese normalization keeps embedded Latin words as units, as in the existing golden checker.

## Interpret results

- metadata.json records hardware, OS, source revision, working-tree state, executable hash, checkpoint file hashes and arguments.
- Each input manifest records the exact source sequence and expected transcript. Each result stores the hypothesis, error numerator and denominator, timing, memory and cancellation/recovery observations. Realtime runs also save timestamped protocol events.
- The memory trace spans startup, warmup, the long session and cancellation/recovery. Compare equivalent phases; an encoder allocation or stored transcript is not by itself a leak. The warm baseline, session peak, post-session and post-recovery values separate those phases.
- HTTP real-time factor is request wall time divided by audio duration. Realtime wall time includes intentional input pacing; its ratio is not pure model throughput. Finalization time starts after the last audio frame is sent and includes any accumulated server backlog.
- A measured status means the protocol, scoring and recovery measurement completed. It is not an accuracy acceptance threshold. Review the error rate, generated-token finish reason, cancellation comparison and memory trace before drawing conclusions.

Compare before/after builds with identical weights, MLX version, input hashes, pacing and hardware. Keep existing API tests and corpus goldens as correctness gates; this benchmark is an opt-in measurement, not a timing-sensitive unit-test gate.
