set -e

uv run audio_pose_interleaved_inference.py
uv run eval_pose.py
uv run eval_audio.py
uv run decode_pose.py
uv run decode_audio.py
uv run comparisons.py