"""Side-by-side comparison GIFs of eval-truth vs model-generated pose
(optionally with a uniform-random-codes baseline on the far right).

For each index ``n`` that has both ``eval_pose_n.gif`` and ``pose_n.gif`` in
INFERENCE_OUTPUTS_DIR, writes ``comparison_n.gif`` with the panels played
side-by-side at the same fps, frame 0 aligned. If panels differ in length,
shorter ones freeze on their last frame until the longest finishes.

If ``DO_RANDOM`` is True, a third panel is rendered per index: random codes
sampled uniformly in [0, codebook_size) for each of the active codebooks and
each token-rate timestep (matching pose_n.pt's token length), then passed
through the same xabi PoseTokenizer used by decode_pose.py. All pose rendering
delegates to decode_pose so there's exactly one rendering implementation.
"""

import glob
import os
import re
import sys

import yaml
from PIL import Image, ImageDraw, ImageFont


DO_RANDOM = True


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "tokenizer_config.yaml")
CFG = yaml.safe_load(open(CONFIG_PATH))

INFERENCE_OUTPUTS_DIR = CFG["inference_outputs_dir"]
GIF_FPS = CFG["gif"]["fps"]

LABEL_H = 28
LABEL_FONT_SIZE = 18


def _gif_indices(prefix: str) -> set[int]:
    paths = glob.glob(os.path.join(INFERENCE_OUTPUTS_DIR, f"{prefix}_*.gif"))
    pattern = re.compile(rf"{re.escape(prefix)}_(\d+)\.gif$")
    return {int(m.group(1)) for p in paths if (m := pattern.search(p))}


def _load_frames(path: str) -> list[Image.Image]:
    img = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(img.copy().convert("RGB"))
            img.seek(img.tell() + 1)
    except EOFError:
        pass
    return frames


def _label(width: int, text: str) -> Image.Image:
    img = Image.new("RGB", (width, LABEL_H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", LABEL_FONT_SIZE)
    except OSError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((width - tw) // 2, (LABEL_H - th) // 2 - bbox[1]),
              text, fill="black", font=font)
    return img


def _compose(panels: list[tuple[str, list[Image.Image]]]):
    """panels: list of (label, frames). Returns a list of PIL frames."""
    n = max(len(frames) for _, frames in panels)
    sizes = [frames[0].size for _, frames in panels]
    widths = [w for w, _ in sizes]
    heights = [h for _, h in sizes]
    total_w = sum(widths)
    panel_h = max(heights)
    labels = [_label(w, name) for (name, _), w in zip(panels, widths)]

    out = []
    for i in range(n):
        canvas = Image.new("RGB", (total_w, panel_h + LABEL_H), "white")
        x = 0
        for (_, frames), label, (w, h) in zip(panels, labels, sizes):
            canvas.paste(label, (x, 0))
            f = frames[min(i, len(frames) - 1)]
            canvas.paste(f, (x, LABEL_H + (panel_h - h) // 2))
            x += w
        out.append(canvas)
    return out


# ---------------------------------------------------------------------------
# Random-baseline rendering (optional)
# ---------------------------------------------------------------------------

if DO_RANDOM:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    import torch

    from paths import JAMES_TOKENIZER_PATH, XABI_TOKENIZER_PATH
    from pose_tokenizer_xabi import PoseTokenizer
    from decode_pose import decoded_features_to_positions, render_pose_frames

    WEIGHTS_PATH = {"xabi": XABI_TOKENIZER_PATH, "james": JAMES_TOKENIZER_PATH}[CFG["package"]]
    N_CODEBOOKS_OVERRIDE = CFG.get("n_codebooks")

    _device_cfg = CFG.get("device", "auto")
    DEVICE = ("cuda" if torch.cuda.is_available() else "cpu") if _device_cfg == "auto" else _device_cfg

    tokenizer = PoseTokenizer.from_pretrained(WEIGHTS_PATH, device=DEVICE)
    print(f"loaded PoseTokenizer ({CFG['package']}) from {WEIGHTS_PATH} on {DEVICE} "
          f"for random baseline")

    if N_CODEBOOKS_OVERRIDE is not None:
        tokenizer.model.set_active_codebooks(N_CODEBOOKS_OVERRIDE)

    ACTIVE_N_CODEBOOKS = (
        N_CODEBOOKS_OVERRIDE if N_CODEBOOKS_OVERRIDE is not None
        else tokenizer.config.n_codebooks
    )
    CODEBOOK_SIZE = tokenizer.config.codebook_size
    print(f"  random codes: ({ACTIVE_N_CODEBOOKS}, T_tokens) in [0, {CODEBOOK_SIZE})")

    @torch.inference_mode()
    def _random_frames(t_tokens: int) -> list[Image.Image]:
        codes_t = torch.randint(
            low=0, high=CODEBOOK_SIZE,
            size=(ACTIVE_N_CODEBOOKS, t_tokens),
            dtype=torch.long, device=DEVICE,
        )
        codes = list(codes_t.unbind(0))
        recon = tokenizer.decode(codes)
        positions = decoded_features_to_positions(recon)
        return render_pose_frames(positions)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

eval_idx = _gif_indices("eval_pose")
gen_idx = _gif_indices("pose")

if not eval_idx and not gen_idx:
    raise FileNotFoundError(
        f"no eval_pose_*.gif or pose_*.gif found in {INFERENCE_OUTPUTS_DIR}. "
        f"Run `python inference/eval_pose.py` and `python inference/decode_pose.py` "
        f"first to produce the per-sample gifs that this script composes."
    )
if not eval_idx:
    raise FileNotFoundError(
        f"no eval_pose_*.gif in {INFERENCE_OUTPUTS_DIR}. "
        f"Run `python inference/eval_pose.py` first."
    )
if not gen_idx:
    raise FileNotFoundError(
        f"no pose_*.gif in {INFERENCE_OUTPUTS_DIR}. "
        f"Run `python inference/decode_pose.py` first."
    )

shared = sorted(eval_idx & gen_idx)
if not shared:
    raise FileNotFoundError(
        f"found eval_pose_*.gif indices {sorted(eval_idx)} and pose_*.gif indices "
        f"{sorted(gen_idx)} in {INFERENCE_OUTPUTS_DIR}, but no shared indices. "
        f"Re-run inference/eval_pose.py and inference/decode_pose.py on the same "
        f"sample range so their outputs align."
    )

print(f"composing {len(shared)} comparison gifs at {GIF_FPS} fps: {shared}")

for n in shared:
    eval_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"eval_pose_{n}.gif")
    gen_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"pose_{n}.gif")
    eval_frames = _load_frames(eval_path)
    gen_frames = _load_frames(gen_path)

    panels = [("eval", eval_frames), ("generated", gen_frames)]
    info = f"eval={len(eval_frames)}f gen={len(gen_frames)}f"

    if DO_RANDOM:
        pose_pt = os.path.join(INFERENCE_OUTPUTS_DIR, f"pose_{n}.pt")
        t_tokens = int(torch.load(pose_pt, map_location="cpu").shape[1])
        random_frames = _random_frames(t_tokens)
        panels.append(("random", random_frames))
        info += f" random={len(random_frames)}f"

    combined = _compose(panels)
    out_path = os.path.join(INFERENCE_OUTPUTS_DIR, f"comparison_{n}.gif")
    combined[0].save(
        out_path,
        save_all=True,
        append_images=combined[1:],
        duration=int(1000 / GIF_FPS),
        loop=0,
    )
    print(f"  comparison_{n}.gif  {info} -> {len(combined)}f")
