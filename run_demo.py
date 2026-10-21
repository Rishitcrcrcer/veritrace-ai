"""End-to-end demo: image + question -> answer -> several attribution methods -> overlay.

    python run_demo.py --image car.jpg --question "What color is the car?" --word red
"""
import faulthandler
import matplotlib  # noqa: F401  (load order matters on this machine)
import numpy  # noqa: F401
faulthandler.enable()   # prints where it died, even if a native library crashes (common on Windows)

print("[1/4] starting...", flush=True)
import torch  # noqa: E402,F401  keep this FIRST: avoids Windows DLL load-order crashes
print(f"[2/4] torch {torch.__version__} loaded | CUDA available: {torch.cuda.is_available()}", flush=True)

import argparse  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from veritrace.attribution import available, get  # noqa: E402
from veritrace.models.qwen2vl import Qwen2VLWrapper  # noqa: E402
from veritrace.viz.overlay import pick_tokens, save_comparison  # noqa: E402

print("[3/4] imports ok", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", default="Describe this image.")
    ap.add_argument("--methods", nargs="+", default=["raw_attention", "rollout"],
                    help=f"available: {available()}")
    ap.add_argument("--word", default=None, help="draw the tokens containing this word")
    ap.add_argument("--out", default="comparison.png")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"Image not found: {os.path.abspath(args.image)}")

    print("[4/4] loading model (the first load takes a while)...", flush=True)
    wrapper = Qwen2VLWrapper()
    run = wrapper.run(Image.open(args.image), args.question, args.max_new_tokens)
    print(f"\nQ: {run.question}\nA: {run.answer_text}")
    print(f"Image grid: {run.grid[0]} x {run.grid[1]} | answer tokens: {run.n_answer} "
          f"| sequence length: {run.seq_len}\n")

    explanations = [get(name).explain(run) for name in args.methods]
    token_ids = pick_tokens(run.tokens, args.word)

    for ex in explanations:
        ent = ex.entropy()
        print(f"== {ex.method}  {ex.meta}")
        print(f"   mean entropy {np.nanmean(ent):.3f} | mean image mass {ex.image_mass.mean():.3f}")
        print(f"   {'token':<14}{'top patch':<12}{'entropy':<10}{'image mass'}")
        for i in token_ids:
            top = np.unravel_index(ex.heatmaps[i].argmax(), ex.heatmaps[i].shape)
            print(f"   {run.tokens[i]!r:<14}{str(top):<12}{ent[i]:<10.3f}{ex.image_mass[i]:.3f}")
        print()

    save_comparison(run, explanations, args.out, token_ids)
    print(f"Saved -> {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()