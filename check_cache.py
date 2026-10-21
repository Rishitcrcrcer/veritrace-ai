"""Reads every tensor from the cached Qwen2-VL weight files, without transformers.

    python check_cache.py

If this crashes or the file sizes look wrong -> the cached download is damaged: delete and re-download.
If it finishes with "all tensors read OK"   -> the files are fine: it is a library/version problem.
"""
import faulthandler
faulthandler.enable()

import torch  # noqa: E402,F401  (load torch first, as in run_demo.py)
import glob  # noqa: E402
import os  # noqa: E402

from safetensors import safe_open  # noqa: E402

root = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub",
                    "models--Qwen--Qwen2-VL-2B-Instruct")
files = sorted(glob.glob(os.path.join(root, "snapshots", "*", "*.safetensors")))
print("cache folder:", root)
print("weight files found:", len(files))
if not files:
    raise SystemExit("No .safetensors files found: the model is not downloaded (or is in another cache folder).")

for p in files:
    print(f"\n{os.path.basename(p)}  size: {os.path.getsize(p) / 1e9:.2f} GB", flush=True)
    with safe_open(p, framework="pt") as f:
        keys = list(f.keys())
        for i, k in enumerate(keys):
            f.get_tensor(k)
            if i % 100 == 0:
                print(f"  read {i}/{len(keys)}", flush=True)
        print(f"  read {len(keys)}/{len(keys)}", flush=True)

print("\nall tensors read OK")
print("Expected sizes: about 4.0 GB + 0.4 GB (4.42 GB in total).")
