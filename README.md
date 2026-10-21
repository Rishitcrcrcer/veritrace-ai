# VeriTrace AI (solo build)

Step 2: modular core. Raw attention and rollout as plug-in attribution methods.

## Run
    pip install -r requirements.txt
    python run_demo.py --image car.jpg --question "What color is the car?" --word red

## Test (no GPU needed)
    python -m pytest -q tests

## Layout
    veritrace/core.py            ModelRun, Explanation (plain numpy containers)
    veritrace/models/qwen2vl.py  the only torch/transformers file
    veritrace/attribution/       base.py, registry.py, attention.py, rollout.py
    veritrace/viz/overlay.py     heatmap drawing
    tests/                       synthetic-attention tests

## Add a new attribution method
1. Create veritrace/attribution/mymethod.py
2. Decorate a subclass of AttributionMethod with @register("my_method"), implement explain(run)
3. Add one import line in veritrace/attribution/__init__.py
