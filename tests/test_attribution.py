"""GPU-free tests: synthetic attention matrices with known answers."""
import numpy as np
import pytest

from veritrace.attribution import available, get
from veritrace.core import Explanation, ModelRun


def random_causal(L, rng):
    m = np.tril(rng.random((L, L)) + 0.01)
    return (m / m.sum(axis=1, keepdims=True)).astype(np.float32)


def fake_run(layer_attn, P, A, img_pos, grid):
    return ModelRun(answer_text="", tokens=[f"t{i}" for i in range(A)],
                    layer_attn=layer_attn, prompt_len=P,
                    img_pos=np.asarray(img_pos), grid=grid)


def test_both_methods_registered():
    assert {"raw_attention", "rollout"} <= set(available())


def test_unknown_method_gives_helpful_error():
    with pytest.raises(KeyError, match="Available"):
        get("does_not_exist")


@pytest.mark.parametrize("name", ["raw_attention", "rollout"])
def test_each_token_map_sums_to_one(name):
    rng = np.random.default_rng(0)
    L, P, A = 30, 20, 10
    run = fake_run([random_causal(L, rng) for _ in range(6)], P, A, np.arange(5, 17), (3, 4))
    ex = get(name).explain(run)
    assert ex.heatmaps.shape == (A, 3, 4)
    np.testing.assert_allclose(ex.heatmaps.sum(axis=(1, 2)), 1.0, atol=1e-5)
    assert np.all(ex.image_mass > 0) and np.all(ex.image_mass <= 1.0 + 1e-6)


def test_rollout_with_one_layer_and_no_residual_equals_raw():
    """Sanity link between the two methods."""
    rng = np.random.default_rng(1)
    run = fake_run([random_causal(30, rng)], 20, 10, np.arange(5, 17), (3, 4))
    raw = get("raw_attention", layer_start_frac=0.0).explain(run)
    roll = get("rollout", residual_weight=0.0).explain(run)
    np.testing.assert_allclose(raw.heatmaps, roll.heatmaps, atol=1e-5)


def test_rollout_finds_indirect_path_that_raw_attention_misses():
    """Token 0 = image. Layer 1: text token 1 copies from the image.
    Layer 2: the deciding token (2) reads token 1. Last-layer attention never
    touches the image; rollout traces the 2-hop path."""
    A1 = np.array([[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
    A2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=np.float32)
    run = fake_run([A1, A2], P=3, A=1, img_pos=[0], grid=(1, 1))
    raw = get("raw_attention").explain(run)                     # last layer only
    roll = get("rollout", residual_weight=0.0).explain(run)
    assert raw.image_mass[0] < 1e-6
    assert roll.image_mass[0] == pytest.approx(1.0)


def test_entropy_bounds():
    uniform = Explanation("u", np.full((1, 2, 2), 0.25, np.float32), np.array([1.0], np.float32), ["a"])
    point = np.zeros((1, 2, 2), np.float32)
    point[0, 0, 0] = 1.0
    peaked = Explanation("p", point, np.array([1.0], np.float32), ["a"])
    assert uniform.entropy()[0] == pytest.approx(1.0, abs=1e-6)
    assert peaked.entropy()[0] == pytest.approx(0.0, abs=1e-6)


def test_entropy_is_nan_when_token_ignores_image():
    ex = Explanation("x", np.zeros((1, 2, 2), np.float32), np.array([0.0], np.float32), ["a"])
    assert np.isnan(ex.entropy()[0])


def test_grid_mismatch_raises():
    rng = np.random.default_rng(2)
    run = fake_run([random_causal(30, rng)], 20, 10, np.arange(5, 17), (3, 5))   # 12 tokens, grid 15
    with pytest.raises(ValueError, match="grid"):
        get("raw_attention").explain(run)
