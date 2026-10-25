import numpy as np

from recs.sasrec_np import SASRec


def tiny(path, n_items=6, dim=4, layers=1):
    """Random weights under torch's names, the shape eval/sasrec.py saves."""
    r = np.random.default_rng(0)
    w = {"item.weight": r.normal(size=(n_items + 1, dim)), "pos.weight": r.normal(size=(50, dim)),
         "norm.weight": np.ones(dim), "norm.bias": np.zeros(dim)}
    for i in range(layers):
        p = f"enc.layers.{i}."
        w |= {p + "self_attn.in_proj_weight": r.normal(size=(3 * dim, dim)), p + "self_attn.in_proj_bias": np.zeros(3 * dim),
              p + "self_attn.out_proj.weight": r.normal(size=(dim, dim)), p + "self_attn.out_proj.bias": np.zeros(dim),
              p + "linear1.weight": r.normal(size=(2 * dim, dim)), p + "linear1.bias": np.zeros(2 * dim),
              p + "linear2.weight": r.normal(size=(dim, 2 * dim)), p + "linear2.bias": np.zeros(dim),
              p + "norm1.weight": np.ones(dim), p + "norm1.bias": np.zeros(dim),
              p + "norm2.weight": np.ones(dim), p + "norm2.bias": np.zeros(dim)}
    np.savez(path, item_ids=np.arange(100, 100 + n_items), **{k: v.astype(np.float32) for k, v in w.items()})
    return SASRec(path)


def test_a_step_never_sees_the_games_after_it(tmp_path):
    m = tiny(tmp_path / "m.npz")
    a = m.hidden(np.array([[1, 2, 3, 0]]))
    b = m.hidden(np.array([[1, 2, 5, 4]]))
    assert np.allclose(a[0, :2], b[0, :2]) and not np.allclose(a[0, 2], b[0, 2])


def test_ranking_leaves_out_owned_games(tmp_path):
    m = tiny(tmp_path / "m.npz")
    picks = m.ranking([100, 101, 999], 3)  # 999 is unknown to the model: ignored
    assert len(picks) == 3 and not {100, 101} & set(picks) and set(picks) <= set(range(102, 106))
