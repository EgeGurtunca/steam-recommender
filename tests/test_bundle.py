import io

import numpy as np

from recs import bundle


def test_the_bundle_drops_the_training_histories_kept_for_checks(tmp_path):
    np.savez(tmp_path / "w.npz", item_ids=np.arange(3), dim=8, check_seq=np.ones((8, 50)), check_scores=np.ones(8))
    kept = np.load(io.BytesIO(bundle.without_checks(tmp_path / "w.npz")))
    assert sorted(kept.keys()) == ["dim", "item_ids"]


def test_the_notice_carries_the_mit_text_and_both_copyright_lines():
    assert "Permission is hereby granted" in bundle.NOTICE and "WITHOUT WARRANTY" in bundle.NOTICE
    assert "Copyright (c) SibamSamanta07" in bundle.NOTICE and "Copyright (c) Martin Bustos" in bundle.NOTICE
