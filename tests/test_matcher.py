import numpy as np
import pytest

from duplicate_check import matcher


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
def test_count_good_matches_dtype_handling(dtype):
    cv2 = pytest.importorskip("cv2")
    rng = np.random.default_rng(42)
    desc1 = (rng.random((32, 64)) * (255 if dtype == np.uint8 else 1)).astype(dtype)
    desc2 = desc1.copy().astype(dtype)
    result = matcher._count_good_matches(desc1, desc2)
    assert isinstance(result, int)
    assert result >= 0


def test_count_good_matches_mixed_dtype():
    pytest.importorskip("cv2")
    rng = np.random.default_rng(7)
    desc1 = (rng.random((16, 32)) * 255).astype(np.uint8)
    desc2 = desc1.astype(np.float32) / 255.0
    result = matcher._count_good_matches(desc1, desc2)
    assert isinstance(result, int)
    assert result >= 0
