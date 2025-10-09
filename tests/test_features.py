import numpy as np
import pytest
from pathlib import Path
from PIL import Image

from duplicate_check import features


@pytest.fixture()
def sample_image(tmp_path: Path) -> Path:
    path = tmp_path / "sample.png"
    img = Image.new("RGB", (96, 80), color=(128, 128, 128))
    for x in range(96):
        for y in range(80):
            img.putpixel((x, y), (x % 256, y % 256, (x + y) % 256))
    img.save(path)
    return path


def test_compute_phash_variants_multiscale(sample_image: Path):
    variants = features.compute_phash_variants(sample_image)
    unique = {v for v in variants if v}
    assert len(variants) >= len(features.MULTISCALE_LEVELS), "expect multi-scale hashes"
    assert len(unique) >= len(features.MULTISCALE_LEVELS), "hashes should cover multiple scales/orientations"


def test_compute_tile_hashes_structure(sample_image: Path):
    tiles = features.compute_tile_hashes(sample_image, grid=4)
    assert tiles, "tiles should not be empty"
    scales = {tile.get("scale") for tile in tiles}
    assert features.MULTISCALE_LEVELS[0] in scales
    w, h = Image.open(sample_image).size
    for tile in tiles:
        bbox = tile.get("bbox")
        assert isinstance(bbox, tuple) and len(bbox) == 4
        x0, y0, x1, y1 = bbox
        assert 0 <= x0 <= x1 <= w
        assert 0 <= y0 <= y1 <= h


def test_compute_embedding_returns_vector(sample_image: Path):
    emb = features.compute_embedding(sample_image)
    assert emb is not None
    arr = np.asarray(emb)
    assert arr.ndim == 1 and arr.size > 0


def test_compute_features_attaches_tiles(sample_image: Path):
    feats = features.compute_features(sample_image)
    assert feats.tiles is not None and len(feats.tiles) > 0
    assert isinstance(feats.tiles[0], dict)
