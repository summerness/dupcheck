"""Run a simple smoke test of the duplicate check pipeline without pytest.

Creates temporary directories with tiny JPEG fixtures and runs the main flow.
"""
import base64
import tempfile
from pathlib import Path

from duplicate_check import indexer, features, matcher, report


_TINY_JPEG_B64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wCEAAkGBxISEBUQEBAVFRUVFRUVFRUVFRUVFRUXFhUVFRUYHSggGBolGxUVITEhJSkrLi4uFx8zODMsNygtLisBCgoKDg0OGhAQGy0lICYtLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLS0tLf/AABEIAJ8BPgMBIgACEQEDEQH/xAAbAAABBQEBAAAAAAAAAAAAAAAAAQIEBQYDB//EADwQAAEDAgQDBgMHAwMFAAAAAAEAAgMEEQUSITEGE0FRMmFxgZGh8COhsUIjUmKyweHxFSNDU5LxJENT/8QAGQEAAwEBAQAAAAAAAAAAAAAAAAECAwQF/8QAJhEBAAICAgIBAwUAAAAAAAAAAAECAxESIQQxQVEiUYGh8GH/2gAMAwEAAhEDEQA/AO4gAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAD//Z"
)


def _write_tiny_jpeg(path: Path) -> None:
    """Write a minimal 1x1 JPEG so PIL/OpenCV can read it."""
    path.write_bytes(base64.b64decode(_TINY_JPEG_B64))


def run():
    with tempfile.TemporaryDirectory() as db_dir, tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
        dbp = Path(db_dir)
        inp = Path(in_dir)
        outp = Path(out_dir)
        # create tiny but valid JPEG fixtures
        _write_tiny_jpeg(dbp / "db_1.jpg")
        _write_tiny_jpeg(inp / "new_1.jpg")

        print("Building index...")
        idx = indexer.build_index(dbp)
        print("Computing features for input...")
        feats = features.compute_features(inp / "new_1.jpg")
        print("Recalling candidates...")
        cands = matcher.recall_candidates(feats, idx)
        print("Reranking/verifying...")
        rows = matcher.rerank_and_verify(inp / "new_1.jpg", cands, idx)
        csvp = outp / "dup_report.csv"
        report.write_csv(rows, csvp)
        print(f"Smoke run complete. Report: {csvp}")


if __name__ == "__main__":
    run()
