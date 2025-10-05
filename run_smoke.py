"""Run a simple smoke test of the duplicate check pipeline without pytest.

Creates temporary directories with stub files and runs the main flow.
"""
import tempfile
from pathlib import Path

from duplicate_check import indexer, features, matcher, report


def run():
    with tempfile.TemporaryDirectory() as db_dir, tempfile.TemporaryDirectory() as in_dir, tempfile.TemporaryDirectory() as out_dir:
        dbp = Path(db_dir)
        inp = Path(in_dir)
        outp = Path(out_dir)
        # create stub files
        (dbp / "db_1.jpg").write_text("stub")
        (inp / "new_1.jpg").write_text("stub")

        print("Building index...")
        idx = indexer.build_index(dbp)
        print("Computing features for input...")
        feats = features.compute_features(inp / "new_1.jpg")
        print("Recalling candidates...")
        cands = matcher.recall_candidates(feats, idx)
        print("Reranking/verifying...")
        rows = matcher.rerank_and_verify(inp / "new_1.jpg", cands)
        csvp = outp / "dup_report.csv"
        report.write_csv(rows, csvp)
        print(f"Smoke run complete. Report: {csvp}")


if __name__ == "__main__":
    run()
