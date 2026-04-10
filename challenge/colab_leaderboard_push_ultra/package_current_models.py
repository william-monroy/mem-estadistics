from __future__ import annotations

import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "results-pre-stack"
TARGET_ROOT = ROOT / "colab_leaderboard_push_ultra" / "upload_bundles"
TARGET_ZIP = TARGET_ROOT / "current_artifacts_for_lb_push.zip"

INCLUDE_PATHS = [
    RESULTS_ROOT / "signal_features_ultra" / "output" / "challenge_10_signal_features_colab_ultra",
    RESULTS_ROOT / "knn_cleaning_ultra" / "output" / "challenge_08_knn_cleaning_colab_ultra",
    RESULTS_ROOT / "output" / "challenge_12_final_stacking_colab",
]


def main() -> None:
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    if TARGET_ZIP.exists():
        TARGET_ZIP.unlink()

    with zipfile.ZipFile(TARGET_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in INCLUDE_PATHS:
            if not path.exists():
                print(f"Skip missing path: {path}")
                continue
            for file_path in path.rglob("*"):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(ROOT))

    print(f"Wrote {TARGET_ZIP}")


if __name__ == "__main__":
    main()
