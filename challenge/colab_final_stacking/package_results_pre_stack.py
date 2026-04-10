from __future__ import annotations

import zipfile
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
CHALLENGE_ROOT = THIS_DIR.parent
RESULTS_ROOT = CHALLENGE_ROOT / "results-pre-stack"
OUTPUT_DIR = THIS_DIR / "upload_bundles"


def should_include_run(run_dir: Path) -> bool:
    output_dir = run_dir / "output"
    return any(output_dir.rglob("oof_probabilities.csv")) and any(output_dir.rglob("test_probabilities.csv"))


def build_bundle(run_dir: Path) -> Path:
    bundle_path = OUTPUT_DIR / f"{run_dir.name}_for_final_stacking.zip"
    if bundle_path.exists():
        bundle_path.unlink()

    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for top_name in ["output", "submissions"]:
            top_path = run_dir / top_name
            if not top_path.exists():
                continue
            for nested in top_path.rglob("*"):
                if nested.is_dir():
                    continue
                relative_path = nested.relative_to(run_dir)
                zip_file.write(nested, arcname=str(relative_path))

    return bundle_path


def main() -> None:
    if not RESULTS_ROOT.exists():
        raise FileNotFoundError(f"Results root not found: {RESULTS_ROOT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    created = []
    for run_dir in sorted(RESULTS_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        if not should_include_run(run_dir):
            print(f"Skipping {run_dir.name}: no complete OOF/test probability artifacts found.")
            continue
        bundle_path = build_bundle(run_dir)
        created.append(bundle_path)
        print(f"Created {bundle_path}")

    if not created:
        print("No upload-ready bundles were created.")
    else:
        print("\nUpload these bundles to the final Colab notebook:")
        for bundle_path in created:
            print("-", bundle_path)


if __name__ == "__main__":
    main()
