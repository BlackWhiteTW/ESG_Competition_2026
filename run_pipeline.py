import subprocess
import sys
import os
import time
import argparse

# 將 src 加入路徑以導入 config
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import config
from utils import print_system_info


def run_step(command, description):
    print(f"\n[PIPELINE] Executing: {description}")
    start_time = time.time()
    process = subprocess.Popen(command, shell=True)
    process.wait()
    if process.returncode != 0:
        print(f"[ERROR] {description} failed.")
        sys.exit(1)
    print(f"[SUCCESS] {description} done. ({time.time() - start_time:.2f}s)")


def main():
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=config.DATA_PATH)
    parser.add_argument("--folds", type=int, default=config.DEFAULT_FOLDS)
    parser.add_argument("--epochs", type=int, default=config.DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    overall_start = time.time()
    python_cmd = sys.executable

    if not args.skip_train:
        run_step(
            [
                python_cmd,
                "scripts/train_kfold.py",
                "--folds",
                str(args.folds),
                "--epochs",
                str(args.epochs),
                "--batch-size",
                str(args.batch_size),
                "--lr",
                str(args.lr),
                "--data",
                args.data,
            ],
            "K-Fold Training",
        )

    run_step([python_cmd, "scripts/merge_models.py"], "Model Merging")

    stitched_model = config.DEFAULT_CHECKPOINT
    run_step(
        [
            python_cmd,
            "scripts/optimize_thresholds.py",
            "--checkpoint",
            stitched_model,
            "--data",
            args.data,
        ],
        "Threshold Optimization",
    )
    run_step(
        [
            python_cmd,
            "scripts/run_inference.py",
            "--checkpoint",
            stitched_model,
            "--data",
            args.data,
        ],
        "Final Inference",
    )
    run_step(
        [
            python_cmd,
            "scripts/evaluate_model.py",
            "--checkpoint",
            stitched_model,
            "--data",
            args.data,
        ],
        "Validation",
    )

    print(f"\n[FINISH] Overall time: {(time.time() - overall_start) / 60:.2f} mins")


if __name__ == "__main__":
    main()
