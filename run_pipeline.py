"""
ESG 競賽 - 全自動化工作流主控腳本 (End-to-End Pipeline)
流程：K-Fold 訓練 -> 模型權重融合 -> 自動門檻優化 -> 最終推理輸出 -> 分數驗證
"""

import subprocess
import sys
import os
import time
import argparse

def run_step(command, description):
    """執行單一步驟並監控輸出"""
    print("\n" + "="*80)
    print(f"[PIPELINE] 正在執行: {description}")
    print(f"指令: {' '.join(command)}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    # 使用 subprocess 執行，保留顏色與進度條
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    if process.returncode != 0:
        print(f"\n[ERROR] {description} 失敗 (返回碼: {process.returncode})，自動化流程中斷。")
        sys.exit(1)
    
    elapsed = time.time() - start_time
    print(f"\n[SUCCESS] {description} 完成！耗時: {elapsed/60:.2f} 分鐘")

def main():
    parser = argparse.ArgumentParser(description="ESG 競賽全自動化管線")
    parser.add_argument("--data", default="data/vpesg4k_train_1000 V1.json", help="訓練與驗證資料路徑")
    parser.add_argument("--folds", type=int, default=5, help="K-Fold 折數")
    parser.add_argument("--epochs", type=int, default=20, help="每折訓練輪數")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小")
    parser.add_argument("--lr", type=float, default=1e-5, help="學習率")
    parser.add_argument("--skip-train", action="store_true", help="跳過訓練階段，直接從現有折權重開始融合")
    
    args = parser.parse_args()
    
    overall_start = time.time()
    python_cmd = sys.executable

    # --- 階段 1: K-Fold 訓練 ---
    if not args.skip_train:
        run_step([
            python_cmd, "1_Training/train_kfold.py",
            "--folds", str(args.folds),
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--lr", str(args.lr),
            "--data", args.data
        ], "階段 1: K-Fold 交叉驗證訓練")
    else:
        print("[INFO] 已跳過訓練階段。")

    # --- 階段 2: 模型權重融合 (Stitching) ---
    run_step([
        python_cmd, "src/merge_best_parts.py"
    ], "階段 2: 權重融合 (Weight Averaging)")

    # --- 階段 3: 自動門檻優化 ---
    # 使用融合後的模型在驗證集上尋找最佳 P/E Threshold
    stitched_model = "models/checkpoints/final_stitched_model.pt"
    run_step([
        python_cmd, "2_Evaluation/optimize_thresholds.py",
        "--checkpoint", stitched_model,
        "--data", args.data
    ], "階段 3: 自動門檻優化")

    # --- 階段 4: 執行最終推理 ---
    # 會自動載入 outputs/best_thresholds.json
    run_step([
        python_cmd, "3_Inference/run_inference.py",
        "--checkpoint", stitched_model,
        "--data", args.data
    ], "階段 4: 執行最終推理並生成 CSV 報表")

    # --- 階段 5: 最終評分確認 ---
    run_step([
        python_cmd, "2_Evaluation/evaluate_model.py",
        "--checkpoint", stitched_model,
        "--data", args.data
    ], "階段 5: 最終模擬評分驗證")

    overall_elapsed = time.time() - overall_start
    print(f"\n" + "!"*80)
    print(f"  恭喜！全自動化流程執行完畢。")
    print(f"  總耗時: {overall_elapsed/60:.2f} 分鐘")
    print(f"  最終模型位置: {stitched_model}")
    print(f"  最佳門檻設定: outputs/best_thresholds.json")
    print(f"  預測結果位置: outputs/predictions_YYYYMMDD_HHMMSS.csv")
    print("!"*80 + "\n")

if __name__ == "__main__":
    main()
