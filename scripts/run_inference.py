"""
推理腳本
"""

import os
import sys
import argparse
import torch
from datetime import datetime
import glob
import json

# 將 src 目錄加入 Python 路徑
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from dataset import ESGDataset
from model import ESGMultiTaskModel
from inference import ESGInference
from utils import print_header, print_system_info
import config


def inference_model(
    checkpoint_path: str,
    json_file: str = config.DATA_PATH,
    output_format: str = "csv",
    promise_threshold: float = 0.5,
    evidence_threshold: float = 0.5,
):
    """執行推理主函式 (支援集成)"""
    print_header("開始推理流程")
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()

    # 智慧型檢查點偵測：優先抓取 models/best_model_X.pt 進行 10-Fold 集成
    fold_ckpts = glob.glob("models/best_model_[0-9]*.pt")

    if fold_ckpts:
        print(f"偵測到多個 Fold 模型，自動啟動集成模式 (共 {len(fold_ckpts)} 個模型)")
        # 排序確保穩定性
        checkpoint_paths = sorted(fold_ckpts)
    else:
        checkpoint_paths = [checkpoint_path]
        print(f"使用單一模型模式: {checkpoint_path}")

    print(
        f"判定門檻設定: Promise={promise_threshold:.2f}, Evidence={evidence_threshold:.2f}"
    )

    if not os.path.exists(json_file):
        print(f"錯誤：找不到資料: {json_file}")
        return

    # ========== 載入資料 ==========
    test_dataset = ESGDataset(
        json_file=json_file,
        model_name=config.MODEL_NAME,
        max_length=config.MAX_SEQ_LENGTH,
        debug=False,
    )

    print(f"資料集已載入: {len(test_dataset)} 筆樣本")

    # ========== 載入模型 ==========
    print_header("載入模型架構")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(model_name=config.MODEL_NAME)

    inference_engine = ESGInference(
        model=model, checkpoint_paths=checkpoint_paths, device=device
    )

    # ========== 執行推理 ==========
    print_header("執行推理")

    predictions = inference_engine.inference_on_dataset(
        test_dataset,
        batch_size=16,
        promise_threshold=promise_threshold,
        evidence_threshold=evidence_threshold,
    )

    # ========== 匯出結果 ==========
    print_header("匯出結果")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 確保輸出目錄存在
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"已建立輸出目錄: {output_dir}")

    if output_format == "csv":
        output_file = os.path.join(output_dir, f"predictions_{timestamp}.csv")
        inference_engine.export_predictions_to_csv(predictions, output_file)
    elif output_format == "json":
        output_file = os.path.join(output_dir, f"predictions_{timestamp}.json")
        inference_engine.export_predictions_to_json(predictions, output_file)

    print(f"結果已匯出: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG 永續承諾驗證競賽 - 推理腳本")
    parser.add_argument(
        "--checkpoint", default=config.DEFAULT_CHECKPOINT, help="模型檢查點路徑"
    )
    parser.add_argument("--data", default=config.DATA_PATH, help="推理資料路徑")
    parser.add_argument(
        "--format", choices=["csv", "json"], default="csv", help="輸出格式"
    )
    parser.add_argument(
        "--promise-threshold",
        type=float,
        default=None,
        help="承諾判定門檻 (預設自動載入最佳值)",
    )
    parser.add_argument(
        "--evidence-threshold",
        type=float,
        default=None,
        help="證據判定門檻 (預設自動載入最佳值)",
    )

    args = parser.parse_args()

    # 嘗試自動載入最佳門檻
    p_thresh = args.promise_threshold
    e_thresh = args.evidence_threshold

    if p_thresh is None or e_thresh is None:
        config_path = config.BEST_THRESHOLDS_PATH
        if os.path.exists(config_path):

            with open(config_path, "r", encoding="utf-8") as f:
                best_params = json.load(f)
                if p_thresh is None:
                    p_thresh = best_params.get("promise_threshold", 0.5)
                if e_thresh is None:
                    e_thresh = best_params.get("evidence_threshold", 0.5)
                print(f"已自動載入最佳門檻設定: {config_path}")
        else:
            if p_thresh is None:
                p_thresh = 0.5
            if e_thresh is None:
                e_thresh = 0.5

    inference_model(
        checkpoint_path=args.checkpoint,
        json_file=args.data,
        output_format=args.format,
        promise_threshold=p_thresh,
        evidence_threshold=e_thresh,
    )
