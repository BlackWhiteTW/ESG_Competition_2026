"""
判斷結果檢視工具
"""

import os
import sys
import argparse
import torch
import random

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from dataset import ESGDataset
from model import ESGMultiTaskModel
from inference import ESGInference
from utils import print_system_info
import config


def print_comparison(title: str, pred: str, truth: str = None):
    """格式化輸出對比結果"""
    if truth is not None:
        status = "[PASS]" if pred == truth else "[FAIL]"
        print(f"  {title:15s}: {pred:10s} (標籤: {truth:10s}) {status}")
    else:
        print(f"  {title:15s}: {pred}")


def inspect_results(
    checkpoint_path: str,
    json_file: str,
    num_samples: int = 10,
    random_sample: bool = True,
):
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()
    """檢視模型判斷結果"""
    print("\n" + "=" * 80)
    print("[INSPECT] 模型判斷結果檢視")
    print("=" * 80)

    # ========== 載入資料 ==========
    # 使用 config 中的預設模型與長度
    dataset = ESGDataset(
        json_file=json_file,
        model_name=config.MODEL_NAME,
        max_length=config.MAX_SEQ_LENGTH,
        debug=False,
    )

    # 選擇要顯示的索引
    if random_sample:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    else:
        indices = list(range(min(num_samples, len(dataset))))

    # ========== 載入模型 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(model_name=config.MODEL_NAME)

    inference_engine = ESGInference(
        model=model, checkpoint_paths=[checkpoint_path], device=device
    )

    # ========== 執行推理 ==========
    predictions = inference_engine.inference_on_dataset(
        dataset, batch_size=1, promise_threshold=0.5, evidence_threshold=0.5
    )

    # 過濾出我們選中的索引對應的預測結果
    selected_preds = [predictions[i] for i in indices]

    # ========== 顯示結果 ==========
    for i, res in enumerate(selected_preds):
        print("\n" + "-" * 40)
        print(f"樣本索引: {indices[i]} | ID: {res['index']}")
        print("-" * 40)

        # 顯示原始文字（前150字）
        text_display = res["data"][:150] + "..." if len(res["data"]) > 150 else res["data"]
        print(f"原始文本: \n{text_display}\n")

        # 顯示各項判斷
        print_comparison("永續承諾 (Promise)", res["promise_status"])
        if res["promise_status"] == "Yes":
            print(f'  提取片段: "{res["promise_string"]}"')

        print_comparison("ESG 類型", res["esg_label"])
        print(f"  驗證時間線: {res['timeline_label']}")

        print_comparison("證據狀態 (Evidence)", res["evidence_status"])
        if res["evidence_status"] == "Yes":
            print(f'  證據片段: "{res["evidence_string"]}"')

    print("\n" + "=" * 80)
    print(f"[SUCCESS] 已完成 {len(selected_preds)} 筆數據的檢視")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG 競賽 - 判斷結果檢視工具")
    parser.add_argument(
        "--checkpoint", default=config.DEFAULT_CHECKPOINT, help="模型路徑"
    )
    parser.add_argument("--data", default=config.DATA_PATH, help="資料路徑")
    parser.add_argument("--samples", type=int, default=5, help="顯示樣本數")
    parser.add_argument(
        "--no-random", action="store_true", help="不使用隨機抽樣，按順序顯示"
    )

    args = parser.parse_args()

    inspect_results(
        checkpoint_path=args.checkpoint,
        json_file=args.data,
        num_samples=args.samples,
        random_sample=not args.no_random,
    )
