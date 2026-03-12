"""
ESG 永續承諾驗證競賽 - 推理腳本
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import ESGDataset
from model import ESGMultiTaskModel
from inference import ESGInference


def print_header(message: str):
    """印出格式化的標題"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


def inference_model(
    checkpoint_path: str,
    json_file: str = "data/vpesg4k_train_1000 V1.json",
    output_format: str = "csv"
):
    """執行推理主函式"""
    print_header("🔮 開始推理流程")
    
    if not os.path.exists(json_file):
        print(f"❌ 錯誤：找不到資料: {json_file}")
        return
    
    # ========== 載入資料 ==========
    test_dataset = ESGDataset(
        json_file=json_file,
        model_name="hfl/chinese-roberta-wwm-ext",
        max_length=256,
        debug=False
    )
    
    print(f"✅ 資料集已載入: {len(test_dataset)} 筆樣本")
    
    # ========== 載入模型 ==========
    print_header("🧠 載入模型")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(
        model_name="hfl/chinese-roberta-wwm-ext"
    )
    
    inference_engine = ESGInference(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # ========== 執行推理 ==========
    print_header("🚀 執行推理")
    
    predictions = inference_engine.inference_on_dataset(
        test_dataset,
        batch_size=16,
        promise_threshold=0.5,
        evidence_threshold=0.5
    )
    
    # ========== 匯出結果 ==========
    print_header("💾 匯出結果")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if output_format == "csv":
        output_file = f"predictions_{timestamp}.csv"
        inference_engine.export_predictions_to_csv(predictions, output_file)
    elif output_format == "json":
        output_file = f"predictions_{timestamp}.json"
        inference_engine.export_predictions_to_json(predictions, output_file)
    
    print(f"✅ 結果已匯出: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG 永續承諾驗證競賽 - 推理腳本")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt", help="模型檢查點路徑")
    parser.add_argument("--data", default="data/vpesg4k_train_1000 V1.json", help="推理資料路徑")
    parser.add_argument("--format", choices=["csv", "json"], default="csv", help="輸出格式")
    
    args = parser.parse_args()
    
    inference_model(
        checkpoint_path=args.checkpoint,
        json_file=args.data,
        output_format=args.format
    )
