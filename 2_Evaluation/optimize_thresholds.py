"""
ESG 競賽 - 最佳門檻優化腳本
"""

import os
import sys
import argparse
import torch

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset import ESGDataset
from model import ESGMultiTaskModel
from inference import ESGInference, ThresholdOptimizer
from analyzer import ESGMockEvaluator
from train import create_data_splits

def optimize():
    parser = argparse.ArgumentParser(description="ESG 競賽 - 門檻優化工具 (2D 自動縮放搜尋)")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt", help="模型路徑")
    parser.add_argument("--data", default="data/vpesg4k_train_1000 V1.json", help="資料路徑")
    
    args = parser.parse_args()
    
    # 1. 載入驗證集
    dataset = ESGDataset(
        json_file=args.data,
        model_name="hfl/chinese-roberta-wwm-ext",
        max_length=256,
        debug=False
    )
    
    _, val_dataloader = create_data_splits(dataset, train_ratio=0.8, seed=42)
    val_dataset = val_dataloader.dataset
    
    # 2. 載入模型與推理引擎
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(model_name="hfl/chinese-roberta-wwm-ext")
    inference_engine = ESGInference(model, args.checkpoint, device)
    
    # 3. 初始化評分器與優化器
    evaluator = ESGMockEvaluator(inference_engine)
    optimizer = ThresholdOptimizer(inference_engine, evaluator)
    
    # 4. 開始執行 2D 自動層級搜尋
    best_params = optimizer.find_optimal_threshold(val_dataset)
    
    # 5. 儲存結果
    import json
    output_dir = "outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    config_path = os.path.join(output_dir, "best_thresholds.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=4)
    
    print(f"\n最佳門檻已儲存至: {config_path}")
    print(f"下次執行 run_inference.py 時將會自動載入此數值。")

if __name__ == "__main__":
    optimize()
