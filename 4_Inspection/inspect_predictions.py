"""
ESG 永續承諾驗證競賽 - 判斷結果檢視工具
"""

import os
import sys
import argparse
import torch
import random
from datetime import datetime

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset import ESGDataset
from model import ESGMultiTaskModel
from inference import ESGInference


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
    random_sample: bool = True
):
    """檢視模型判斷結果"""
    print("\n" + "=" * 80)
    print("[INSPECT] 模型判斷結果檢視")
    print("=" * 80)
    
    # ========== 載入資料 ==========
    dataset = ESGDataset(
        json_file=json_file,
        model_name="hfl/chinese-roberta-wwm-ext",
        max_length=256,
        debug=False
    )
    
    # 選擇要顯示的索引
    if random_sample:
        indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    else:
        indices = list(range(min(num_samples, len(dataset))))
    
    # 建立一個臨時資料集只包含選中的樣本
    from torch.utils.data import Subset
    subset_dataset = Subset(dataset, indices)
    
    # ========== 載入模型 ==========
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(model_name="hfl/chinese-roberta-wwm-ext")
    
    inference_engine = ESGInference(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    # ========== 執行推理 ==========
    # 注意：我們需要直接使用 subset_dataset，但 ESGInference.inference_on_dataset 預期的是 ESGDataset
    # 這裡我們手動處理一下
    predictions = inference_engine.inference_on_dataset(
        dataset, 
        batch_size=1, 
        promise_threshold=0.5, 
        evidence_threshold=0.5
    )
    
    # 過濾出我們選中的索引對應的預測結果
    # 因為 Dataset 順序固定，我們可以直接按索引取值
    selected_preds = [predictions[i] for i in indices]
    
    # ========== 顯示結果 ==========
    for i, res in enumerate(selected_preds):
        print("\n" + "-" * 40)
        print(f"樣本索引: {indices[i]} | ID: {res['id']}")
        print(f"公司名稱: {res['company']}")
        print("-" * 40)
        
        # 顯示原始文字（前100字）
        text_display = res['text'][:150] + "..." if len(res['text']) > 150 else res['text']
        print(f"原始文本: \n{text_display}\n")
        
        # 顯示各項判斷
        truth_p = res.get('truth_promise_status')
        print_comparison("永續承諾 (Promise)", res['promise_status'], truth_p)
        
        if res['promise_status'] == 'Yes':
            print(f"  提取片段: \"{res['promise_string']}\"")
            
        truth_esg = res.get('truth_esg_type')
        print_comparison("ESG 類型", res['esg_type'], truth_esg)
        
        print(f"  驗證時間線: {res['verification_timeline']}")
        
        print_comparison("證據狀態 (Evidence)", res['evidence_status'])
        if res['evidence_status'] == 'Yes':
            print(f"  證據片段: \"{res['evidence_string']}\"")
            
    print("\n" + "=" * 80)
    print(f"[SUCCESS] 已完成 {len(selected_preds)} 筆數據的檢視")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG 競賽 - 判斷結果檢視工具")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt", help="模型路徑")
    parser.add_argument("--data", default="data/vpesg4k_train_1000 V1.json", help="資料路徑")
    parser.add_argument("--samples", type=int, default=5, help="顯示樣本數")
    parser.add_argument("--no-random", action="store_true", help="不使用隨機抽樣，按順序顯示")
    
    args = parser.parse_args()
    
    inspect_results(
        checkpoint_path=args.checkpoint,
        json_file=args.data,
        num_samples=args.samples,
        random_sample=not args.no_random
    )
