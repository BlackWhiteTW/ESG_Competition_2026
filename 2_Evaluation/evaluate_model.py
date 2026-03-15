"""
ESG 永續承諾驗證競賽 - 驗證評估腳本
"""

import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset import ESGDataset
from model import ESGMultiTaskModel
from train import create_data_splits
from inference import ESGInference
from analyzer import ESGMockEvaluator


def print_header(message: str):
    """印出格式化的標題"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


def evaluate_model(
    checkpoint_path: str,
    json_file: str = "data/vpesg4k_train_1000 V1.json",
    batch_size: int = 8
):
    """評估模型主函式"""
    print_header("開始評估流程")
    
    if not os.path.exists(json_file):
        print(f"錯誤：找不到資料: {json_file}")
        return
    
    # ========== 載入資料並分割 ==========
    dataset = ESGDataset(
        json_file=json_file,
        model_name="hfl/chinese-roberta-wwm-ext-large",
        max_length=256,
        debug=False
    )
    
    # 使用與訓練時相同的分割方式取得驗證集
    _, val_dataloader = create_data_splits(
        dataset,
        train_ratio=0.8,
        batch_size=batch_size,
        seed=42
    )
    
    print(f"驗證集已載入: {len(val_dataloader.dataset)} 筆樣本")
    
    # ========== 載入模型 ==========
    print_header("載入模型")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(
        model_name="hfl/chinese-roberta-wwm-ext-large"
    )
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"模型已載入: {checkpoint_path}")
    else:
        print(f"錯誤: 檢查點不存在 {checkpoint_path}")
        return
        
    model.to(device)
    model.eval()
    
    # ========== [新增] 執行模擬評分與錯誤歸納 ==========
    # 支援單一模型或集成模型
    checkpoint_paths = [checkpoint_path]
    inference_engine = ESGInference(
        model=model, 
        checkpoint_paths=checkpoint_paths, 
        device=device
    )
    evaluator = ESGMockEvaluator(inference_engine)
    
    # 從 DataLoader 中還原原始的 Dataset 物件進行分析
    evaluator.analyze_performance(val_dataloader.dataset)
    
    # ========== 執行損失驗證 ==========
    print_header("執行損失驗證")
    
    total_loss = 0.0
    task_losses = {
        'promise_loss': 0.0,
        'promise_span_loss': 0.0,
        'evidence_loss': 0.0,
        'evidence_span_loss': 0.0,
        'esg_loss': 0.0,
        'timeline_loss': 0.0
    }
    
    with torch.no_grad():
        for batch in val_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                token_type_ids=batch['token_type_ids']
            )
            
            loss, loss_breakdown = model.compute_loss(outputs, batch)
            
            total_loss += loss.item()
            for k in task_losses:
                if k in loss_breakdown:
                    task_losses[k] += loss_breakdown[k]
    
    # ========== 顯示結果 ==========
    avg_loss = total_loss / len(val_dataloader)
    print(f"\n平均總驗證損失: {avg_loss:.4f}")
    
    print_header("評估完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG 永續承諾驗證競賽 - 驗證評估腳本")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt", help="模型檢查點路徑")
    parser.add_argument("--data", default="data/vpesg4k_train_1000 V1.json", help="資料路徑")
    parser.add_argument("--batch-size", type=int, default=8)
    
    args = parser.parse_args()
    
    evaluate_model(
        checkpoint_path=args.checkpoint,
        json_file=args.data,
        batch_size=args.batch_size
    )
