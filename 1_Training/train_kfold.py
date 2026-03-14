"""
ESG 競賽 - K-Fold 交叉驗證訓練腳本
"""

import os
import sys
import argparse
import torch
from datetime import datetime

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from dataset import ESGDataset
from model import ESGMultiTaskModel
from train import ESGTrainer, create_kfold_splits

def train_kfold():
    parser = argparse.ArgumentParser(description="ESG 競賽 - K-Fold 訓練器")
    parser.add_argument("--folds", type=int, default=5, help="總折數")
    parser.add_argument("--epochs", type=int, default=20, help="總訓練輪數")
    parser.add_argument("--batch-size", type=int, default=2, help="批次大小 (Large 模型建議 2)")
    parser.add_argument("--lr", type=float, default=1e-5, help="學習率 (建議 1e-5)")
    parser.add_argument("--data", default="data/vpesg4k_train_1000 V1.json")
    
    args = parser.parse_args()
    
    # 載入全量資料集
    # 確保這裡使用與 model.py 一致的 Large 模型
    model_name = "hfl/chinese-roberta-wwm-ext-large"
    
    dataset = ESGDataset(
        json_file=args.data,
        model_name=model_name,
        max_length=256,
        debug=False
    )
    
    print(f"\n[START] 啟動 {args.folds}-Fold 訓練流程...")
    print(f"模式: Large 模型 | 學習率: {args.lr} | 早停耐心: 5")
    
    for fold in range(args.folds):
        print(f"\n" + "="*80)
        print(f"[FOLD] 開始訓練 Fold {fold+1}/{args.folds}")
        print("="*80)
        
        # 1. 取得該折資料
        train_loader, val_loader = create_kfold_splits(
            dataset, 
            num_folds=args.folds, 
            fold_idx=fold, 
            batch_size=args.batch_size
        )
        
        # 2. 初始化全新模型 (自動對齊 hidden_size)
        model = ESGMultiTaskModel(model_name=model_name)
        
        # 3. 建立訓練器
        fold_checkpoint_dir = f"models/checkpoints/fold_{fold}"
        trainer = ESGTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            checkpoint_dir=fold_checkpoint_dir,
            early_stopping_patience=5  # 給予更多耐心
        )
        
        # 4. 執行訓練
        trainer.train(resume_from_checkpoint=None)
        
    print(f"\n" + "="*80)
    print(f"[SUCCESS] 所有 {args.folds} 折訓練已完成！")
    print(f"模型已存放於 models/checkpoints/fold_X/")
    print("=" * 80)

if __name__ == "__main__":
    train_kfold()
