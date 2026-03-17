"""
K-Fold 交叉驗證訓練腳本
"""

import os
import sys
import argparse

# 將 src 目錄加入 Python 路徑
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from dataset import ESGDataset
from model import ESGMultiTaskModel
from train import ESGTrainer, create_kfold_splits
from utils import print_header, print_system_info, find_latest_epoch_checkpoint
import config
import torch


def train_kfold():
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()
    parser = argparse.ArgumentParser(description="ESG 競賽 - K-Fold 訓練器")
    parser.add_argument("--folds", type=int, default=5, help="總折數")
    parser.add_argument("--epochs", type=int, default=20, help="總訓練輪數")
    parser.add_argument(
        "--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE, help="批次大小"
    )
    parser.add_argument("--lr", type=float, default=config.DEFAULT_LR, help="學習率")
    parser.add_argument("--data", default=config.DATA_PATH)

    args = parser.parse_args()

    # 1. 偵測是否有現有的進度
    latest_ckpt = find_latest_epoch_checkpoint(config.MODELS_DIR)
    start_fold = 0
    resume_ckpt_path = None

    if latest_ckpt:
        checkpoint = torch.load(latest_ckpt, map_location="cpu", weights_only=False)
        start_fold = checkpoint.get("fold_idx", 0)
        resume_ckpt_path = latest_ckpt
        print(f"[RESUME] 偵測到現有進度：從 Fold {start_fold + 1} 開始恢復。")

    # 載入全量資料集
    model_name = config.MODEL_NAME

    dataset = ESGDataset(
        json_file=args.data,
        model_name=model_name,
        max_length=config.MAX_SEQ_LENGTH,
        debug=False,
    )

    print_header(f"啟動 {args.folds}-Fold 訓練流程")
    print(f"模式: {model_name} | 學習率: {args.lr} | 早停耐心: 5")

    for fold in range(start_fold, args.folds):
        print_header(f"開始訓練 Fold {fold + 1}/{args.folds}")

        # 1. 取得該折資料
        train_loader, val_loader = create_kfold_splits(
            dataset, num_folds=args.folds, fold_idx=fold, batch_size=args.batch_size
        )

        # 2. 初始化全新模型
        model = ESGMultiTaskModel(model_name=model_name)

        # 3. 建立訓練器
        trainer = ESGTrainer(
            model=model,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            num_epochs=args.epochs,
            learning_rate=args.lr,
            checkpoint_dir=config.MODELS_DIR,
            early_stopping_patience=5,
            fold_idx=fold,
        )

        # 4. 執行訓練 (如果是第一折且有恢復路徑，則載入權重)
        if fold == start_fold and resume_ckpt_path:
            trainer.train(resume_from_checkpoint=resume_ckpt_path)
            # 使用完畢後清除恢復路徑，後續 Fold 仍需從頭訓練
            resume_ckpt_path = None
        else:
            trainer.train(resume_from_checkpoint=None)

    print_header(f"所有 {args.folds} 折訓練已完成！")
    print(f"模型已存放於 {config.MODELS_DIR}/")


if __name__ == "__main__":
    train_kfold()
