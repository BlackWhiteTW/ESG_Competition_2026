"""
訓練腳本
"""

import os
import sys
import argparse
import glob
import re
import torch

# 將 src 目錄加入 Python 路徑
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

from dataset import ESGDataset
from model import ESGMultiTaskModel
from train import ESGTrainer, create_data_splits
from utils import print_header, print_system_info, find_latest_epoch_checkpoint
import config


def train_model(
    resume_checkpoint: str = None,
    json_file: str = config.DATA_PATH,
    num_epochs: int = 16,
    batch_size: int = config.DEFAULT_BATCH_SIZE,
    learning_rate: float = config.DEFAULT_LR,
    gradient_accumulation_steps: int = 2,
    warmup_ratio: float = 0.08,
    early_stopping_patience: int = 5,
    min_delta: float = 0.0,
    train_scope: str = "full-model",
):
    """訓練模型主函式"""
    print_header("開始訓練流程")
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()

    # ========== 載入資料 ==========
    print_header("載入訓練資料")

    if not os.path.exists(json_file):
        print(f"錯誤：找不到訓練資料: {json_file}")
        return

    dataset = ESGDataset(
        json_file=json_file,
        model_name=config.MODEL_NAME,
        max_length=config.MAX_SEQ_LENGTH,
        debug=True,
    )

    print(f"資料集已載入: {len(dataset)} 筆樣本")

    # ========== 建立資料加載器 ==========
    print_header("建立資料加載器")

    train_dataloader, val_dataloader = create_data_splits(
        dataset, train_ratio=0.8, batch_size=batch_size, seed=42
    )

    # ========== 建立模型 ==========
    print_header("初始化模型")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(model_name=config.MODEL_NAME, dropout_rate=0.3)

    # ========== 建立訓練器 ==========
    print_header("初始化訓練器")

    trainer = ESGTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_ratio=warmup_ratio,
        early_stopping_patience=early_stopping_patience,
        min_delta=min_delta,
        train_scope=train_scope,
        device=device,
        checkpoint_dir=config.MODELS_DIR,
    )

    # ========== 開始訓練 ==========
    print_header("開始訓練")
    os.path.join(trainer.checkpoint_dir, "best_model.pt")

    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n從檢查點恢復訓練: {resume_checkpoint}\n")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        # 如果是 fresh 模式，resume_checkpoint 會是 None
        print("\n從頭開始新的訓練流程...")
        trainer.train(resume_from_checkpoint=None)

    print_header("訓練完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG 永續承諾驗證競賽 - 訓練腳本")
    parser.add_argument(
        "mode",
        choices=["fresh", "resume", "auto"],
        default="auto",
        nargs="?",
        help="訓練模式",
    )
    parser.add_argument(
        "--checkpoint", default=config.DEFAULT_CHECKPOINT, help="檢查點路徑"
    )
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--train-scope", choices=["full-model", "heads-only"], default="full-model"
    )

    args = parser.parse_args()

    actual_mode = args.mode
    checkpoint_to_use = args.checkpoint

    if actual_mode == "auto":
        latest = find_latest_epoch_checkpoint(config.MODELS_DIR)
        if latest:
            checkpoint_to_use = latest
            actual_mode = "resume"
        elif os.path.exists(checkpoint_to_use):
            actual_mode = "resume"
        else:
            actual_mode = "fresh"
            checkpoint_to_use = None

    train_model(
        resume_checkpoint=checkpoint_to_use if actual_mode == "resume" else None,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        train_scope=args.train_scope,
    )
