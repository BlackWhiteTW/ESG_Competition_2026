"""
ESG 永續承諾驗證競賽 - 訓練腳本
"""

import os
import sys
import argparse
import glob
import re
import torch
from datetime import datetime

# 將 src 目錄加入 Python 路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from dataset import ESGDataset
from model import ESGMultiTaskModel
from train import ESGTrainer, create_data_splits


def find_latest_epoch_checkpoint(checkpoint_dir: str = "models/checkpoints") -> str:
    """尋找最新的 epoch 檢查點（checkpoint_epoch_X_step_Y.pt）。"""
    pattern = os.path.join(checkpoint_dir, "checkpoint_epoch_*_step_*.pt")
    candidates = glob.glob(pattern)
    if not candidates:
        return ""

    def sort_key(path: str):
        filename = os.path.basename(path)
        match = re.match(r"checkpoint_epoch_(\d+)_step_(\d+)\.pt", filename)
        if not match:
            return (-1, -1)
        epoch = int(match.group(1))
        step = int(match.group(2))
        return (epoch, step)

    return max(candidates, key=sort_key)


def print_header(message: str):
    """印出格式化的標題"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


def print_system_info():
    """印出系統資訊"""
    print("\n" + "=" * 80)
    print("📊 系統資訊")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"計算設備: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"PyTorch 版本: {torch.__version__}")
    print("=" * 80)


def train_model(
    resume_checkpoint: str = None,
    json_file: str = "data/vpesg4k_train_1000 V1.json",
    num_epochs: int = 16,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 2,
    warmup_ratio: float = 0.08,
    early_stopping_patience: int = 3,
    min_delta: float = 0.0,
    train_scope: str = "full-model"
):
    """訓練模型主函式"""
    print_header("🚀 開始訓練流程")
    print_system_info()
    
    # ========== 載入資料 ==========
    print_header("📚 載入訓練資料")
    
    if not os.path.exists(json_file):
        print(f"❌ 錯誤：找不到訓練資料: {json_file}")
        return
    
    dataset = ESGDataset(
        json_file=json_file,
        model_name="hfl/chinese-roberta-wwm-ext",
        max_length=256,
        debug=True
    )
    
    print(f"✅ 資料集已載入: {len(dataset)} 筆樣本")
    
    # ========== 建立資料加載器 ==========
    print_header("⚙️  建立資料加載器")
    
    train_dataloader, val_dataloader = create_data_splits(
        dataset,
        train_ratio=0.8,
        batch_size=batch_size,
        seed=42
    )
    
    # ========== 建立模型 ==========
    print_header("🧠 初始化模型")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ESGMultiTaskModel(
        model_name="hfl/chinese-roberta-wwm-ext",
        dropout_rate=0.1
    )
    
    # ========== 建立訓練器 ==========
    print_header("⚡ 初始化訓練器")
    
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
        checkpoint_dir="models/checkpoints"
    )
    
    # ========== 開始訓練 ==========
    print_header("🔥 開始訓練")
    best_checkpoint = os.path.join(trainer.checkpoint_dir, "best_model.pt")
    
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"\n⏮️  從檢查點恢復訓練: {resume_checkpoint}\n")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        # 如果是 fresh 模式，resume_checkpoint 會是 None
        print(f"\n✨ 從頭開始新的訓練流程...")
        trainer.train(resume_from_checkpoint=None)
    
    print_header("✅ 訓練完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESG 永續承諾驗證競賽 - 訓練腳本")
    parser.add_argument("mode", choices=["fresh", "resume", "auto"], default="auto", nargs="?", help="訓練模式")
    parser.add_argument("--checkpoint", default="models/checkpoints/best_model.pt", help="檢查點路徑")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--train-scope", choices=["full-model", "heads-only"], default="full-model")
    
    args = parser.parse_args()
    
    actual_mode = args.mode
    checkpoint_to_use = args.checkpoint
    
    if actual_mode == "auto":
        latest = find_latest_epoch_checkpoint()
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
        train_scope=args.train_scope
    )
