import torch
import os

def print_header(message: str):
    """印出格式化的標題"""
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)

def print_system_info():
    """印出系統資訊"""
    print_header("📊 系統資訊")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"計算設備: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print(f"PyTorch 版本: {torch.__version__}")
    print("=" * 80)
