import torch
import os
import glob
import re


def print_header(message: str):
    """
    工具函式庫，包含系統資訊列印與檢查點管理等功能。
    """
    print("\n" + "=" * 80)
    print(f"  {message}")
    print("=" * 80)


# 偵測並印出當前執行環境的系統資訊。
def print_system_info():
    """
    偵測並印出當前執行環境的系統資訊。
    """
    print_header("系統資訊")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"計算設備: {device}")

    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.2f} GB")

    print(f"PyTorch 版本: {torch.__version__}")


def find_latest_epoch_checkpoint(checkpoint_dir: str = "models") -> str:
    """
    自動掃描指定目錄及其子目錄，尋找最新生成的訓練檢查點檔案。
    檔名格式：checkpoint_kfold_X_epoch_Y.pt

    Args:
        checkpoint_dir (str): 存放檢查點的目錄路徑。

    Returns:
        str: 最新檢查點的完整檔案路徑。若找不到任何符合的檔案，則傳回空字串。
    """
    # [優化] 使用遞迴搜尋所有子目錄中的檢查點
    pattern = os.path.join(checkpoint_dir, "**", "checkpoint_kfold_*_epoch_*.pt")
    candidates = glob.glob(pattern, recursive=True)

    if not candidates:
        return ""

    def sort_key(path: str):
        """
        解析檔名中的 KFold 與 Epoch 數值，用於正確排序。
        """
        filename = os.path.basename(path)
        match = re.match(r"checkpoint_kfold_(\d+)_epoch_(\d+)\.pt", filename)
        if not match:
            return (-1, -1)
        kfold = int(match.group(1))
        epoch = int(match.group(2))
        return (kfold, epoch)

    # 傳回排序後的最大值 (即序號最大的 KFold 與 Epoch)
    return max(candidates, key=sort_key)
