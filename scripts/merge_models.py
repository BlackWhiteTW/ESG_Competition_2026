import torch
import os
import glob


import sys
# 將 src 目錄加入 Python 路徑
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)
import config
from utils import print_system_info

def merge_models(fold_dir=config.MODELS_DIR, output_path=config.DEFAULT_CHECKPOINT):
    # 偵測並印出當前執行環境的系統資訊。
    print_system_info()
    """
    將指定目錄下所有符合模式的模型進行平均融合。
    """
    checkpoint_paths = glob.glob(config.FOLD_CHECKPOINT_PATTERN)
    
    if not checkpoint_paths:
        print(f"[ERROR] 找不到任何模型在 {config.FOLD_CHECKPOINT_PATTERN}")
        return

    # 依照編號排序以確保穩定
    checkpoint_paths.sort()
    print(f"\n[START] 準備融合 {len(checkpoint_paths)} 個模型...")
    for path in checkpoint_paths:
        print(f" - 載入: {path}")

    # 讀取第一個模型作為基準
    base_state = torch.load(checkpoint_paths[0], map_location="cpu", weights_only=False)
    merged_state_dict = base_state["model_state_dict"]

    # 初始化累加
    for key in merged_state_dict.keys():
        merged_state_dict[key] = merged_state_dict[key].float()

    # 累加其餘模型
    for i in range(1, len(checkpoint_paths)):
        current_state = torch.load(
            checkpoint_paths[i], map_location="cpu", weights_only=False
        )["model_state_dict"]
        for key in merged_state_dict.keys():
            merged_state_dict[key] += current_state[key].float()

    # 計算平均
    num_models = len(checkpoint_paths)
    for key in merged_state_dict.keys():
        merged_state_dict[key] = (
            merged_state_dict[key] / num_models
        ).half()  # 轉回 FP16 節省空間

    # 封裝成官方格式
    final_checkpoint = {
        "model_state_dict": merged_state_dict,
        "metadata": {
            "merged_folds": len(checkpoint_paths),
            "merge_method": "Simple Weight Averaging",
            "base_model": config.MODEL_NAME,
        },
    }

    # 儲存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_checkpoint, output_path)
    print("\n" + "=" * 80)
    print("[SUCCESS] 終極融合模型已生成！")
    print(f"位置: {output_path}")
    print(f"大小: {os.path.getsize(output_path) / 1e9:.2f} GB")
    print("=" * 80)


if __name__ == "__main__":
    merge_models()
