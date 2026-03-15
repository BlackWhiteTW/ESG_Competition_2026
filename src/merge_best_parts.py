import torch
import os
import glob
from collections import OrderedDict

def merge_models(fold_dir="models/checkpoints", output_path="models/checkpoints/final_stitched_model.pt"):
    """
    將 K-Fold 訓練出的多個模型權重進行平均 (Weight Averaging)。
    這能顯著提升泛化能力，且推論速度與單一模型相同。
    """
    checkpoint_paths = glob.glob(os.path.join(fold_dir, "fold_*/best_model.pt"))
    
    if not checkpoint_paths:
        print(f"[ERROR] 找不到任何 fold_X/best_model.pt 檔案在 {fold_dir}")
        return

    print(f"\n[START] 準備融合 {len(checkpoint_paths)} 個模型...")
    for path in checkpoint_paths:
        print(f" - 載入: {path}")

    # 讀取第一個模型作為基準
    base_state = torch.load(checkpoint_paths[0], map_location="cpu")
    merged_state_dict = base_state['model_state_dict']
    
    # 初始化累加
    for key in merged_state_dict.keys():
        merged_state_dict[key] = merged_state_dict[key].float()

    # 累加其餘模型
    for i in range(1, len(checkpoint_paths)):
        current_state = torch.load(checkpoint_paths[i], map_location="cpu")['model_state_dict']
        for key in merged_state_dict.keys():
            merged_state_dict[key] += current_state[key].float()

    # 計算平均
    num_models = len(checkpoint_paths)
    for key in merged_state_dict.keys():
        merged_state_dict[key] = (merged_state_dict[key] / num_models).half() # 轉回 FP16 節省空間

    # 封裝成官方格式
    final_checkpoint = {
        'model_state_dict': merged_state_dict,
        'metadata': {
            'merged_folds': len(checkpoint_paths),
            'merge_method': 'Simple Weight Averaging',
            'base_model': 'hfl/chinese-roberta-wwm-ext-large'
        }
    }

    # 儲存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_checkpoint, output_path)
    print(f"\n" + "="*80)
    print(f"[SUCCESS] 終極融合模型已生成！")
    print(f"位置: {output_path}")
    print(f"大小: {os.path.getsize(output_path) / 1e9:.2f} GB")
    print("="*80)

if __name__ == "__main__":
    merge_models()
