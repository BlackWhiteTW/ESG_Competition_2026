import os

# ==============================================================================
# 檔案路徑指揮中心 (Path Center)
# ==============================================================================

# 專案根目錄
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# 1. 資料相關
DATA_PATH = os.path.join(BASE_DIR, "data", "vpesg4k_train_1000 V1.json")

# 2. 模型相關
MODEL_NAME = "hfl/chinese-roberta-wwm-ext-large"
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 預設的最佳模型 (Stitched / Ensemble)
# 預設使用 final_stitched_model.pt
DEFAULT_CHECKPOINT = os.path.join(MODELS_DIR, "final_stitched_model.pt")

# K-Fold 產出的個別模型路徑
FOLD_CHECKPOINT_PATTERN = os.path.join(MODELS_DIR, "best_model_*.pt")

# 3. 輸出相關
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
BEST_THRESHOLDS_PATH = os.path.join(OUTPUTS_DIR, "best_thresholds.json")

# 4. 參數設定 (超參數)
MAX_SEQ_LENGTH = 256
DEFAULT_BATCH_SIZE = 4
DEFAULT_LR = 2e-5
DEFAULT_EPOCHS = 20
DEFAULT_FOLDS = 10
DEFAULT_GRADIENT_ACCUMULATION_STEPS = 8

# 確保目錄存在
for d in [MODELS_DIR, OUTPUTS_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)
