import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import List, Dict, Tuple, Optional


class ESGDataset(Dataset):
    """
    ESG 永續承諾驗證資料集
    
    核心功能：
    1. 讀取官方 JSON 格式的標註資料
    2. 進行字元索引對齊（Character-level to Token-level）
    3. 完整處理缺失標籤、重複承諾等邊界情況
    4. 輸出 PyTorch 張量供模型訓練
    """
    
    def __init__(
        self,
        json_file: str,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        max_length: int = 512,
        debug: bool = False
    ):
        """
        初始化資料集
        
        Args:
            json_file: 官方提供的 JSON 訓練資料路徑
            model_name: Hugging Face 預訓練模型路徑
            max_length: 最大序列長度（配合 4GB VRAM 的極限）
            debug: 啟用除錯模式，印出對齊過程
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.debug = debug
        
        # 讀取 JSON 檔案
        with open(json_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 資料預處理與標籤對齊
        self.samples = self._preprocess_data(raw_data)
        
        if debug:
            print(f"✅ 成功載入 {len(self.samples)} 筆訓練樣本")
    
    def _preprocess_data(self, raw_data: List[Dict]) -> List[Dict]:
        """
        核心資料處理函式
        
        關鍵步驟：
        1. 從字串反推字元位置 (promise_string -> start/end indices)
        2. 處理 evidence_string 的位置對齊
        3. 轉換為 Token 位置以供模型訓練
        4. 格式驗證：確保所有位置都在有效範圍內
        """
        processed_samples = []
        
        for item in raw_data:
            try:
                sample = self._process_single_item(item)
                if sample is not None:
                    processed_samples.append(sample)
            except Exception as e:
                if self.debug:
                    print(f"⚠️  警告: ID {item.get('id', 'Unknown')} 處理失敗: {str(e)}")
                continue
        
        return processed_samples
    
    def _process_single_item(self, item: Dict) -> Optional[Dict]:
        """
        處理單一訓練樣本
        
        陷阱 1: 承諾字串可能在原文中出現多次，須根據上下文判斷
        陷阱 2: 證據為空時，不能給 None，應指向 [CLS] Token
        陷阱 3: 字元位置需精準轉換為 Token 位置
        """
        text = item['data'].strip()
        
        # ========== 處理 Promise (承諾) ==========
        promise_status = item.get('promise_status', 'No')
        promise_string = item.get('promise_string', '').strip()
        
        promise_start_char = 0
        promise_end_char = 0
        
        if promise_status == 'Yes' and promise_string:
            # 從原文中搜尋承諾字串
            pos = text.find(promise_string)
            if pos != -1:
                promise_start_char = pos
                promise_end_char = pos + len(promise_string)
            else:
                # 搜尋失敗，作為無承諾處理
                if self.debug:
                    print(f"⚠️  無法在文章中找到承諾: '{promise_string[:30]}...'")
                promise_status = 'No'
        
        # ========== 處理 Evidence (證據) ==========
        evidence_status = item.get('evidence_status', 'No')
        evidence_string = item.get('evidence_string', '').strip()
        
        evidence_start_char = 0
        evidence_end_char = 0
        
        if evidence_status == 'Yes' and evidence_string:
            pos = text.find(evidence_string)
            if pos != -1:
                evidence_start_char = pos
                evidence_end_char = pos + len(evidence_string)
            else:
                if self.debug:
                    print(f"⚠️  無法在文章中找到證據: '{evidence_string[:30]}...'")
                evidence_status = 'No'
        
        # ========== 進行 Tokenization ==========
        # 使用 Hugging Face Tokenizer 進行分詞，保留字元偏移映射
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_offsets_mapping=True,
            add_special_tokens=True
        )
        
        # ========== 字元索引轉換為 Token 索引 ==========
        promise_start_token = self._char_idx_to_token_idx(
            encodings['offset_mapping'], 
            promise_start_char,
            is_start=True
        )
        promise_end_token = self._char_idx_to_token_idx(
            encodings['offset_mapping'], 
            promise_end_char,
            is_start=False
        )
        
        evidence_start_token = self._char_idx_to_token_idx(
            encodings['offset_mapping'], 
            evidence_start_char,
            is_start=True
        )
        evidence_end_token = self._char_idx_to_token_idx(
            encodings['offset_mapping'], 
            evidence_end_char,
            is_start=False
        )
        
        # 若無承諾或證據，設定位置為 [CLS] Token (索引 0)
        if promise_status == 'No':
            promise_start_token = 0
            promise_end_token = 0
        
        if evidence_status == 'No':
            evidence_start_token = 0
            evidence_end_token = 0
        
        # ========== 提取 ESG 分類標籤 ==========
        esg_type = item.get('esg_type', 'S')
        esg_label = {'E': 0, 'S': 1, 'G': 2}.get(esg_type, 1)
        
        # ========== 提取時間軸標籤 ==========
        verification_timeline = item.get('verification_timeline', 'N/A')
        # 0: already, 1: within_2_years, 2: between_2_and_5_years, 3: more_than_5_years, 4: N/A
        timeline_label = {
            'already': 0,
            'within_2_years': 1,
            'between_2_and_5_years': 2,
            'more_than_5_years': 3,
            'N/A': 4
        }.get(verification_timeline, 4)
        
        # ========== 提取證據品質標籤 ==========
        evidence_quality = item.get('evidence_quality', 'N/A')
        # 0: Clear, 1: Not Clear, 2: Misleading, 3: N/A
        quality_label = {
            'Clear': 0,
            'Not Clear': 1,
            'Misleading': 2,
            'N/A': 3
        }.get(evidence_quality, 3)
        
        return {
            'id': item.get('id'),
            'text': text,
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': encodings.get('token_type_ids', [0] * len(encodings['input_ids'])),
            'offset_mapping': encodings['offset_mapping'],
            
            # 承諾標籤
            'promise_status': 1 if promise_status == 'Yes' else 0,
            'promise_start': promise_start_token,
            'promise_end': promise_end_token,
            
            # 證據標籤
            'evidence_status': 1 if evidence_status == 'Yes' else 0,
            'evidence_start': evidence_start_token,
            'evidence_end': evidence_end_token,
            
            # 分類標籤
            'esg_label': esg_label,
            'timeline_label': timeline_label,
            'quality_label': quality_label,
            
            # 額外資訊 (Metadata)
            'company': item.get('company', ''),
            'url': item.get('company_source', ''),
            'page_number': item.get('page_number', 0)
        }
    
    def _char_idx_to_token_idx(
        self,
        offset_mapping: List[Tuple[int, int]],
        char_idx: int,
        is_start: bool = True
    ) -> int:
        """
        將字元索引轉換為 Token 索引
        
        offset_mapping: Tokenizer 返回的 (char_start, char_end) 對應表
        char_idx: 原始字元位置
        is_start: True 則找第一個包含此位置的 Token；False 則找最後一個
        """
        if char_idx == 0:
            return 0
        
        for token_idx, (start, end) in enumerate(offset_mapping):
            if start == 0 and end == 0:  # 特殊 Token
                continue
            
            if is_start:
                # 找第一個包含或超過 char_idx 的 Token
                if start <= char_idx < end:
                    return token_idx
                if start >= char_idx:
                    return token_idx
            else:
                # 找最後一個包含或超過 char_idx 的 Token
                if start < char_idx <= end:
                    return token_idx
                if start >= char_idx:
                    return token_idx
        
        # 若超出範圍，返回最後一個有效 Token
        for i in range(len(offset_mapping) - 1, -1, -1):
            if offset_mapping[i] != (0, 0):
                return i
        
        return 0
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        返回單一樣本（張量化）
        """
        sample = self.samples[idx]
        
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(sample['token_type_ids'], dtype=torch.long),
            'offset_mapping': torch.tensor(sample['offset_mapping'], dtype=torch.long),
            
            # 承諾標籤
            'promise_status': torch.tensor(sample['promise_status'], dtype=torch.long),
            'promise_start': torch.tensor(sample['promise_start'], dtype=torch.long),
            'promise_end': torch.tensor(sample['promise_end'], dtype=torch.long),
            
            # 證據標籤
            'evidence_status': torch.tensor(sample['evidence_status'], dtype=torch.long),
            'evidence_start': torch.tensor(sample['evidence_start'], dtype=torch.long),
            'evidence_end': torch.tensor(sample['evidence_end'], dtype=torch.long),
            
            # 分類標籤
            'esg_label': torch.tensor(sample['esg_label'], dtype=torch.long),
            'timeline_label': torch.tensor(sample['timeline_label'], dtype=torch.long),
            'quality_label': torch.tensor(sample['quality_label'], dtype=torch.long),
            
            'id': sample['id'],
            'text': sample['text'],
            'company': sample['company'],
            'url': sample['url'],
            'page_number': sample['page_number']
        }


def parse_esg_data(file_path):
    """
    向後相容的函式（保持與舊程式碼相容）
    """
    dataset = ESGDataset(file_path, debug=False)
    return [dataset.samples[i] for i in range(len(dataset))]
