import re
from typing import Dict, List, Tuple

class ESGKeywordAnalyzer:
    """
    基於 MSCI ESG 框架 (ACL 2023 論文標準) 的關鍵字分析器
    用於輔助判定文本屬於 E, S 還是 G。
    """
    
    # 定義三大支柱的關鍵字字典 (包含繁體中文、簡體中文與英文關鍵字)
    KEYWORDS = {
        'E': {
            'themes': ['氣候變遷', '自然資本', '污染與廢棄物', '環境機會'],
            'keywords': [
                '碳排放', '淨零', '減碳', '氣候', '再生能源', '綠能', '太陽能', '風力', 
                '水資源', '生物多樣性', '土地利用', '有害廢棄物', '回收', '包裝', '環保', 
                '清潔技術', '綠建築', '碳足跡', '能源效率', '節能', '低碳', '排放量',
                '廢水', '空汙', '廢棄物管理', '資源效率', '脫碳', '甲烷'
            ]
        },
        'S': {
            'themes': ['人力資本', '產品責任', '利益相關者', '社會機會'],
            'keywords': [
                '勞工', '員工', '人才', '健康', '安全', '職安', '培訓', '福利', '產假', 
                '薪資', '人權', '供應鏈', '性別平等', '多元化', '包容', '社區', '捐贈', 
                '公益', '產品安全', '隱私', '數據安全', '個資', '消費者保護', '普及化', 
                '育兒', '職涯', '退休金', '勞資糾紛', '社區關係', '社會責任'
            ]
        },
        'G': {
            'themes': ['公司治理', '企業行為'],
            'keywords': [
                '董事會', '高管', '薪酬', '審計', '所有權', '控制權', '誠信', '道德', 
                '商業行為', '反競爭', '稅務', '透明度', '腐敗', '賄賂', '風險管理', 
                '永續報告書', '資訊揭露', '股東權益', '內部控制', '法令遵循', '防制洗錢', 
                '數位轉型', '策略規劃', '願景', '使命', '卓越服務', '永續發展策略'
            ]
        }
    }

    @classmethod
    def analyze_text(cls, text: str) -> Dict[str, any]:
        """
        分析文本中出現的 ESG 關鍵字頻率與權重。
        """
        results = {
            'E': {'count': 0, 'matches': []},
            'S': {'count': 0, 'matches': []},
            'G': {'count': 0, 'matches': []}
        }
        
        # 進行關鍵字匹配
        for pillar, data in cls.KEYWORDS.items():
            for word in data['keywords']:
                # 使用正則表達式進行全詞或片段匹配
                matches = re.findall(word, text, re.IGNORECASE)
                if matches:
                    results[pillar]['count'] += len(matches)
                    results[pillar]['matches'].append(word)
        
        # 計算建議類型
        counts = {p: results[p]['count'] for p in ['E', 'S', 'G']}
        max_count = max(counts.values())
        
        # 如果沒有匹配到任何關鍵字，預設為 S (社會責任通常最廣泛)
        if max_count == 0:
            suggested_type = 'S'
            confidence = 0.0
        else:
            # 取得最高計分的類別 (若平手則回傳多個)
            suggested_types = [p for p, c in counts.items() if c == max_count]
            suggested_type = suggested_types[0]
            confidence = max_count / sum(counts.values())
            
        return {
            'scores': counts,
            'suggested_type': suggested_type,
            'confidence': round(confidence, 2),
            'detailed_matches': results
        }

if __name__ == "__main__":
    # 測試範例 1: 聯發科技產假 (S)
    test_text_1 = "聯發科技自 2024 年起提供女性員工在分娩前後計有 12 週共 84 天的產假，兩者合計共有 10 天陪產假可運用。"
    
    # 測試範例 2: 台泥減碳 (E)
    test_text_2 = "本公司致力於推動低碳轉型，目標於 2030 年減少 30% 碳排放量，並全面導入再生能源使用。"
    
    # 測試範例 3: 萬海策略 (G)
    test_text_3 = "萬海將持續落實 ESG 策略，強化董事會監督職能，並秉持誠信經營之核心價值。"

    for i, t in enumerate([test_text_1, test_text_2, test_text_3]):
        print(f"\n測試範例 {i+1}: {t[:30]}...")
        analysis = ESGKeywordAnalyzer.analyze_text(t)
        print(f"建議類別: {analysis['suggested_type']} (信心度: {analysis['confidence']})")
        print(f"各項分數: {analysis['scores']}")
        print(f"匹配關鍵字: {analysis['detailed_matches'][analysis['suggested_type']]['matches']}")
