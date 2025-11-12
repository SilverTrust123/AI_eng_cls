# 導入所需的庫
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os

# --- VADER 情感分析器基於英文詞彙和語法規則 ---
# VADER 模型主要針對英文訓練。輸入中文評論時，其準確度可能會降低。
# 建議您輸入英文評論以獲得最佳分析結果。

def ensure_nltk_data():
    """確保 NLTK VADER 詞典已下載。"""
    try:
        # 檢查 VADER 資源是否已存在
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        print("下載 'vader_lexicon' 資源...")
        nltk.download('vader_lexicon')
    except LookupError:
        # 如果路徑不在搜尋路徑中，再次嘗試下載
        print("NLTK 資料路徑查找失敗，正在下載 'vader_lexicon'...")
        nltk.download('vader_lexicon')

def run_interactive_analyzer():
    """
    執行互動式情感分析迴圈。
    """
    # 確保所需的 NLTK 資源已下載
    ensure_nltk_data()

    # 初始化 VADER 情感分析器
    sid = SentimentIntensityAnalyzer()
    
    print("=" * 60)
    print("      互動式電影評論情感分析器 (基於 NLTK VADER)      ")
    print("=" * 60)
    print("請輸入您的電影評論 (英文為佳)，或輸入 'exit' / 'quit' 結束程式。")
    print("-" * 60)

    while True:
        # 獲取使用者輸入
        review = input(">>> 輸入評論: ")
        
        # 檢查退出指令
        if review.lower() in ['exit', 'quit']:
            print("-" * 60)
            print("感謝使用！程式已結束。")
            break
        
        # 忽略空輸入
        if not review.strip():
            continue

        try:
            # 獲取情感分數
            scores = sid.polarity_scores(review)
            
            # VADER 的 Compound (綜合) 分數
            compound_score = scores['compound']
            
            # 根據 Compound 分數進行分類
            if compound_score >= 0.05:
                sentiment = "正面 (Positive)"
            elif compound_score <= -0.05:
                sentiment = "負面 (Negative)"
            else:
                sentiment = "中性 (Neutral)"
                
            # 輸出結果
            print(f"\n[ 分析結果 ]")
            print(f"  情感分類: {sentiment}")
            print(f"  VADER 詳細分數: {scores}")
            print(f"  綜合分數 (Compound): {compound_score:.4f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"分析時發生錯誤: {e}")
            print("-" * 60)


# 執行互動式分析
if __name__ == "__main__":
    run_interactive_analyzer()