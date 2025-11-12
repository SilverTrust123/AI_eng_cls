import requests
from lxml import html
import sys

def get_all_review_titles_by_xpath(url):
    """
    使用 requests 和 lxml 透過相對 XPath 爬取頁面上所有評論的標題。

    :param url: IMDb 評論頁面的完整 URL
    :return: 評論標題列表
    """
    # 設置 User-Agent 模擬瀏覽器請求
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print(f"步驟 1: 正在請求網址: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        print("步驟 2: 成功獲取內容，正在解析 HTML。")
        tree = html.fromstring(response.content)
        
        # 3. 執行更穩定的相對 XPath 查詢
        # 尋找所有 'article' 標籤 (評論卡片的容器) 內部的 'h3' 標籤 (評論標題)
        # 這是最核心的修改，用來取代易碎的絕對路徑
        relative_xpath = "//article[contains(@class, 'user-review-item')]//h3"
        
        # 如果上面的 class 查詢失敗，可以使用更簡單但可能抓到雜訊的：
        # relative_xpath = "//article//h3" 

        print(f"步驟 3: 執行相對 XPath 查詢: {relative_xpath}")
        
        # lxml.xpath 會返回所有匹配的元素列表
        title_elements = tree.xpath(relative_xpath)
        
        if not title_elements:
            # 備用：如果新的 class 結構失效，嘗試更泛用的路徑
            title_elements = tree.xpath("//article//h3")
            if title_elements:
                 print("    （使用了備用泛用 XPath 查詢。）")
            else:
                 print("    ⚠️ 警告：XPath 查詢未找到任何標題元素。")
                 return []
        
        # 4. 提取文字內容
        titles = []
        for element in title_elements:
            titles.append(element.text_content().strip())
        
        return titles

    except requests.exceptions.RequestException as err:
        print(f"❌ 發生請求錯誤: {err}", file=sys.stderr)
        return []
    except Exception as e:
        print(f"❌ 發生未知錯誤: {e}", file=sys.stderr)
        return []

# --- 程式主入口 ---
if __name__ == "__main__":
    # 目標網址
    target_url = "https://www.imdb.com/title/tt0903747/reviews/?ref_=tt_ov_ururv"

    print("--- IMDb 所有評論標題 XPath 爬蟲 ---")
    all_titles = get_all_review_titles_by_xpath(target_url)
    print("--------------------------------------")
    
    if all_titles:
        print(f"總共爬取到 {len(all_titles)} 條評論標題。")
        print("\n**前 10 條評論標題:**")
        
        # 只顯示前 10 條，或者如果不足 10 條則顯示所有
        titles_to_display = all_titles[:100] 
        
        for i, title in enumerate(titles_to_display):
            print(f"  {i+1}. **{title}**")
            
        if len(all_titles) > 10:
            print("\n... 還有更多評論，已成功抓取。")
    else:
        print("未能成功爬取任何評論標題。")
    print("--------------------------------------")