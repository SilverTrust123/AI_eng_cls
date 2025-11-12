import time
import sys
import nltk
import re
from collections import Counter
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from lxml import html
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def scrape_all_titles_with_see_all(url, max_reviews=500):
    LOAD_MORE_XPATH = "//button[contains(., 'more')]" 
    SEE_ALL_XPATH = "//button[contains(., 'all')]" 
    REVIEW_CARD_XPATH = "//article[contains(@class, 'user-review-item')]"
    TITLE_XPATH = "//article//h3"
    
    print("Step 1: Launching browser and loading page...")
    try:
        service = Service(ChromeDriverManager().install())
        options = webdriver.ChromeOptions()
        options.add_argument('--log-level=3') 
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(url)
    except Exception as e:
        print(f"❌ Failed to launch browser. Error: {e}", file=sys.stderr)
        return []
    
    titles = []
    current_review_count = 0
    wait = WebDriverWait(driver, 15)

    try:
        print("Step 2: Trying to click 'See all' button...")
        try:
            see_all_button = wait.until(EC.element_to_be_clickable((By.XPATH, SEE_ALL_XPATH)))
            see_all_button.click()
            print("Step 2: 'See all' clicked successfully. Waiting for update...")
            time.sleep(5)
        except Exception:
            print("Step 2: 'See all' not found or failed to click. Skipping.")
            pass

        print("Step 3: Entering main loop to load more reviews...")
        while current_review_count < max_reviews:
            review_elements = driver.find_elements(By.XPATH, REVIEW_CARD_XPATH)
            new_review_count = len(review_elements)
            
            if new_review_count <= current_review_count and new_review_count > 0:
                print(f"Step 3: Review count ({new_review_count}) stopped increasing.")
                break
            
            if new_review_count > 0:
                print(f"Step 3: Loaded {new_review_count} reviews so far. Trying to load more...")
                current_review_count = new_review_count
                try:
                    last_review_element = review_elements[-1]
                    driver.execute_script("arguments[0].scrollIntoView(true);", last_review_element) 
                    time.sleep(2) 
                except IndexError:
                    pass 

            try:
                load_more_button = wait.until(EC.element_to_be_clickable((By.XPATH, LOAD_MORE_XPATH)))
                load_more_button.click()
                time.sleep(4) 
            except Exception:
                print(f"Step 3: No more 'Load More' button. Total loaded: {current_review_count}.")
                break
        
        print("\nStep 4: Parsing HTML and extracting titles...")
        tree = html.fromstring(driver.page_source)
        title_elements = tree.xpath(TITLE_XPATH)
        for element in title_elements:
            titles.append(element.text_content().strip())
        
        print(f"Step 5: Extraction complete. {len(titles)} titles found. Closing browser.")
        return titles

    except Exception as e:
        print(f"❌ Unexpected error: {e}", file=sys.stderr)
        return []
    finally:
        driver.quit()

def ensure_nltk_data():
    try:
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        print("Downloading 'vader_lexicon' resource...")
        nltk.download('vader_lexicon')
    except LookupError:
        print("Downloading 'vader_lexicon'...")
        nltk.download('vader_lexicon')

def preprocess_review(review):
    review = re.sub(r'^\s*[\d\.\)]+\s*(\*\*|\*|)\s*', '', review).strip()
    review = re.sub(r'[\*\`\#]+', '', review)
    return review.strip()

def calculate_and_print_summary(history, title="COMPREHENSIVE MOVIE REVIEW SUMMARY"):
    if not history:
        print(f"\n[ {title} ]")
        print("No reviews entered yet.")
        print("-" * 60)
        return

    counts = Counter(history)
    total = len(history)
    pos_count = counts.get('Positive', 0)
    neg_count = counts.get('Negative', 0)
    neu_count = counts.get('Neutral', 0)

    if pos_count > neg_count and pos_count > neu_count:
        overall_sentiment = "OVERALL POSITIVE (Strongly Recommended)"
    elif neg_count > pos_count and neg_count > neu_count:
        overall_sentiment = "OVERALL NEGATIVE (Not Recommended)"
    else:
        overall_sentiment = "OVERALL MIXED/NEUTRAL (Proceed with Caution)"

    print("\n" + "=" * 60)
    print(f"{title:^60}")
    print("=" * 60)
    print(f"Total Reviews Analyzed: {total}")
    print(f"  - Positive Reviews: {pos_count} ({pos_count/total:.1%})")
    print(f"  - Negative Reviews: {neg_count} ({neg_count/total:.1%})")
    print(f"  - Neutral Reviews:  {neu_count} ({neu_count/total:.1%})")
    print("-" * 60)
    print(f"Movie Assessment: {overall_sentiment}")
    print("=" * 60)

def run_batch_analysis(sid, reviews):
    batch_history = []
    print("\n" + "#" * 60)
    print("           AUTOMATIC BATCH ANALYSIS STARTED             ")
    print("#" * 60)
    
    for i, raw_review in enumerate(reviews):
        review = preprocess_review(raw_review)
        scores = sid.polarity_scores(review)
        compound_score = scores['compound']
        
        if compound_score >= 0.05:
            sentiment = "Positive"
        elif compound_score <= -0.05:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
        
        batch_history.append(sentiment)
        print(f"Review {i+1} (Processed: '{review[:30]}...'): '{raw_review[:40]}...' -> Classification: {sentiment} (Compound: {compound_score:.4f})")

    calculate_and_print_summary(batch_history, title="BATCH ANALYSIS SUMMARY (Scraped Reviews)")
    return batch_history

def run_interactive_analyzer(sid, initial_history):
    analysis_history = initial_history.copy()
    print("=" * 60)
    print("         INTERACTIVE ANALYSIS MODE ACTIVATED          ")
    print("=" * 60)
    print("Please enter your movie review (English is highly recommended).")
    print("Type 'result' for a summary of all reviews so far.")
    print("Type 'exit' or 'quit' to end the program.")
    print("-" * 60)

    while True:
        review = input(">>> Enter review: ")
        
        if review.lower() == 'result':
            calculate_and_print_summary(analysis_history, title="CUMULATIVE REVIEW SUMMARY")
            continue
        
        if review.lower() in ['exit', 'quit']:
            calculate_and_print_summary(analysis_history, title="FINAL CUMULATIVE SUMMARY")
            print("-" * 60)
            print("Thank you for using the analyzer. Program terminated.")
            break

        if not review.strip():
            continue

        try:
            cleaned_review = preprocess_review(review)
            scores = sid.polarity_scores(cleaned_review)
            compound_score = scores['compound']
            
            if compound_score >= 0.05:
                sentiment = "Positive"
            elif compound_score <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            analysis_history.append(sentiment)

            print(f"\n[ Analysis Result ]")
            print(f"  Sentiment Classification: {sentiment}")
            print(f"  VADER Detailed Scores (Neg/Neu/Pos): {scores['neg']:.4f}, {scores['neu']:.4f}, {scores['pos']:.4f}")
            print(f"  Compound Score: {compound_score:.4f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            print("-" * 60)

if __name__ == "__main__":
    target_url = "https://www.imdb.com/title/tt7817340/reviews/?ref_=tt_ov_ururv"
    print("--- IMDb All Review Titles Scraper (See all -> Load More Strategy) ---")
    all_titles = scrape_all_titles_with_see_all(target_url, max_reviews=50) 
    print("-------------------------------------------------------")

    if all_titles:
        print(f"Fetched {len(all_titles)} review titles successfully.")
        print("\n**First 10 Review Titles:**")
        for i, title in enumerate(all_titles[:10]):
            print(f"  {i+1}. **{title}**")
    else:
        print("No review titles were fetched.")
    print("-------------------------------------------------------")

    ensure_nltk_data()
    sid = SentimentIntensityAnalyzer()
    initial_history = run_batch_analysis(sid, all_titles)
    run_interactive_analyzer(sid, initial_history)
