import requests
from lxml import html
import sys
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from collections import Counter 
import re

def get_all_review_titles_by_xpath(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        print(f"Step 1: Requesting URL: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        print("Step 2: Successfully fetched content. Parsing HTML.")
        tree = html.fromstring(response.content)
        
        relative_xpath = "//article[contains(@class, 'user-review-item')]//h3"
        print(f"Step 3: Executing relative XPath query: {relative_xpath}")
        title_elements = tree.xpath(relative_xpath)
        
        if not title_elements:
            title_elements = tree.xpath("//article//h3")
            if title_elements:
                print("    (Using fallback generic XPath query.)")
            else:
                print("    Warning: No title elements found.")
                return []
        
        titles = []
        for element in title_elements:
            titles.append(element.text_content().strip())
        
        return titles

    except requests.exceptions.RequestException as err:
        print(f" Request error: {err}", file=sys.stderr)
        return []
    except Exception as e:
        print(f" Unknown error: {e}", file=sys.stderr)
        return []

if __name__ == "__main__":
    target_url = "https://www.imdb.com/title/tt6483832/reviews/?ref_=tt_ov_ururv"
    print("--- IMDb Review Titles Scraper ---")
    all_titles = get_all_review_titles_by_xpath(target_url)
    print("--------------------------------------")
    
    if all_titles:
        print(f"Total {len(all_titles)} review titles fetched.")
        print("\n**First 10 Review Titles:**")
        titles_to_display = all_titles[:10] 
        for i, title in enumerate(titles_to_display):
            print(f"  {i+1}. **{title}**")
        if len(all_titles) > 10:
            print("\n... More reviews fetched successfully.")
    else:
        print("No review titles were fetched.")
    print("--------------------------------------")

    def ensure_nltk_data():
        try:
            nltk.data.find('sentiment/vader_lexicon.zip')
        except nltk.downloader.DownloadError:
            print("Downloading 'vader_lexicon' resource...")
            nltk.download('vader_lexicon')
        except LookupError:
            print("NLTK data path lookup failed. Downloading 'vader_lexicon'...")
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

    ensure_nltk_data()
    sid = SentimentIntensityAnalyzer()
    initial_history = run_batch_analysis(sid, all_titles)
    run_interactive_analyzer(sid, initial_history)
