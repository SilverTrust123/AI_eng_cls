# Import necessary libraries
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from collections import Counter 
import re # Added for text preprocessing

# --- VADER Sentiment Analyzer is based on English lexicon and rules ---
# VADER is primarily trained on English text. Its accuracy may decrease 
# significantly when analyzing non-English languages.
# It is highly recommended to input English reviews for the best results.

# BATCH DATA: Simulated list of 10 reviews that the user "scraped"
# NOTE: The content here simulates the raw, uncleaned text that might
# include leading numbers, markdown, etc., which will be cleaned below.
SAMPLE_REVIEWS = [
    "1. **It's ok I guess**", # Example with number and markdown
    "2. Really Great!",
    "3) 99.1% pure gold!",
    "4. The Best movie of the decade, truly spectacular.",
    "5. A complete waste of two hours. Awful and disjointed!",
    "6. The cinematography was breathtaking, but the plot dragged.",
    "7. I have no strong feelings about this movie either way. Perfectly average.",
    "8. Too long and the script was lazy and uninspired.",
    "9. Must see it again. It was fantastic and emotionally resonant.",
    "10. The film was mediocre, neither good nor bad, just forgettable."
]

def ensure_nltk_data():
    """Ensures the NLTK VADER lexicon resource is downloaded."""
    try:
        # Check if the VADER resource already exists
        nltk.data.find('sentiment/vader_lexicon.zip')
    except nltk.downloader.DownloadError:
        print("Downloading 'vader_lexicon' resource...")
        nltk.download('vader_lexicon')
    except LookupError:
        # If the path is not in the search path, try downloading again
        print("NLTK data path lookup failed. Attempting to download 'vader_lexicon'...")
        nltk.download('vader_lexicon')

def preprocess_review(review):
    """
    Cleans a scraped review string by removing common noise like
    leading numbers/bullets and simple Markdown formatting.
    """
    # 1. Remove leading numbers/bullets (e.g., '1. ', '1) ', '- ', '* ', or numbers followed by **)
    # This regex handles leading numbers followed by punctuation or just space, and optional starting markdown
    review = re.sub(r'^\s*[\d\.\)]+\s*(\*\*|\*|)\s*', '', review).strip()
    
    # 2. Remove common markdown/formatting (e.g., **bold**, *italic*, `code`)
    review = re.sub(r'[\*\`\#]+', '', review) # Remove **, *, `, #
    
    # 3. Trim remaining leading/trailing whitespace
    return review.strip()

def get_simulated_streaming_data():
    """
    Simulates fetching top TV series data from a streaming service.
    NOTE: Actual web scraping is restricted in this environment.
    """
    # Simulated data for a hypothetical streaming service (based on a single streaming format)
    data = [
        {"title": "The Quantum Enigma", "genre": "Sci-Fi", "year": 2024, 
         "description": "A brilliant physicist discovers a parallel universe where time runs backward. The pacing is intense, but the final season felt a bit rushed."},
        
        {"title": "Dragon's Ascent", "genre": "Fantasy", "year": 2023, 
         "description": "Epic tale of warring kingdoms and ancient magic. Visually stunning, but the main character arc was confusing and underdeveloped."},
         
        {"title": "The Baker's Secret", "genre": "Drama", "year": 2022, 
         "description": "A heartwarming story about a family recipe that changes a small town. Critics praised the strong performances and emotional depth."},
         
        {"title": "Cyber Heist 2077", "genre": "Action", "year": 2024, 
         "description": "High-octane action in a dystopian future. Many reviewers found the plot derivative and the violence excessive. A massive disappointment."},
         
        {"title": "Historical Echoes", "genre": "Documentary", "year": 2023, 
         "description": "A highly acclaimed documentary series exploring forgotten historical figures. It is insightful, meticulously researched, and a wonderful watch."},
    ]
    return data

def display_streaming_data():
    """Prints the simulated streaming data in a formatted way."""
    data = get_simulated_streaming_data()
    print("\n" + "=" * 60)
    print("           SIMULATED STREAMING TV SERIES LIST           ")
    print("=" * 60)
    print("To analyze a show, copy its 'Description' and paste it")
    print("back into the analyzer.")
    print("-" * 60)
    
    for i, item in enumerate(data):
        print(f"[{i+1}] Title: {item['title']} ({item['year']})")
        print(f"    Genre: {item['genre']}")
        print(f"    Description (for analysis): {item['description']}")
        print("-" * 60)

def calculate_and_print_summary(history, title="COMPREHENSIVE MOVIE REVIEW SUMMARY"):
    """
    Calculates and prints the comprehensive sentiment summary from the history of classifications.
    The 'title' argument allows reusing this for both batch and interactive summaries.
    """
    if not history:
        print(f"\n[ {title} ]")
        print("No reviews entered yet.")
        print("-" * 60)
        return

    # Count the occurrences of each classification
    counts = Counter(history)
    total = len(history)

    pos_count = counts.get('Positive', 0)
    neg_count = counts.get('Negative', 0)
    neu_count = counts.get('Neutral', 0)

    # Determine overall movie sentiment based on majority
    if pos_count > neg_count and pos_count > neu_count:
        overall_sentiment = "OVERALL POSITIVE (Strongly Recommended)"
    elif neg_count > pos_count and neg_count > neu_count:
        overall_sentiment = "OVERALL NEGATIVE (Not Recommended)"
    else:
        # If there is a tie or Neutral is the majority
        overall_sentiment = "OVERALL MIXED/NEUTRAL (Proceed with Caution)"

    print("\n" + "=" * 60)
    print(f"{title:^60}") # Center the dynamic title
    print("=" * 60)
    print(f"Total Reviews Analyzed: {total}")
    print(f"  - Positive Reviews: {pos_count} ({pos_count/total:.1%})")
    print(f"  - Negative Reviews: {neg_count} ({neg_count/total:.1%})")
    print(f"  - Neutral Reviews:  {neu_count} ({neu_count/total:.1%})")
    print("-" * 60)
    print(f"Movie Assessment: {overall_sentiment}")
    print("=" * 60)

def run_batch_analysis(sid):
    """
    Runs analysis on the predefined list of SAMPLE_REVIEWS and prints a summary.
    Includes a PREPROCESSING step for each review.
    """
    batch_history = []
    print("\n" + "#" * 60)
    print("           AUTOMATIC BATCH ANALYSIS STARTED             ")
    print("#" * 60)
    
    for i, raw_review in enumerate(SAMPLE_REVIEWS):
        # Preprocessing Step (New): Clean the raw text before analysis
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
        
        # Display the RAW review, but mention that it was pre-processed
        print(f"Review {i+1} (Processed: '{review[:30]}...'): '{raw_review[:40]}...' -> Classification: {sentiment} (Compound: {compound_score:.4f})")

    # Print the overall summary for the batch
    calculate_and_print_summary(
        batch_history, 
        title="BATCH ANALYSIS SUMMARY (10 Scraped Reviews)"
    )
    return batch_history

def run_interactive_analyzer(sid, initial_history):
    """
    Runs the interactive sentiment analysis loop using the provided SentimentIntensityAnalyzer
    and initialized with the history from the batch analysis.
    """
    # Note: In interactive mode, we assume the user provides clean text, 
    # but the preprocessor can still be used if needed for robustness.
    analysis_history = initial_history.copy()
    
    print("=" * 60)
    print("         INTERACTIVE ANALYSIS MODE ACTIVATED          ")
    print("=" * 60)
    print("Please enter your movie review (English is highly recommended).")
    print("Type 'list' to show simulated streaming series data.")
    print("Type 'result' for a summary of ALL reviews entered so far (batch + interactive).")
    print("Type 'exit' or 'quit' to end the program.")
    print("-" * 60)

    while True:
        # Get user input
        review = input(">>> Enter review: ")
        
        # Check for list command
        if review.lower() == 'list':
            display_streaming_data()
            continue
        
        # Check for summary command
        if review.lower() == 'result':
            calculate_and_print_summary(
                analysis_history, 
                title="CUMULATIVE REVIEW SUMMARY"
            )
            continue
        
        # Check for exit command
        if review.lower() in ['exit', 'quit']:
            # Print final cumulative summary before exiting
            calculate_and_print_summary(
                analysis_history, 
                title="FINAL CUMULATIVE SUMMARY"
            )
            print("-" * 60)
            print("Thank you for using the analyzer. Program terminated.")
            break
        
        # Ignore empty input
        if not review.strip():
            continue

        try:
            # Preprocess the interactive input as well for robustness
            cleaned_review = preprocess_review(review)
            
            # Get sentiment scores
            scores = sid.polarity_scores(cleaned_review)
            compound_score = scores['compound']
            
            # Classify based on the Compound score thresholds
            if compound_score >= 0.05:
                sentiment = "Positive"
            elif compound_score <= -0.05:
                sentiment = "Negative"
            else:
                sentiment = "Neutral"
            
            # Store the classification for later summary
            analysis_history.append(sentiment) 
                
            # Output the result
            print(f"\n[ Analysis Result ]")
            print(f"  Sentiment Classification: {sentiment}")
            print(f"  VADER Detailed Scores (Neg/Neu/Pos): {scores['neg']:.4f}, {scores['neu']:.4f}, {scores['pos']:.4f}")
            print(f"  Compound Score: {compound_score:.4f}")
            print("-" * 60)
            
        except Exception as e:
            print(f"An error occurred during analysis: {e}")
            print("-" * 60)


# Main execution block
if __name__ == "__main__":
    ensure_nltk_data()
    
    # Initialize VADER once
    sid = SentimentIntensityAnalyzer()
    
    # 1. Run batch analysis on the "scraped" data first
    initial_history = run_batch_analysis(sid)
    
    # 2. Then enter interactive mode, continuing the history
    run_interactive_analyzer(sid, initial_history)