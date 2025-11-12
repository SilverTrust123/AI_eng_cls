import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import os
from collections import Counter # Added for sentiment summary calculation

# --- VADER Sentiment Analyzer is based on English lexicon and rules ---
# VADER is primarily trained on English text. Its accuracy may decrease 
# significantly when analyzing non-English languages.
# It is highly recommended to input English reviews for the best results.

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

def calculate_and_print_summary(history):
    """
    Calculates and prints the comprehensive sentiment summary from the history of classifications.
    """
    if not history:
        print("\n[ Summary Result ]")
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
    print("           COMPREHENSIVE MOVIE REVIEW SUMMARY           ")
    print("=" * 60)
    print(f"Total Reviews Analyzed: {total}")
    print(f"  - Positive Reviews: {pos_count} ({pos_count/total:.1%})")
    print(f"  - Negative Reviews: {neg_count} ({neg_count/total:.1%})")
    print(f"  - Neutral Reviews:  {neu_count} ({neu_count/total:.1%})")
    print("-" * 60)
    print(f"Movie Assessment: {overall_sentiment}")
    print("=" * 60)

def run_interactive_analyzer():
    """
    Runs the interactive sentiment analysis loop, prompting the user for input
    and displaying real-time analysis results based on the VADER compound score.
    """
    # Ensure the required NLTK resource is downloaded
    ensure_nltk_data()

    # Initialize the VADER Sentiment Analyzer
    sid = SentimentIntensityAnalyzer()
    
    # List to store the classification of each entered review
    analysis_history = [] 
    
    print("=" * 60)
    print("      Interactive Movie Review Sentiment Analyzer (NLTK VADER)      ")
    print("=" * 60)
    print("Please enter your movie review (English is highly recommended).")
    print("Type 'result' for a summary of all reviews.") # New Instruction
    print("Type 'exit' or 'quit' to end the program.")
    print("-" * 60)

    while True:
        # Get user input
        review = input(">>> Enter review: ")
        
        # Check for summary command
        if review.lower() == 'result':
            calculate_and_print_summary(analysis_history)
            continue # Go back to the input prompt
        
        # Check for exit command
        if review.lower() in ['exit', 'quit']:
            # Print final summary before exiting
            calculate_and_print_summary(analysis_history)
            print("-" * 60)
            print("Thank you for using the analyzer. Program terminated.")
            break
        
        # Ignore empty input
        if not review.strip():
            continue

        try:
            # Get sentiment scores
            # polarity_scores returns a dictionary with 'neg', 'neu', 'pos', 'compound'
            scores = sid.polarity_scores(review)
            
            # The Compound score is a single, normalized metric from VADER
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


# Execute the interactive analysis
if __name__ == "__main__":
    run_interactive_analyzer()