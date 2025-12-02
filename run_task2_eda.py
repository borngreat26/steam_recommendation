"""
Runner script for Task 2 EDA
"""
import warnings
warnings.filterwarnings('ignore')

from task2_eda import extract_review_text_from_steam_reviews, run_task2_eda

if __name__ == '__main__':
    # Extract review data from Australian User Reviews JSON
    print("Loading review data from Australian User Reviews dataset...")
    reviews_df = extract_review_text_from_steam_reviews('data/Australian User Reviews.json')

    # Run the complete EDA
    run_task2_eda(reviews_df, text_column='review_text')
