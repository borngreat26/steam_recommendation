"""
Data Loading Module
Loads all datasets from JSON files
"""
import json
import pandas as pd


def load_data(data_dir='data'):
    """
    Load all datasets

    Returns:
        user_items: List of user item data
        user_reviews: List of user review data
        games_data: List of game metadata
    """
    # Load Australian Users Items (user game libraries)
    with open(f'{data_dir}/Australian Users Items.json', 'r', encoding='utf-8') as f:
        user_items = [json.loads(line) for line in f]

    # Load Australian User Reviews
    with open(f'{data_dir}/Australian User Reviews.json', 'r', encoding='utf-8') as f:
        user_reviews = [json.loads(line) for line in f]

    # Load Steam Games Dataset
    with open(f'{data_dir}/Steam Games Dataset.json', 'r', encoding='utf-8') as f:
        games_data = [json.loads(line) for line in f]

    print(f"✓ Loaded {len(user_items)} users")
    print(f"✓ Loaded {len(user_reviews)} user reviews")
    print(f"✓ Loaded {len(games_data)} games")

    return user_items, user_reviews, games_data


def create_dataframes(user_items, user_reviews, games_data):
    """
    Convert raw data to pandas DataFrames

    Returns:
        playtime_df: User-item interactions with playtime
        review_df: User reviews
        games_df: Game metadata
    """
    # Create playtime DataFrame
    playtime_data = []
    for user in user_items:
        for item in user['items']:
            playtime_data.append({
                'user_id': user['user_id'],
                'item_id': item['item_id'],
                'item_name': item['item_name'],
                'playtime_forever': item['playtime_forever']
            })
    playtime_df = pd.DataFrame(playtime_data)

    # Create review DataFrame
    review_data = []
    for user_review in user_reviews:
        for review in user_review['reviews']:
            review_data.append({
                'user_id': user_review['user_id'],
                'item_id': review['item_id'],
                'recommend': review['recommend'],
                'posted': review.get('posted', None),
                'helpful': review.get('helpful', 'No ratings yet')
            })
    review_df = pd.DataFrame(review_data)

    # Create games DataFrame
    games_df = pd.DataFrame(games_data)

    print(f"\n✓ Created DataFrames:")
    print(f"  - Playtime: {len(playtime_df):,} interactions")
    print(f"  - Reviews: {len(review_df):,} reviews")
    print(f"  - Games: {len(games_df):,} games")

    return playtime_df, review_df, games_df