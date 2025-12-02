"""
Data Preprocessing and Temporal Split Module
Handles data preparation and temporal validation splits
"""
import pandas as pd
import numpy as np


def prepare_interaction_data(playtime_df, review_df):
    """
    Create unified interaction dataset with binary labels

    Args:
        playtime_df: User-item playtime data
        review_df: User review data

    Returns:
        interaction_data: DataFrame with interaction labels
    """
    # Start with playtime data (only games that were played)
    interaction_data = playtime_df[playtime_df['playtime_forever'] > 0].copy()

    # Merge with review data
    interaction_data = interaction_data.merge(
        review_df[['user_id', 'item_id', 'recommend']],
        on=['user_id', 'item_id'],
        how='left'
    )

    # Create binary interaction:
    # Positive if played >60 min OR explicitly recommended
    interaction_data['interaction'] = (
            (interaction_data['playtime_forever'] > 60) |  # More than 1 hour
            (interaction_data['recommend'] == True)
    ).astype(int)

    print(f"\n✓ Prepared interaction data:")
    print(f"  - Total interactions: {len(interaction_data):,}")
    print(f"  - Positive interactions: {interaction_data['interaction'].sum():,}")
    print(f"  - Positive rate: {interaction_data['interaction'].mean():.2%}")

    return interaction_data


def create_temporal_split_by_release_date(user_items, games_data, split_ratio=0.8):
    """
    Create temporal split using game release dates as proxy for acquisition time.

    Methodology:
    - Sort each user's games by release date
    - First 80% (older games) = Train
    - Last 20% (newer games) = Test

    Args:
        user_items: Raw user items data
        games_data: Raw games data
        split_ratio: Train/test split ratio (default 0.8)

    Returns:
        train_data: Training DataFrame
        test_data: Test DataFrame
        full_data: Complete interaction DataFrame
    """
    # Build games dictionary with release dates
    games_dict = {}
    for game in games_data:
        if 'id' in game and 'release_date' in game:
            try:
                release_date = pd.to_datetime(game['release_date'])
                games_dict[game['id']] = {
                    'release_date': release_date,
                    'name': game.get('app_name', 'Unknown'),
                    'genres': game.get('genres', []),
                    'tags': game.get('tags', [])
                }
            except:
                pass  # Skip invalid dates

    print(f"✓ Games with valid release dates: {len(games_dict):,}")

    # Build interaction data with release dates
    all_interactions = []
    for user in user_items:
        user_id = user['user_id']

        for item in user['items']:
            item_id = item['item_id']
            playtime = item['playtime_forever']

            # Only include games that were played and have release dates
            if playtime > 0 and item_id in games_dict:
                all_interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'item_name': item['item_name'],
                    'playtime_forever': playtime,
                    'release_date': games_dict[item_id]['release_date'],
                    'interaction': 1 if playtime > 60 else 0
                })

    interactions_df = pd.DataFrame(all_interactions)
    print(f"✓ Total interactions with release dates: {len(interactions_df):,}")

    # Perform temporal split per user
    train_list = []
    test_list = []
    users_kept = 0
    users_skipped = 0

    for user_id in interactions_df['user_id'].unique():
        user_data = interactions_df[interactions_df['user_id'] == user_id].copy()

        # Sort by release date (temporal ordering)
        user_data = user_data.sort_values('release_date')

        # Need at least 5 games for meaningful split
        if len(user_data) < 5:
            users_skipped += 1
            continue

        users_kept += 1

        # Split at 80/20 boundary
        split_idx = int(len(user_data) * split_ratio)
        train_list.append(user_data.iloc[:split_idx])
        test_list.append(user_data.iloc[split_idx:])

    train_data = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    test_data = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    print(f"\n=== Temporal Split Statistics ===")
    print(f"Users kept: {users_kept:,}")
    print(f"Users skipped (<5 games): {users_skipped:,}")
    print(f"Train interactions: {len(train_data):,}")
    print(f"Test interactions: {len(test_data):,}")
    print(f"Train date range: {train_data['release_date'].min().date()} → {train_data['release_date'].max().date()}")
    print(f"Test date range: {test_data['release_date'].min().date()} → {test_data['release_date'].max().date()}")

    return train_data, test_data, interactions_df


def create_hybrid_temporal_split(user_items, user_reviews, games_data, split_ratio=0.8):
    """
    Hybrid approach: Use review timestamps when available,
    fall back to release dates otherwise.

    This provides the most accurate temporal split by combining:
    1. Actual review timestamps (high confidence)
    2. Game release dates (medium confidence proxy)

    Args:
        user_items: Raw user items data
        user_reviews: Raw user reviews data
        games_data: Raw games data
        split_ratio: Train/test split ratio

    Returns:
        train_data: Training DataFrame
        test_data: Test DataFrame
        full_data: Complete interaction DataFrame
    """
    # Build games dictionary
    games_dict = {}
    for game in games_data:
        if 'id' in game and 'release_date' in game:
            try:
                games_dict[game['id']] = {
                    'release_date': pd.to_datetime(game['release_date']),
                    'name': game.get('app_name', 'Unknown')
                }
            except:
                pass

    all_interactions = []

    # 1. Add interactions from reviews (HIGH CONFIDENCE - actual timestamp)
    for user_review in user_reviews:
        user_id = user_review['user_id']

        for review in user_review['reviews']:
            if 'posted' in review and review['posted']:
                try:
                    # Parse review date
                    review_date = pd.to_datetime(
                        review['posted'].replace('Posted ', '').replace('.', '')
                    )

                    all_interactions.append({
                        'user_id': user_id,
                        'item_id': review['item_id'],
                        'timestamp': review_date,
                        'interaction': 1 if review['recommend'] else 0,
                        'timestamp_type': 'review',
                        'confidence_score': 1.0
                    })
                except:
                    pass

    # 2. Add interactions from user items (MEDIUM CONFIDENCE - release date proxy)
    for user in user_items:
        user_id = user['user_id']

        for item in user['items']:
            item_id = item['item_id']
            playtime = item['playtime_forever']

            # Only include if played significantly
            if playtime > 60 and item_id in games_dict:
                all_interactions.append({
                    'user_id': user_id,
                    'item_id': item_id,
                    'timestamp': games_dict[item_id]['release_date'],
                    'interaction': 1,
                    'timestamp_type': 'release_date',
                    'confidence_score': 0.5
                })

    # Create DataFrame and handle duplicates (prefer review timestamps)
    interactions_df = pd.DataFrame(all_interactions)
    interactions_df = interactions_df.sort_values('confidence_score', ascending=False)
    interactions_df = interactions_df.drop_duplicates(
        subset=['user_id', 'item_id'],
        keep='first'
    )

    print(f"\n✓ Timestamp Quality:")
    print(interactions_df['timestamp_type'].value_counts())

    # Perform temporal split per user
    train_list = []
    test_list = []

    for user_id in interactions_df['user_id'].unique():
        user_data = interactions_df[interactions_df['user_id'] == user_id].sort_values('timestamp')

        if len(user_data) < 5:
            continue

        split_idx = int(len(user_data) * split_ratio)
        train_list.append(user_data.iloc[:split_idx])
        test_list.append(user_data.iloc[split_idx:])

    train_data = pd.concat(train_list, ignore_index=True) if train_list else pd.DataFrame()
    test_data = pd.concat(test_list, ignore_index=True) if test_list else pd.DataFrame()

    print(f"\n=== Hybrid Temporal Split Statistics ===")
    print(f"Train interactions: {len(train_data):,}")
    print(f"Test interactions: {len(test_data):,}")

    return train_data, test_data, interactions_df