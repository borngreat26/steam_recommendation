"""
Exploratory Data Analysis Module
Contains functions for dataset exploration and visualization
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def explore_dataset_overview(user_items, user_reviews, games_data):
    """
    Print high-level dataset statistics
    """
    print("\n" + "=" * 70)
    print("DATASET OVERVIEW")
    print("=" * 70)
    print(f"Total users: {len(user_items):,}")
    print(f"Total user reviews: {len(user_reviews):,}")
    print(f"Total games: {len(games_data):,}")
    print("=" * 70 + "\n")


def analyze_user_activity(playtime_df, save_path=None):
    """
    Analyze and visualize user activity patterns

    Args:
        playtime_df: DataFrame with user-item playtime data
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Distribution of playtime (log scale)
    ax1 = axes[0, 0]
    played_games = playtime_df[playtime_df['playtime_forever'] > 0]
    ax1.hist(played_games['playtime_forever'], bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Playtime (minutes)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Distribution of Game Playtime', fontsize=11, fontweight='bold')
    ax1.set_xscale('log')

    # 2. Games per user distribution
    ax2 = axes[0, 1]
    games_per_user = playtime_df.groupby('user_id').size()
    ax2.hist(games_per_user, bins=50, edgecolor='black', alpha=0.7, color='#3498db')
    ax2.set_xlabel('Number of Games Owned', fontsize=10)
    ax2.set_ylabel('Number of Users', fontsize=10)
    ax2.set_title('Distribution of Games per User', fontsize=11, fontweight='bold')

    # 3. Playtime per user (total hours)
    ax3 = axes[1, 0]
    total_playtime_per_user = playtime_df.groupby('user_id')['playtime_forever'].sum() / 60
    ax3.hist(total_playtime_per_user, bins=50, edgecolor='black', alpha=0.7, color='#e74c3c')
    ax3.set_xlabel('Total Playtime (hours)', fontsize=10)
    ax3.set_ylabel('Number of Users', fontsize=10)
    ax3.set_title('Distribution of Total Playtime per User', fontsize=11, fontweight='bold')
    ax3.set_xscale('log')

    # 4. Engagement rate (% of owned games with >1hr playtime)
    ax4 = axes[1, 1]
    user_engagement = playtime_df.groupby('user_id').apply(
        lambda x: (x['playtime_forever'] > 60).sum() / len(x)
    )
    ax4.hist(user_engagement, bins=30, edgecolor='black', alpha=0.7, color='#2ecc71')
    ax4.set_xlabel('Engagement Rate (% games >1hr)', fontsize=10)
    ax4.set_ylabel('Number of Users', fontsize=10)
    ax4.set_title('User Engagement Distribution', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    # Print statistics
    print("\n📊 User Activity Statistics:")
    print(f"   Median games per user: {games_per_user.median():.0f}")
    print(f"   Mean games per user: {games_per_user.mean():.1f}")
    print(f"   Median playtime per user: {total_playtime_per_user.median():.1f} hours")
    print(f"   Mean playtime per user: {total_playtime_per_user.mean():.1f} hours")
    print(f"   Median engagement rate: {user_engagement.median():.2%}")


def analyze_genre_distribution(games_data, save_path=None):
    """
    Analyze and visualize game genre distribution

    Args:
        games_data: List of game metadata dictionaries
        save_path: Optional path to save figure
    """
    # Count genres
    genre_counts = Counter()
    for game in games_data:
        if 'genres' in game and game['genres']:
            genre_counts.update(game['genres'])

    # Get top 15 genres
    top_genres = dict(genre_counts.most_common(15))

    # Create visualization
    plt.figure(figsize=(12, 6))
    plt.barh(list(top_genres.keys()), list(top_genres.values()),
             edgecolor='black', alpha=0.7)
    plt.xlabel('Number of Games', fontsize=11)
    plt.ylabel('Genre', fontsize=11)
    plt.title('Most Common Game Genres (Top 15)', fontsize=12, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    print(f"\n🎮 Genre Statistics:")
    print(f"   Total unique genres: {len(genre_counts)}")
    print(f"   Most common: {genre_counts.most_common(1)[0][0]} ({genre_counts.most_common(1)[0][1]:,} games)")


def analyze_review_patterns(review_df, save_path=None):
    """
    Analyze review recommendation patterns

    Args:
        review_df: DataFrame with user reviews
        save_path: Optional path to save figure
    """
    recommendation_rate = review_df['recommend'].mean()

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 1. Recommendation distribution
    ax1 = axes[0]
    rec_counts = review_df['recommend'].value_counts()
    ax1.bar(['Not Recommended', 'Recommended'],
            [rec_counts.get(False, 0), rec_counts.get(True, 0)],
            color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Number of Reviews', fontsize=10)
    ax1.set_title('Review Recommendations', fontsize=11, fontweight='bold')

    # 2. Reviews per user
    ax2 = axes[1]
    reviews_per_user = review_df.groupby('user_id').size()
    ax2.hist(reviews_per_user, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    ax2.set_xlabel('Number of Reviews', fontsize=10)
    ax2.set_ylabel('Number of Users', fontsize=10)
    ax2.set_title('Reviews per User', fontsize=11, fontweight='bold')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

    print(f"\n📝 Review Statistics:")
    print(f"   Total reviews: {len(review_df):,}")
    print(f"   Recommendation rate: {recommendation_rate:.2%}")
    print(f"   Median reviews per user: {reviews_per_user.median():.0f}")


def analyze_sparsity(playtime_df, games_df):
    """
    Calculate and report data sparsity

    Args:
        playtime_df: User-item playtime DataFrame
        games_df: Games metadata DataFrame
    """
    total_interactions = len(playtime_df[playtime_df['playtime_forever'] > 0])
    total_users = playtime_df['user_id'].nunique()
    total_games = len(games_df)
    total_possible = total_users * total_games
    sparsity = 1 - (total_interactions / total_possible)

    print(f"\n📉 Data Sparsity Analysis:")
    print(f"   Total users: {total_users:,}")
    print(f"   Total games: {total_games:,}")
    print(f"   Observed interactions: {total_interactions:,}")
    print(f"   Possible interactions: {total_possible:,}")
    print(f"   Matrix sparsity: {sparsity:.4%}")
    print(f"   Density: {(1 - sparsity):.4%}")


def run_full_eda(user_items, user_reviews, games_data, playtime_df, review_df, games_df):
    """
    Run complete exploratory data analysis

    Args:
        user_items: Raw user items data
        user_reviews: Raw user reviews data
        games_data: Raw games data
        playtime_df: Playtime DataFrame
        review_df: Reviews DataFrame
        games_df: Games DataFrame
    """
    print("\n" + "=" * 70)
    print("RUNNING EXPLORATORY DATA ANALYSIS")
    print("=" * 70)

    # Dataset overview
    explore_dataset_overview(user_items, user_reviews, games_data)

    # User activity analysis
    print("\n1. Analyzing user activity patterns...")
    analyze_user_activity(playtime_df)

    # Genre distribution
    print("\n2. Analyzing genre distribution...")
    analyze_genre_distribution(games_data)

    # Review patterns
    print("\n3. Analyzing review patterns...")
    analyze_review_patterns(review_df)

    # Sparsity analysis
    print("\n4. Analyzing data sparsity...")
    analyze_sparsity(playtime_df, games_df)

    print("\n" + "=" * 70)
    print("EDA COMPLETE")
    print("=" * 70 + "\n")