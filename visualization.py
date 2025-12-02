"""
Visualization Module
Contains all visualization functions for EDA and temporal split analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def visualize_release_date_split(train_data, test_data, full_data, save_path=None):
    """
    Comprehensive visualization of temporal split based on release dates

    Args:
        train_data: Training DataFrame
        test_data: Test DataFrame
        full_data: Complete interaction DataFrame
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. Release date distribution
    ax1 = axes[0, 0]
    ax1.hist([train_data['release_date'], test_data['release_date']],
             bins=50, label=['Train', 'Test'], alpha=0.7, color=['#3498db', '#e74c3c'])
    ax1.set_xlabel('Game Release Date', fontsize=10)
    ax1.set_ylabel('Number of Interactions', fontsize=10)
    ax1.set_title('Temporal Split by Release Date', fontsize=11, fontweight='bold')
    ax1.legend()
    ax1.tick_params(axis='x', rotation=45)

    # 2. Per-user split example (sample 10 users)
    ax2 = axes[0, 1]
    sample_users = np.random.choice(
        full_data['user_id'].unique(),
        min(10, full_data['user_id'].nunique()),
        replace=False
    )

    for i, user_id in enumerate(sample_users):
        user_train = train_data[train_data['user_id'] == user_id]['release_date']
        user_test = test_data[test_data['user_id'] == user_id]['release_date']

        if len(user_train) > 0:
            ax2.scatter(user_train, [i] * len(user_train), alpha=0.6, s=30, color='#3498db')
        if len(user_test) > 0:
            ax2.scatter(user_test, [i] * len(user_test), alpha=0.6, s=30,
                        color='#e74c3c', marker='s')

    ax2.set_xlabel('Game Release Date', fontsize=10)
    ax2.set_ylabel('Sample Users', fontsize=10)
    ax2.set_title('Per-User Temporal Split', fontsize=11, fontweight='bold')
    ax2.legend(['Train', 'Test'], loc='upper left')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Games per user distribution
    ax3 = axes[0, 2]
    train_games_per_user = train_data.groupby('user_id').size()
    test_games_per_user = test_data.groupby('user_id').size()

    ax3.hist([train_games_per_user, test_games_per_user],
             bins=30, label=['Train', 'Test'], alpha=0.7, color=['#3498db', '#e74c3c'])
    ax3.set_xlabel('Games per User', fontsize=10)
    ax3.set_ylabel('Number of Users', fontsize=10)
    ax3.set_title('Interaction Count Distribution', fontsize=11, fontweight='bold')
    ax3.legend()

    # 4. Release year distribution
    ax4 = axes[1, 0]
    train_years = train_data['release_date'].dt.year.value_counts().sort_index()
    test_years = test_data['release_date'].dt.year.value_counts().sort_index()

    years = sorted(set(train_years.index) | set(test_years.index))
    train_counts = [train_years.get(y, 0) for y in years]
    test_counts = [test_years.get(y, 0) for y in years]

    x = np.arange(len(years))
    width = 0.35
    ax4.bar(x - width / 2, train_counts, width, label='Train', alpha=0.7, color='#3498db')
    ax4.bar(x + width / 2, test_counts, width, label='Test', alpha=0.7, color='#e74c3c')
    ax4.set_xlabel('Release Year', fontsize=10)
    ax4.set_ylabel('Number of Interactions', fontsize=10)
    ax4.set_title('Split by Release Year', fontsize=11, fontweight='bold')
    ax4.set_xticks(x[::2])  # Show every other year
    ax4.set_xticklabels(years[::2], rotation=45)
    ax4.legend()

    # 5. Temporal gap analysis
    ax5 = axes[1, 1]
    temporal_gaps = []

    for user_id in set(train_data['user_id']) & set(test_data['user_id']):
        user_train = train_data[train_data['user_id'] == user_id]['release_date']
        user_test = test_data[test_data['user_id'] == user_id]['release_date']

        if len(user_train) > 0 and len(user_test) > 0:
            gap_days = (user_test.min() - user_train.max()).days
            temporal_gaps.append(gap_days)

    if temporal_gaps:
        ax5.hist(temporal_gaps, bins=50, edgecolor='black', alpha=0.7, color='#2ecc71')
        ax5.axvline(np.median(temporal_gaps), color='red', linestyle='--', linewidth=2,
                    label=f'Median: {np.median(temporal_gaps):.0f} days')
        ax5.set_xlabel('Days Between Train and Test', fontsize=10)
        ax5.set_ylabel('Number of Users', fontsize=10)
        ax5.set_title('Temporal Gap Distribution', fontsize=11, fontweight='bold')
        ax5.legend()

    # 6. Cumulative distribution
    ax6 = axes[1, 2]
    all_dates_sorted = full_data.sort_values('release_date')['release_date'].reset_index(drop=True)
    cumulative = np.arange(1, len(all_dates_sorted) + 1)

    ax6.plot(all_dates_sorted, cumulative, color='gray', alpha=0.5, linewidth=2)

    # Mark 80% split point
    split_point = int(len(all_dates_sorted) * 0.8)
    ax6.axvline(all_dates_sorted.iloc[split_point], color='red', linestyle='--',
                linewidth=2, label='80/20 Split')
    ax6.set_xlabel('Game Release Date', fontsize=10)
    ax6.set_ylabel('Cumulative Interactions', fontsize=10)
    ax6.set_title('Cumulative Timeline', fontsize=11, fontweight='bold')
    ax6.legend()
    ax6.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")

    plt.show()

    # Print statistics
    print_split_statistics(train_data, test_data, full_data, temporal_gaps)


def print_split_statistics(train_data, test_data, full_data, temporal_gaps):
    """
    Print detailed temporal split statistics
    """
    print("\n" + "=" * 70)
    print("TEMPORAL SPLIT VALIDATION STATISTICS")
    print("=" * 70)

    print(f"\n📊 Data Split:")
    print(f"   Train: {len(train_data):,} interactions ({len(train_data) / len(full_data) * 100:.1f}%)")
    print(f"   Test:  {len(test_data):,} interactions ({len(test_data) / len(full_data) * 100:.1f}%)")

    print(f"\n📅 Release Date Ranges:")
    print(f"   Train: {train_data['release_date'].min().date()} → {train_data['release_date'].max().date()}")
    print(f"   Test:  {test_data['release_date'].min().date()} → {test_data['release_date'].max().date()}")

    print(f"\n👥 User Statistics:")
    print(f"   Total users: {full_data['user_id'].nunique():,}")
    print(f"   Users in train: {train_data['user_id'].nunique():,}")
    print(f"   Users in test: {test_data['user_id'].nunique():,}")
    overlap = len(set(train_data['user_id']) & set(test_data['user_id']))
    print(f"   Users in both: {overlap:,} ({overlap / train_data['user_id'].nunique() * 100:.1f}%)")

    if temporal_gaps:
        print(f"\n⏰ Temporal Gaps:")
        print(f"   Median: {np.median(temporal_gaps):.0f} days ({np.median(temporal_gaps) / 365:.1f} years)")
        print(f"   Mean: {np.mean(temporal_gaps):.0f} days ({np.mean(temporal_gaps) / 365:.1f} years)")
        print(f"   Range: {np.min(temporal_gaps):.0f} - {np.max(temporal_gaps):.0f} days")

    print(f"\n✅ Temporal Ordering:")
    avg_train_year = train_data['release_date'].dt.year.mean()
    avg_test_year = test_data['release_date'].dt.year.mean()
    print(f"   Avg train year: {avg_train_year:.1f}")
    print(f"   Avg test year: {avg_test_year:.1f}")
    print(f"   Difference: +{avg_test_year - avg_train_year:.1f} years")

    if avg_test_year > avg_train_year:
        print(f"   ✓ Test games are newer (temporal ordering preserved)")
    else:
        print(f"   ⚠ Warning: Test games not consistently newer")

    print("=" * 70 + "\n")