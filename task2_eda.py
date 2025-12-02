"""
Task 2: Review Sentiment Prediction - EDA Module
Additional analysis specific to text classification task
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
from wordcloud import WordCloud


def analyze_review_sentiment_distribution(review_df, save_path=None):
    """
    Analyze distribution of review recommendations (sentiment)
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Overall sentiment distribution
    ax1 = axes[0]
    sentiment_counts = review_df['recommend'].value_counts()
    colors = ['#e74c3c', '#2ecc71']
    ax1.pie(sentiment_counts, labels=['Not Recommend', 'Recommend'],
            autopct='%1.1f%%', colors=colors, startangle=90)
    ax1.set_title('Sentiment Distribution', fontsize=12, fontweight='bold')

    # 2. Sentiment by user (users' recommendation rate)
    ax2 = axes[1]
    user_rec_rate = review_df.groupby('user_id')['recommend'].mean()
    ax2.hist(user_rec_rate, bins=30, edgecolor='black', alpha=0.7, color='#3498db')
    ax2.set_xlabel('User Recommendation Rate', fontsize=10)
    ax2.set_ylabel('Number of Users', fontsize=10)
    ax2.set_title('User Recommendation Patterns', fontsize=11, fontweight='bold')
    ax2.axvline(user_rec_rate.median(), color='red', linestyle='--',
                label=f'Median: {user_rec_rate.median():.2f}')
    ax2.legend()

    # 3. Reviews per user by sentiment
    ax3 = axes[2]
    reviews_per_user_pos = review_df[review_df['recommend'] == True].groupby('user_id').size()
    reviews_per_user_neg = review_df[review_df['recommend'] == False].groupby('user_id').size()

    ax3.hist([reviews_per_user_pos, reviews_per_user_neg],
             bins=20, label=['Positive', 'Negative'],
             alpha=0.7, color=['#2ecc71', '#e74c3c'])
    ax3.set_xlabel('Reviews per User', fontsize=10)
    ax3.set_ylabel('Number of Users', fontsize=10)
    ax3.set_title('Review Count by Sentiment', fontsize=11, fontweight='bold')
    ax3.legend()
    ax3.set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n📊 Sentiment Distribution Statistics:")
    print(f"   Total reviews: {len(review_df):,}")
    print(
        f"   Positive (Recommend): {sentiment_counts.get(True, 0):,} ({sentiment_counts.get(True, 0) / len(review_df) * 100:.1f}%)")
    print(
        f"   Negative (Not Recommend): {sentiment_counts.get(False, 0):,} ({sentiment_counts.get(False, 0) / len(review_df) * 100:.1f}%)")
    print(f"   Class Balance Ratio: {sentiment_counts.get(True, 0) / sentiment_counts.get(False, 1):.2f}:1")

    # Check for class imbalance
    imbalance_ratio = max(sentiment_counts) / min(sentiment_counts)
    if imbalance_ratio > 1.5:
        print(f"   ⚠️  CLASS IMBALANCE DETECTED! Ratio: {imbalance_ratio:.2f}:1")
        print(f"   → Consider: stratified sampling, class weights, SMOTE")
    else:
        print(f"   ✓ Classes relatively balanced (ratio: {imbalance_ratio:.2f}:1)")


def extract_review_text_from_steam_reviews(steam_reviews_file='data/Steam Reviews.json'):
    """
    Extract review text from Steam Reviews dataset
    Returns DataFrame with user_id, product_id, review_text, recommend
    """
    import json

    reviews_data = []
    with open(steam_reviews_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                review = json.loads(line)
                if 'review' in review and 'recommended' in review:
                    reviews_data.append({
                        'user_id': review.get('username', 'unknown'),
                        'product_id': review.get('product_id', None),
                        'review_text': review['review'],
                        'recommend': review['recommended'],
                        'hours': review.get('hours', 0.0),
                        'date': review.get('date', None)
                    })
            except:
                continue

    reviews_df = pd.DataFrame(reviews_data)
    print(f"✓ Extracted {len(reviews_df):,} reviews with text")
    return reviews_df


def analyze_review_text_characteristics(reviews_df, text_column='review_text', save_path=None):
    """
    Analyze characteristics of review text
    """
    # Calculate text statistics
    reviews_df['char_count'] = reviews_df[text_column].str.len()
    reviews_df['word_count'] = reviews_df[text_column].str.split().str.len()
    reviews_df['avg_word_length'] = reviews_df['char_count'] / reviews_df['word_count']
    reviews_df['uppercase_ratio'] = reviews_df[text_column].apply(
        lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
    )
    reviews_df['exclamation_count'] = reviews_df[text_column].str.count('!')
    reviews_df['question_count'] = reviews_df[text_column].str.count('\\?')

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # 1. Review length distribution (characters)
    ax1 = axes[0, 0]
    ax1.hist([reviews_df[reviews_df['recommend'] == True]['char_count'],
              reviews_df[reviews_df['recommend'] == False]['char_count']],
             bins=50, label=['Positive', 'Negative'], alpha=0.7,
             color=['#2ecc71', '#e74c3c'])
    ax1.set_xlabel('Characters', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title('Review Length Distribution', fontsize=11, fontweight='bold')
    ax1.set_xlim(0, 1000)
    ax1.legend()

    # 2. Word count distribution
    ax2 = axes[0, 1]
    ax2.hist([reviews_df[reviews_df['recommend'] == True]['word_count'],
              reviews_df[reviews_df['recommend'] == False]['word_count']],
             bins=50, label=['Positive', 'Negative'], alpha=0.7,
             color=['#2ecc71', '#e74c3c'])
    ax2.set_xlabel('Words', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title('Word Count Distribution', fontsize=11, fontweight='bold')
    ax2.set_xlim(0, 200)
    ax2.legend()

    # 3. Average word length
    ax3 = axes[0, 2]
    ax3.boxplot([reviews_df[reviews_df['recommend'] == True]['avg_word_length'].dropna(),
                 reviews_df[reviews_df['recommend'] == False]['avg_word_length'].dropna()],
                labels=['Positive', 'Negative'])
    ax3.set_ylabel('Average Word Length', fontsize=10)
    ax3.set_title('Average Word Length by Sentiment', fontsize=11, fontweight='bold')

    # 4. Uppercase ratio
    ax4 = axes[1, 0]
    ax4.hist([reviews_df[reviews_df['recommend'] == True]['uppercase_ratio'],
              reviews_df[reviews_df['recommend'] == False]['uppercase_ratio']],
             bins=30, label=['Positive', 'Negative'], alpha=0.7,
             color=['#2ecc71', '#e74c3c'])
    ax4.set_xlabel('Uppercase Ratio', fontsize=10)
    ax4.set_ylabel('Frequency', fontsize=10)
    ax4.set_title('Uppercase Usage', fontsize=11, fontweight='bold')
    ax4.legend()

    # 5. Exclamation marks
    ax5 = axes[1, 1]
    ax5.hist([reviews_df[reviews_df['recommend'] == True]['exclamation_count'],
              reviews_df[reviews_df['recommend'] == False]['exclamation_count']],
             bins=20, label=['Positive', 'Negative'], alpha=0.7,
             color=['#2ecc71', '#e74c3c'], range=(0, 10))
    ax5.set_xlabel('Exclamation Marks', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Exclamation Mark Usage', fontsize=11, fontweight='bold')
    ax5.legend()

    # 6. Scatter: word count vs sentiment
    ax6 = axes[1, 2]
    pos_sample = reviews_df[reviews_df['recommend'] == True].sample(min(1000, sum(reviews_df['recommend'])))
    neg_sample = reviews_df[reviews_df['recommend'] == False].sample(min(1000, sum(~reviews_df['recommend'])))
    ax6.scatter(pos_sample['word_count'], [1] * len(pos_sample), alpha=0.3, s=10, color='#2ecc71', label='Positive')
    ax6.scatter(neg_sample['word_count'], [0] * len(neg_sample), alpha=0.3, s=10, color='#e74c3c', label='Negative')
    ax6.set_xlabel('Word Count', fontsize=10)
    ax6.set_ylabel('Sentiment', fontsize=10)
    ax6.set_title('Word Count vs Sentiment', fontsize=11, fontweight='bold')
    ax6.set_xlim(0, 300)
    ax6.set_yticks([0, 1])
    ax6.set_yticklabels(['Negative', 'Positive'])
    ax6.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Print statistics
    print("\n📝 Review Text Statistics:")
    print("\nPositive Reviews:")
    pos_reviews = reviews_df[reviews_df['recommend'] == True]
    print(
        f"   Median length: {pos_reviews['char_count'].median():.0f} chars, {pos_reviews['word_count'].median():.0f} words")
    print(f"   Mean length: {pos_reviews['char_count'].mean():.0f} chars, {pos_reviews['word_count'].mean():.0f} words")

    print("\nNegative Reviews:")
    neg_reviews = reviews_df[reviews_df['recommend'] == False]
    print(
        f"   Median length: {neg_reviews['char_count'].median():.0f} chars, {neg_reviews['word_count'].median():.0f} words")
    print(f"   Mean length: {neg_reviews['char_count'].mean():.0f} chars, {neg_reviews['word_count'].mean():.0f} words")

    print("\n✓ Observation:")
    if pos_reviews['word_count'].median() > neg_reviews['word_count'].median():
        print("   → Positive reviews tend to be LONGER than negative reviews")
    else:
        print("   → Negative reviews tend to be LONGER than positive reviews")


def analyze_common_words(reviews_df, text_column='review_text', n_top=20, save_path=None):
    """
    Analyze most common words in positive vs negative reviews
    """
    from sklearn.feature_extraction.text import CountVectorizer

    # Separate positive and negative reviews
    pos_reviews = reviews_df[reviews_df['recommend'] == True][text_column].fillna('')
    neg_reviews = reviews_df[reviews_df['recommend'] == False][text_column].fillna('')

    # Create CountVectorizer (remove common English stop words)
    vectorizer = CountVectorizer(max_features=n_top, stop_words='english',
                                 lowercase=True, ngram_range=(1, 1))

    # Get top words for positive reviews
    pos_counts = vectorizer.fit_transform(pos_reviews)
    pos_words = vectorizer.get_feature_names_out()
    pos_word_counts = pos_counts.sum(axis=0).A1
    pos_top_words = sorted(zip(pos_words, pos_word_counts), key=lambda x: x[1], reverse=True)

    # Get top words for negative reviews
    vectorizer_neg = CountVectorizer(max_features=n_top, stop_words='english',
                                     lowercase=True, ngram_range=(1, 1))
    neg_counts = vectorizer_neg.fit_transform(neg_reviews)
    neg_words = vectorizer_neg.get_feature_names_out()
    neg_word_counts = neg_counts.sum(axis=0).A1
    neg_top_words = sorted(zip(neg_words, neg_word_counts), key=lambda x: x[1], reverse=True)

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Positive words
    ax1 = axes[0]
    words, counts = zip(*pos_top_words)
    ax1.barh(range(len(words)), counts, color='#2ecc71', alpha=0.7, edgecolor='black')
    ax1.set_yticks(range(len(words)))
    ax1.set_yticklabels(words)
    ax1.invert_yaxis()
    ax1.set_xlabel('Frequency', fontsize=11)
    ax1.set_title(f'Top {n_top} Words in POSITIVE Reviews', fontsize=12, fontweight='bold')

    # Negative words
    ax2 = axes[1]
    words, counts = zip(*neg_top_words)
    ax2.barh(range(len(words)), counts, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.set_yticks(range(len(words)))
    ax2.set_yticklabels(words)
    ax2.invert_yaxis()
    ax2.set_xlabel('Frequency', fontsize=11)
    ax2.set_title(f'Top {n_top} Words in NEGATIVE Reviews', fontsize=12, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

    print("\n🔤 Most Common Words:")
    print("\nPositive Reviews:")
    for word, count in pos_top_words[:10]:
        print(f"   {word:15s}: {count:,}")

    print("\nNegative Reviews:")
    for word, count in neg_top_words[:10]:
        print(f"   {word:15s}: {count:,}")


def create_word_clouds(reviews_df, text_column='review_text', save_path=None):
    """
    Create word clouds for positive and negative reviews
    """
    try:
        from wordcloud import WordCloud

        # Separate texts
        pos_text = ' '.join(reviews_df[reviews_df['recommend'] == True][text_column].fillna(''))
        neg_text = ' '.join(reviews_df[reviews_df['recommend'] == False][text_column].fillna(''))

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Positive word cloud
        wordcloud_pos = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Greens', max_words=100,
                                  stopwords=set(['game', 'play', 'playing'])).generate(pos_text)
        axes[0].imshow(wordcloud_pos, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('Positive Reviews Word Cloud', fontsize=14, fontweight='bold')

        # Negative word cloud
        wordcloud_neg = WordCloud(width=800, height=400, background_color='white',
                                  colormap='Reds', max_words=100,
                                  stopwords=set(['game', 'play', 'playing'])).generate(neg_text)
        axes[1].imshow(wordcloud_neg, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Negative Reviews Word Cloud', fontsize=14, fontweight='bold')

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

        print("✓ Word clouds generated successfully")
    except ImportError:
        print("⚠️  wordcloud library not installed. Run: pip install wordcloud")


def run_task2_eda(reviews_df, text_column='review_text'):
    """
    Run complete EDA for Task 2 (Sentiment Prediction)
    """
    print("\n" + "=" * 70)
    print("TASK 2: REVIEW SENTIMENT PREDICTION - EDA")
    print("=" * 70)

    # 1. Sentiment distribution
    print("\n1. Analyzing sentiment distribution...")
    analyze_review_sentiment_distribution(reviews_df)

    # 2. Text characteristics
    print("\n2. Analyzing review text characteristics...")
    analyze_review_text_characteristics(reviews_df, text_column)

    # 3. Common words
    print("\n3. Analyzing most common words...")
    analyze_common_words(reviews_df, text_column, n_top=20)

    # 4. Word clouds (optional)
    print("\n4. Generating word clouds...")
    try:
        create_word_clouds(reviews_df, text_column)
    except:
        print("   Skipped (wordcloud library not available)")

    print("\n" + "=" * 70)
    print("TASK 2 EDA COMPLETE")
    print("=" * 70 + "\n")