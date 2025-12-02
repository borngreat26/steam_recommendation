"""
Main Execution Script for Game Recommendation System
Assignment 2 - CSE 158/258

This script runs the complete pipeline:
1. Load data
2. Exploratory Data Analysis
3. Temporal train/test split
4. Train models
5. Evaluate models
6. Visualize results
"""

import warnings

warnings.filterwarnings('ignore')

# Import custom modules
from data_loader import load_data, create_dataframes
from eda import run_full_eda
from preprocessing import create_temporal_split_by_release_date
from models import (
    PopularityRecommender,
    RandomRecommender,
    ItemBasedCF,
    GenreBasedRecommender,
    ContentBasedRecommender
)
from evaluation import compare_models, visualize_model_comparison, print_evaluation_summary
from visualization import visualize_release_date_split


def main():
    """
    Main execution function
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "GAME RECOMMENDATION SYSTEM")
    print(" " * 25 + "Assignment 2 - CSE 158/258")
    print("=" * 80 + "\n")

    # ==================== STEP 1: LOAD DATA ====================
    print("STEP 1: Loading datasets...")
    print("-" * 80)

    user_items, user_reviews, games_data = load_data(data_dir='data')
    playtime_df, review_df, games_df = create_dataframes(user_items, user_reviews, games_data)

    # ==================== STEP 2: EXPLORATORY DATA ANALYSIS ====================
    print("\n\nSTEP 2: Exploratory Data Analysis...")
    print("-" * 80)

    # Run EDA (set to False to skip if you've already seen it)
    RUN_EDA = True
    if RUN_EDA:
        run_full_eda(user_items, user_reviews, games_data,
                     playtime_df, review_df, games_df)
    else:
        print("Skipping EDA (set RUN_EDA=True to run)")

    # ==================== STEP 3: TEMPORAL SPLIT ====================
    print("\n\nSTEP 3: Creating temporal train/test split...")
    print("-" * 80)

    train_data, test_data, full_data = create_temporal_split_by_release_date(
        user_items,
        games_data,
        split_ratio=0.8
    )

    # Visualize split
    print("\nVisualizing temporal split...")
    visualize_release_date_split(train_data, test_data, full_data)

    # # ==================== STEP 4: TRAIN MODELS ====================
    # print("\n\nSTEP 4: Training recommendation models...")
    # print("-" * 80)
    #
    # models = {
    #     'Random': RandomRecommender(seed=42),
    #     'Popularity': PopularityRecommender(),
    #     'Genre-Based': GenreBasedRecommender(games_df),
    #     'Item-Based CF': ItemBasedCF(),
    #     'Content-Based': ContentBasedRecommender(games_df)
    # }
    #
    # for name, model in models.items():
    #     print(f"\nTraining {name}...")
    #     model.fit(train_data)
    #
    # print("\n✓ All models trained successfully!")
    #
    # # ==================== STEP 5: EVALUATE MODELS ====================
    # print("\n\nSTEP 5: Evaluating models...")
    # print("-" * 80)
    #
    # k_values = [5, 10, 20]
    # results_df = compare_models(models, test_data, k_values=k_values)
    #
    # # Print results
    # print_evaluation_summary(results_df)
    #
    # # Save results
    # results_df.to_csv('model_results.csv', index=False)
    # print("\n✓ Results saved to 'model_results.csv'")
    #
    # # ==================== STEP 6: VISUALIZE RESULTS ====================
    # print("\n\nSTEP 6: Visualizing results...")
    # print("-" * 80)
    #
    # visualize_model_comparison(results_df, save_path='model_comparison.png')
    #
    # # ==================== STEP 7: EXAMPLE RECOMMENDATIONS ====================
    # print("\n\nSTEP 7: Example recommendations...")
    # print("-" * 80)
    #
    # # Get a sample user
    # sample_user = test_data['user_id'].iloc[0]
    # print(f"\nGenerating recommendations for user: {sample_user}")
    #
    # # Get user's actual liked games
    # user_test = test_data[
    #     (test_data['user_id'] == sample_user) &
    #     (test_data['interaction'] == 1)
    #     ]
    # actual_games = user_test['item_name'].tolist()
    # print(f"\nUser's actual liked games in test set:")
    # for i, game in enumerate(actual_games[:5], 1):
    #     print(f"  {i}. {game}")
    #
    # # Get recommendations from best model
    # best_model_name = results_df[results_df['K'] == 10].sort_values(
    #     'NDCG@K', ascending=False
    # ).iloc[0]['Model']
    # best_model = models[best_model_name]
    #
    # print(f"\nTop 10 recommendations from {best_model_name}:")
    # recommendations = best_model.recommend(sample_user, k=10)
    #
    # for i, item_id in enumerate(recommendations, 1):
    #     # Try to get game name
    #     game_info = games_df[games_df['id'] == item_id]
    #     if len(game_info) > 0:
    #         game_name = game_info.iloc[0].get('app_name', item_id)
    #     else:
    #         game_name = f"Game {item_id}"
    #     print(f"  {i}. {game_name}")
    #
    # # ==================== DONE ====================
    # print("\n\n" + "=" * 80)
    # print(" " * 30 + "PIPELINE COMPLETE!")
    # print("=" * 80 + "\n")
    #
    # print("Generated files:")
    # print("  - model_results.csv: Numerical results")
    # print("  - model_comparison.png: Visualization of model performance")
    # print("\nYou can now:")
    # print("  1. Review the results in model_results.csv")
    # print("  2. Check the visualizations")
    # print("  3. Use the trained models for predictions")
    #

if __name__ == '__main__':
    main()