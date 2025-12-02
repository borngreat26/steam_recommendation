"""
Evaluation Module
Contains evaluation metrics and model comparison functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def precision_at_k(recommended, actual, k):
    """
    Precision@K: Proportion of recommended items that are relevant

    Args:
        recommended: List of recommended item IDs
        actual: List of actual relevant item IDs
        k: Cutoff position

    Returns:
        Precision score [0, 1]
    """
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(actual))
    return hits / k if k > 0 else 0


def recall_at_k(recommended, actual, k):
    """
    Recall@K: Proportion of relevant items that are recommended

    Args:
        recommended: List of recommended item IDs
        actual: List of actual relevant item IDs
        k: Cutoff position

    Returns:
        Recall score [0, 1]
    """
    recommended_k = recommended[:k]
    hits = len(set(recommended_k) & set(actual))
    return hits / len(actual) if len(actual) > 0 else 0


def ndcg_at_k(recommended, actual, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain
    Measures ranking quality with position discount

    Args:
        recommended: List of recommended item IDs (in rank order)
        actual: List of actual relevant item IDs
        k: Cutoff position

    Returns:
        NDCG score [0, 1]
    """
    recommended_k = recommended[:k]

    # Calculate DCG: sum of relevance / log2(position + 1)
    dcg = sum(
        [1 / np.log2(i + 2) if rec in actual else 0
         for i, rec in enumerate(recommended_k)]
    )

    # Calculate ideal DCG (if all relevant items were at top)
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), k))])

    return dcg / idcg if idcg > 0 else 0


def hit_rate_at_k(recommended, actual, k):
    """
    Hit Rate@K: Binary indicator if at least one relevant item is in top-K

    Args:
        recommended: List of recommended item IDs
        actual: List of actual relevant item IDs
        k: Cutoff position

    Returns:
        1 if hit, 0 otherwise
    """
    recommended_k = recommended[:k]
    return 1 if len(set(recommended_k) & set(actual)) > 0 else 0


def evaluate_model(model, test_data, k_values=[5, 10, 20], verbose=True):
    """
    Comprehensive model evaluation across multiple metrics

    Args:
        model: Trained recommendation model
        test_data: Test DataFrame with columns [user_id, item_id, interaction]
        k_values: List of K values to evaluate
        verbose: Print progress

    Returns:
        Dictionary of averaged results per K
    """
    results = {k: {
        'precision': [],
        'recall': [],
        'ndcg': [],
        'hit_rate': []
    } for k in k_values}

    total_users = test_data['user_id'].nunique()
    evaluated_users = 0

    # Group test data by user
    for user_id in test_data['user_id'].unique():
        user_test = test_data[
            (test_data['user_id'] == user_id) &
            (test_data['interaction'] == 1)
            ]
        actual_items = user_test['item_id'].tolist()

        if len(actual_items) == 0:
            continue

        # Get recommendations
        try:
            recommended = model.recommend(user_id, k=max(k_values))

            if not recommended:  # Empty recommendations
                continue

            evaluated_users += 1

            # Calculate metrics for each k
            for k in k_values:
                results[k]['precision'].append(
                    precision_at_k(recommended, actual_items, k)
                )
                results[k]['recall'].append(
                    recall_at_k(recommended, actual_items, k)
                )
                results[k]['ndcg'].append(
                    ndcg_at_k(recommended, actual_items, k)
                )
                results[k]['hit_rate'].append(
                    hit_rate_at_k(recommended, actual_items, k)
                )
        except Exception as e:
            if verbose:
                print(f"Warning: Error evaluating user {user_id}: {e}")
            continue

    # Average results
    averaged_results = {}
    for k in k_values:
        averaged_results[k] = {
            'precision': np.mean(results[k]['precision']) if results[k]['precision'] else 0,
            'recall': np.mean(results[k]['recall']) if results[k]['recall'] else 0,
            'ndcg': np.mean(results[k]['ndcg']) if results[k]['ndcg'] else 0,
            'hit_rate': np.mean(results[k]['hit_rate']) if results[k]['hit_rate'] else 0,
            'n_users': len(results[k]['precision'])
        }

    if verbose:
        print(f"✓ Evaluated {evaluated_users}/{total_users} users")

    return averaged_results


def compare_models(models_dict, test_data, k_values=[5, 10, 20]):
    """
    Evaluate and compare multiple models

    Args:
        models_dict: Dictionary of {model_name: model_instance}
        test_data: Test DataFrame
        k_values: List of K values

    Returns:
        DataFrame with comparison results
    """
    results_comparison = {}

    for name, model in models_dict.items():
        print(f"\nEvaluating {name}...")
        results_comparison[name] = evaluate_model(model, test_data, k_values)

    # Convert to DataFrame
    results_df = pd.DataFrame([
        {
            'Model': model_name,
            'K': k,
            'Precision@K': results[k]['precision'],
            'Recall@K': results[k]['recall'],
            'NDCG@K': results[k]['ndcg'],
            'Hit Rate@K': results[k]['hit_rate'],
            'Users': results[k]['n_users']
        }
        for model_name, results in results_comparison.items()
        for k in results.keys()
    ])

    return results_df


def visualize_model_comparison(results_df, save_path=None):
    """
    Visualize model comparison across metrics

    Args:
        results_df: Results DataFrame from compare_models()
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = ['Precision@K', 'Recall@K', 'NDCG@K', 'Hit Rate@K']

    for idx, metric in enumerate(metrics):
        ax = axes[idx // 2, idx % 2]

        for model_name in results_df['Model'].unique():
            model_data = results_df[results_df['Model'] == model_name]
            ax.plot(model_data['K'], model_data[metric],
                    marker='o', label=model_name, linewidth=2)

        ax.set_xlabel('K (Number of Recommendations)', fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} vs K', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved figure to {save_path}")

    plt.show()


def print_evaluation_summary(results_df):
    """
    Print formatted evaluation summary

    Args:
        results_df: Results DataFrame from compare_models()
    """
    print("\n" + "=" * 80)
    print("MODEL EVALUATION SUMMARY")
    print("=" * 80)

    for k in sorted(results_df['K'].unique()):
        print(f"\n--- K = {k} ---")
        k_results = results_df[results_df['K'] == k].sort_values(
            'NDCG@K', ascending=False
        )

        for _, row in k_results.iterrows():
            print(f"{row['Model']:20s}: "
                  f"P@{k}={row['Precision@K']:.4f}, "
                  f"R@{k}={row['Recall@K']:.4f}, "
                  f"NDCG@{k}={row['NDCG@K']:.4f}, "
                  f"HR@{k}={row['Hit Rate@K']:.4f}")

    print("\n" + "=" * 80)