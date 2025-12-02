"""
Task 2: Review Sentiment Prediction - Complete Implementation
Binary classification to predict whether user will recommend a game
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns


class MajorityClassifier:
    """
    Baseline 1: Always predict majority class
    """

    def __init__(self):
        self.majority_class = None

    def fit(self, X, y):
        self.majority_class = y.mode()[0]
        return self

    def predict(self, X):
        return np.array([self.majority_class] * len(X))

    def predict_proba(self, X):
        # Return probabilities (all same)
        proba = np.zeros((len(X), 2))
        if self.majority_class == 1:
            proba[:, 1] = 1.0
        else:
            proba[:, 0] = 1.0
        return proba


class LengthBasedClassifier:
    """
    Baseline 2: Classify based on review length
    """

    def __init__(self):
        self.threshold = None
        self.majority_class = None

    def fit(self, X, y):
        # X should be review texts
        lengths = X.str.len()
        self.majority_class = y.mode()[0]

        # Find optimal threshold
        best_acc = 0
        best_threshold = 0
        for threshold in range(50, 500, 50):
            predictions = (lengths > threshold).astype(int)
            acc = accuracy_score(y, predictions)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold

        self.threshold = best_threshold
        return self

    def predict(self, X):
        lengths = X.str.len()
        return (lengths > self.threshold).astype(int)

    def predict_proba(self, X):
        predictions = self.predict(X)
        proba = np.zeros((len(X), 2))
        proba[predictions == 1, 1] = 1.0
        proba[predictions == 0, 0] = 1.0
        return proba


def prepare_task2_data(reviews_df, text_column='review_text',
                       test_size=0.3, random_state=42):
    """
    Prepare data for sentiment classification

    Returns:
        X_train, X_test, y_train, y_test (all DataFrames/Series)
    """
    # Remove rows with missing text or labels
    df = reviews_df[[text_column, 'recommend']].dropna()

    print(f"✓ Data prepared: {len(df):,} reviews")
    print(f"   Positive: {sum(df['recommend']):,} ({sum(df['recommend']) / len(df) * 100:.1f}%)")
    print(f"   Negative: {sum(~df['recommend']):,} ({sum(~df['recommend']) / len(df) * 100:.1f}%)")

    X = df[text_column]
    y = df['recommend'].astype(int)

    # Stratified split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    print(f"\n✓ Split: {len(X_train):,} train, {len(X_test):,} test")

    return X_train, X_test, y_train, y_test


def train_baseline_models(X_train, y_train):
    """
    Train all baseline models
    """
    models = {}

    print("\n" + "=" * 70)
    print("TRAINING BASELINE MODELS")
    print("=" * 70)

    # Baseline 1: Majority Class
    print("\n1. Training Majority Class Classifier...")
    models['Majority Class'] = {
        'model': MajorityClassifier(),
        'requires_features': False
    }
    models['Majority Class']['model'].fit(X_train, y_train)
    print("   ✓ Trained")

    # Baseline 2: Length-Based
    print("\n2. Training Length-Based Classifier...")
    models['Length-Based'] = {
        'model': LengthBasedClassifier(),
        'requires_features': False
    }
    models['Length-Based']['model'].fit(X_train, y_train)
    print(f"   ✓ Trained (threshold: {models['Length-Based']['model'].threshold} chars)")

    # Baseline 3: Bag of Words + Logistic Regression
    print("\n3. Training Bag-of-Words + Logistic Regression...")
    bow_vectorizer = CountVectorizer(max_features=1000, stop_words='english')
    X_train_bow = bow_vectorizer.fit_transform(X_train)

    bow_model = LogisticRegression(max_iter=1000, random_state=42)
    bow_model.fit(X_train_bow, y_train)

    models['Bag-of-Words + LR'] = {
        'model': bow_model,
        'vectorizer': bow_vectorizer,
        'requires_features': True
    }
    print("   ✓ Trained")

    # Baseline 4: Naive Bayes
    print("\n4. Training Multinomial Naive Bayes...")
    nb_vectorizer = CountVectorizer(max_features=5000, stop_words='english')
    X_train_nb = nb_vectorizer.fit_transform(X_train)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_nb, y_train)

    models['Naive Bayes'] = {
        'model': nb_model,
        'vectorizer': nb_vectorizer,
        'requires_features': True
    }
    print("   ✓ Trained")

    # Baseline 5: TF-IDF + Logistic Regression
    print("\n5. Training TF-IDF + Logistic Regression...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

    tfidf_model = LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight='balanced',
        random_state=42
    )
    tfidf_model.fit(X_train_tfidf, y_train)

    models['TF-IDF + LR'] = {
        'model': tfidf_model,
        'vectorizer': tfidf_vectorizer,
        'requires_features': True
    }
    print("   ✓ Trained")

    print("\n" + "=" * 70)
    print("ALL MODELS TRAINED")
    print("=" * 70)

    return models


def evaluate_model(model_dict, X_test, y_test, model_name):
    """
    Evaluate a single model
    """
    model = model_dict['model']

    # Get predictions
    if model_dict['requires_features']:
        X_test_features = model_dict['vectorizer'].transform(X_test)
        y_pred = model.predict(X_test_features)
        y_pred_proba = model.predict_proba(X_test_features)[:, 1]
    else:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba)
    }

    return results, y_pred, y_pred_proba


def evaluate_all_models(models, X_test, y_test):
    """
    Evaluate all models and return results DataFrame
    """
    print("\n" + "=" * 70)
    print("EVALUATING MODELS")
    print("=" * 70)

    results_list = []
    predictions_dict = {}

    for model_name, model_dict in models.items():
        print(f"\nEvaluating {model_name}...")
        results, y_pred, y_pred_proba = evaluate_model(model_dict, X_test, y_test, model_name)
        results_list.append(results)
        predictions_dict[model_name] = {
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

        print(f"   Accuracy:  {results['Accuracy']:.4f}")
        print(f"   Precision: {results['Precision']:.4f}")
        print(f"   Recall:    {results['Recall']:.4f}")
        print(f"   F1-Score:  {results['F1-Score']:.4f}")
        print(f"   ROC-AUC:   {results['ROC-AUC']:.4f}")

    results_df = pd.DataFrame(results_list)

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)

    return results_df, predictions_dict


def visualize_results(results_df, predictions_dict, y_test, save_path=None):
    """
    Visualize model comparison results
    """
    fig = plt.figure(figsize=(16, 10))

    # Create subplots
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. Metrics comparison (bar chart)
    ax1 = fig.add_subplot(gs[0, :])
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    x = np.arange(len(results_df))
    width = 0.15

    for i, metric in enumerate(metrics):
        ax1.bar(x + i * width, results_df[metric], width, label=metric)

    ax1.set_xlabel('Model', fontsize=11)
    ax1.set_ylabel('Score', fontsize=11)
    ax1.set_title('Model Performance Comparison', fontsize=13, fontweight='bold')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(results_df['Model'], rotation=15, ha='right')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, 1.0)

    # 2-4. Confusion matrices for top 3 models
    top_3_models = results_df.nlargest(3, 'F1-Score')['Model'].tolist()

    for idx, model_name in enumerate(top_3_models):
        ax = fig.add_subplot(gs[1, idx])
        y_pred = predictions_dict[model_name]['y_pred']
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        ax.set_title(f'{model_name}\nConfusion Matrix', fontsize=10, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9)

    # 5. ROC Curves
    ax5 = fig.add_subplot(gs[2, :2])
    for model_name in top_3_models:
        y_pred_proba = predictions_dict[model_name]['y_pred_proba']
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = roc_auc_score(y_test, y_pred_proba)
        ax5.plot(fpr, tpr, label=f'{model_name} (AUC={auc:.3f})', linewidth=2)

    ax5.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', linewidth=1)
    ax5.set_xlabel('False Positive Rate', fontsize=11)
    ax5.set_ylabel('True Positive Rate', fontsize=11)
    ax5.set_title('ROC Curves', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(alpha=0.3)

    # 6. F1-Score ranking
    ax6 = fig.add_subplot(gs[2, 2])
    results_sorted = results_df.sort_values('F1-Score')
    colors = plt.cm.RdYlGn(results_sorted['F1-Score'])
    ax6.barh(results_sorted['Model'], results_sorted['F1-Score'], color=colors, edgecolor='black')
    ax6.set_xlabel('F1-Score', fontsize=11)
    ax6.set_title('Models Ranked by F1-Score', fontsize=12, fontweight='bold')
    ax6.set_xlim(0, 1.0)
    ax6.grid(axis='x', alpha=0.3)

    plt.suptitle('Task 2: Review Sentiment Prediction - Model Evaluation',
                 fontsize=14, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to {save_path}")

    plt.show()


def print_detailed_results(results_df, best_model_name, models, X_test, y_test):
    """
    Print detailed classification report for best model
    """
    print("\n" + "=" * 70)
    print("DETAILED RESULTS FOR BEST MODEL")
    print("=" * 70)

    print(f"\nBest Model: {best_model_name}")
    print(f"F1-Score: {results_df[results_df['Model'] == best_model_name]['F1-Score'].values[0]:.4f}")

    # Get predictions
    model_dict = models[best_model_name]
    if model_dict['requires_features']:
        X_test_features = model_dict['vectorizer'].transform(X_test)
        y_pred = model_dict['model'].predict(X_test_features)
    else:
        y_pred = model_dict['model'].predict(X_test)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Not Recommend', 'Recommend'],
                                digits=4))

    # Feature importance (if applicable)
    if best_model_name == 'TF-IDF + LR' or best_model_name == 'Bag-of-Words + LR':
        print("\nTop 10 Most Predictive Features:")
        print("\nPositive Indicators (Recommend):")
        model = model_dict['model']
        vectorizer = model_dict['vectorizer']
        feature_names = vectorizer.get_feature_names_out()

        coef = model.coef_[0]
        top_positive = np.argsort(coef)[-10:][::-1]
        for idx in top_positive:
            print(f"   {feature_names[idx]:20s}: {coef[idx]:+.4f}")

        print("\nNegative Indicators (Not Recommend):")
        top_negative = np.argsort(coef)[:10]
        for idx in top_negative:
            print(f"   {feature_names[idx]:20s}: {coef[idx]:+.4f}")

    print("\n" + "=" * 70)


def cross_validate_best_model(best_model_dict, X_train, y_train, cv=5):
    """
    Perform cross-validation on best model
    """
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION")
    print("=" * 70)

    if best_model_dict['requires_features']:
        X_train_features = best_model_dict['vectorizer'].fit_transform(X_train)
    else:
        X_train_features = X_train

    model = best_model_dict['model']

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)

    print(f"\nPerforming {cv}-fold stratified cross-validation...")

    cv_scores = cross_val_score(model, X_train_features, y_train,
                                cv=skf, scoring='f1')

    print(f"\nCross-Validation F1-Scores:")
    for i, score in enumerate(cv_scores, 1):
        print(f"   Fold {i}: {score:.4f}")

    print(f"\nMean F1-Score: {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

    if cv_scores.std() < 0.02:
        print("✓ Low variance - model is stable!")
    else:
        print("⚠ High variance - consider more data or regularization")

    print("\n" + "=" * 70)

    return cv_scores


def main_task2():
    """
    Main execution function for Task 2
    """
    print("\n" + "=" * 80)
    print(" " * 20 + "TASK 2: REVIEW SENTIMENT PREDICTION")
    print(" " * 25 + "Binary Classification")
    print("=" * 80 + "\n")

    # Load data
    print("Loading data...")
    # You'll need to extract review text from Steam Reviews.json
    # For now, using placeholder
    from task2_eda import extract_review_text_from_steam_reviews
    reviews_df = extract_review_text_from_steam_reviews()

    # Prepare data
    X_train, X_test, y_train, y_test = prepare_task2_data(reviews_df)

    # Train models
    models = train_baseline_models(X_train, y_train)

    # Evaluate models
    results_df, predictions_dict = evaluate_all_models(models, X_test, y_test)

    # Save results
    results_df.to_csv('task2_results.csv', index=False)
    print("\n✓ Results saved to task2_results.csv")

    # Visualize
    visualize_results(results_df, predictions_dict, y_test,
                      save_path='task2_model_comparison.png')

    # Detailed analysis of best model
    best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
    print_detailed_results(results_df, best_model_name, models, X_test, y_test)

    # Cross-validation
    cross_validate_best_model(models[best_model_name], X_train, y_train, cv=5)

    print("\n" + "=" * 80)
    print(" " * 30 + "TASK 2 COMPLETE!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main_task2()