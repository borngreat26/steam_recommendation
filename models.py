"""
Recommendation Models Module
Contains all baseline and advanced recommendation models
"""
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer


class PopularityRecommender:
    """
    Baseline 1: Recommend most popular games globally
    """

    def __init__(self):
        self.popular_items = None

    def fit(self, train_data):
        """
        Learn popular items from training data

        Args:
            train_data: DataFrame with columns [user_id, item_id, interaction]
        """
        # Most interacted items
        self.popular_items = (
            train_data.groupby('item_id')['interaction']
            .sum()
            .sort_values(ascending=False)
            .index.tolist()
        )
        print(f"✓ Popularity model trained: {len(self.popular_items)} items")

    def recommend(self, user_id, k=10):
        """
        Recommend top-K popular items

        Args:
            user_id: User ID (ignored for popularity model)
            k: Number of recommendations

        Returns:
            List of top-K item IDs
        """
        return self.popular_items[:k]


class RandomRecommender:
    """
    Baseline 2: Random recommendations
    """

    def __init__(self, seed=42):
        self.all_items = None
        self.seed = seed
        np.random.seed(seed)

    def fit(self, train_data):
        """
        Learn available items from training data
        """
        self.all_items = train_data['item_id'].unique().tolist()
        print(f"✓ Random model trained: {len(self.all_items)} items")

    def recommend(self, user_id, k=10):
        """
        Recommend K random items
        """
        return np.random.choice(
            self.all_items,
            size=min(k, len(self.all_items)),
            replace=False
        ).tolist()


class ItemBasedCF:
    """
    Baseline 3: Item-Based Collaborative Filtering
    "Users who played X also played Y"
    """

    def __init__(self):
        self.item_similarity = None
        self.user_item_matrix = None

    def fit(self, train_data):
        """
        Build item-item similarity matrix

        Args:
            train_data: DataFrame with columns [user_id, item_id, interaction]
        """
        # Create user-item matrix
        self.user_item_matrix = train_data.pivot_table(
            index='user_id',
            columns='item_id',
            values='interaction',
            fill_value=0
        )

        # Calculate item-item similarity using cosine similarity
        item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity = pd.DataFrame(
            item_similarity_matrix,
            index=self.user_item_matrix.columns,
            columns=self.user_item_matrix.columns
        )

        print(f"✓ Item-based CF trained: {len(self.item_similarity)} items")

    def recommend(self, user_id, k=10):
        """
        Recommend items similar to what user has interacted with

        Args:
            user_id: User ID
            k: Number of recommendations

        Returns:
            List of top-K item IDs
        """
        if user_id not in self.user_item_matrix.index:
            return []

        # Get user's interacted items
        user_items = self.user_item_matrix.loc[user_id]
        interacted_items = user_items[user_items > 0].index.tolist()

        if not interacted_items:
            return []

        # Calculate scores for all items
        scores = {}
        for item in self.item_similarity.columns:
            if item not in interacted_items:
                # Aggregate similarity to all liked items
                score = sum(
                    self.item_similarity.loc[item, liked_item]
                    for liked_item in interacted_items
                    if liked_item in self.item_similarity.columns
                )
                scores[item] = score

        # Return top k
        recommended = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in recommended[:k]]


class GenreBasedRecommender:
    """
    Baseline 4: Genre-Based Recommendations
    Recommend popular games from user's favorite genres
    """

    def __init__(self, games_df):
        self.games_df = games_df
        self.genre_popularity = None
        self.user_genre_preferences = None

    def fit(self, train_data):
        """
        Learn user genre preferences and genre-level popularity

        Args:
            train_data: DataFrame with columns [user_id, item_id, interaction]
        """
        # Create mapping of item_id to genres
        item_genres = {}
        for idx, game in self.games_df.iterrows():
            if 'id' in game and 'genres' in game and game['genres']:
                item_genres[game['id']] = game['genres']

        # Build user genre preferences
        self.user_genre_preferences = {}
        for user_id in train_data['user_id'].unique():
            user_data = train_data[
                (train_data['user_id'] == user_id) &
                (train_data['interaction'] == 1)
                ]

            # Aggregate genres from liked games
            genre_counts = {}
            for item_id in user_data['item_id']:
                if item_id in item_genres:
                    for genre in item_genres[item_id]:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1

            if genre_counts:
                # Store top genres
                self.user_genre_preferences[user_id] = sorted(
                    genre_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]  # Top 3 genres

        # Build genre-level popularity
        self.genre_popularity = {}
        for genre in set(g for genres in item_genres.values() for g in genres):
            genre_items = [
                item_id for item_id, genres in item_genres.items()
                if genre in genres
            ]
            genre_interactions = train_data[
                train_data['item_id'].isin(genre_items)
            ]['interaction'].sum()
            self.genre_popularity[genre] = (genre_items, genre_interactions)

        print(f"✓ Genre-based model trained: {len(self.user_genre_preferences)} users profiled")

    def recommend(self, user_id, k=10):
        """
        Recommend popular games from user's preferred genres
        """
        if user_id not in self.user_genre_preferences:
            return []

        # Get user's top genres
        user_top_genres = [g for g, count in self.user_genre_preferences[user_id]]

        # Aggregate items from these genres
        candidate_items = set()
        for genre in user_top_genres:
            if genre in self.genre_popularity:
                items, _ = self.genre_popularity[genre]
                candidate_items.update(items)

        # Return top K (could be weighted by popularity, but keeping simple)
        return list(candidate_items)[:k]


class ContentBasedRecommender:
    """
    Content-Based Filtering using game features (genres, tags)
    """

    def __init__(self, games_df):
        self.games_df = games_df
        self.game_features = None
        self.user_profiles = None
        self.feature_columns = None

    def fit(self, train_data):
        """
        Build user profiles based on game content features

        Args:
            train_data: DataFrame with columns [user_id, item_id, interaction]
        """
        # Create game feature matrix from genres and tags
        mlb_genres = MultiLabelBinarizer()
        mlb_tags = MultiLabelBinarizer()

        # Process genres
        games_with_genres = self.games_df[
            self.games_df['genres'].notna()
        ].copy()

        if len(games_with_genres) == 0:
            print("Warning: No games with genres found")
            return

        genre_features = mlb_genres.fit_transform(games_with_genres['genres'])

        # Process tags (take top 50 most common)
        games_with_tags = self.games_df[
            self.games_df['tags'].notna()
        ].copy()

        if len(games_with_tags) > 0:
            tag_features = mlb_tags.fit_transform(games_with_tags['tags'])
            # Limit to top 50 tags
            tag_features = tag_features[:, :min(50, tag_features.shape[1])]

            # Combine features (match indices)
            feature_matrix = np.zeros((len(games_with_genres),
                                       genre_features.shape[1] + tag_features.shape[1]))
            feature_matrix[:, :genre_features.shape[1]] = genre_features
            # Only add tag features for games that have both
            common_games = games_with_genres[
                games_with_genres['id'].isin(games_with_tags['id'])
            ]
            # Simplified: just use genre features for now
            feature_matrix = genre_features
        else:
            feature_matrix = genre_features

        # Store features
        self.game_features = pd.DataFrame(
            feature_matrix,
            index=games_with_genres['id']
        )

        # Build user profiles based on games they liked
        self.user_profiles = {}
        for user_id in train_data['user_id'].unique():
            user_data = train_data[
                (train_data['user_id'] == user_id) &
                (train_data['interaction'] == 1)
                ]
            liked_games = user_data['item_id'].tolist()

            # Average feature vector of liked games
            if liked_games:
                liked_features = self.game_features.loc[
                    self.game_features.index.intersection(liked_games)
                ]
                if len(liked_features) > 0:
                    self.user_profiles[user_id] = liked_features.mean(axis=0)

        print(f"✓ Content-based model trained: {len(self.user_profiles)} user profiles")

    def recommend(self, user_id, k=10):
        """
        Recommend games similar to user's profile
        """
        if user_id not in self.user_profiles or self.game_features is None:
            return []

        user_profile = self.user_profiles[user_id]

        # Calculate similarity to all games
        similarities = cosine_similarity(
            user_profile.values.reshape(1, -1),
            self.game_features.values
        )[0]

        # Get top k
        top_indices = np.argsort(similarities)[::-1][:k]
        return self.game_features.index[top_indices].tolist()