"""
Advanced LightFM Models for Telco Recommendation
Implements state-of-the-art techniques for best next offer prediction
"""

import pandas as pd
import numpy as np
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, recall_at_k, auc_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class AdvancedTelcoRecommender:
    """
    Advanced recommender system with multiple LightFM techniques:
    1. Hybrid models with rich features
    2. Multiple loss functions (WARP, BPR, WARP-KOS)
    3. Advanced feature engineering
    4. Cold start handling
    5. Ensemble methods
    """

    def __init__(self, model_config=None):
        self.model_config = model_config or self.get_default_config()
        self.models = {}
        self.dataset = None
        self.feature_names = {}
        self.scaler = StandardScaler()

    @staticmethod
    def get_default_config():
        return {
            'warp': {
                'loss': 'warp',
                'no_components': 64,
                'learning_rate': 0.05,
                'item_alpha': 1e-6,
                'user_alpha': 1e-6,
                'max_sampled': 10,
                'epochs': 30
            },
            'bpr': {
                'loss': 'bpr',
                'no_components': 64,
                'learning_rate': 0.05,
                'item_alpha': 1e-6,
                'user_alpha': 1e-6,
                'epochs': 30
            },
            'warp_kos': {
                'loss': 'warp-kos',
                'no_components': 64,
                'learning_rate': 0.05,
                'item_alpha': 1e-6,
                'user_alpha': 1e-6,
                'k': 5,
                'epochs': 30
            },
            'hybrid_deep': {
                'loss': 'warp',
                'no_components': 128,
                'learning_rate': 0.03,
                'item_alpha': 1e-5,
                'user_alpha': 1e-5,
                'max_sampled': 20,
                'epochs': 50
            }
        }

    def engineer_user_features(self, clients, usage):
        """
        Advanced user feature engineering for telco domain
        """
        # Aggregate usage patterns
        usage_agg = usage.groupby("client_id").agg({
            "data_used_GB": ["mean", "std", "max", "min"],
            "call_minutes": ["mean", "std", "max"],
            "sms_sent": ["mean", "std", "max"]
        }).reset_index()

        usage_agg.columns = ['_'.join(col).strip('_') for col in usage_agg.columns.values]
        usage_agg = usage_agg.rename(columns={'client_id_': 'client_id'})

        # Merge with clients
        clients_enhanced = clients.merge(usage_agg, on="client_id", how="left")
        clients_enhanced = clients_enhanced.fillna(0)

        # Create behavioral features
        clients_enhanced['data_intensity'] = pd.qcut(
            clients_enhanced['data_used_GB_mean'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        ).astype(str)

        clients_enhanced['call_intensity'] = pd.qcut(
            clients_enhanced['call_minutes_mean'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        ).astype(str)

        # Usage variability (high variability = unpredictable usage)
        clients_enhanced['usage_stability'] = np.where(
            clients_enhanced['data_used_GB_std'] / (clients_enhanced['data_used_GB_mean'] + 1) < 0.3,
            'stable',
            'variable'
        )

        # Create composite score
        clients_enhanced['value_score'] = (
            clients_enhanced['data_used_GB_mean'] * 0.4 +
            clients_enhanced['call_minutes_mean'] * 0.3 +
            clients_enhanced['sms_sent_mean'] * 0.3
        )

        # Customer lifecycle stage
        clients_enhanced['lifecycle_stage'] = pd.qcut(
            clients_enhanced['value_score'],
            q=4,
            labels=['bronze', 'silver', 'gold', 'platinum'],
            duplicates='drop'
        ).astype(str)

        return clients_enhanced

    def engineer_item_features(self, plans):
        """
        Advanced item (plan) feature engineering
        """
        plans_enhanced = plans.copy()

        # Price tiers
        plans_enhanced['price_tier'] = pd.qcut(
            plans_enhanced['price_tnd'],
            q=4,
            labels=['budget', 'standard', 'premium', 'luxury'],
            duplicates='drop'
        ).astype(str)

        # Data capacity tiers
        plans_enhanced['data_tier'] = pd.qcut(
            plans_enhanced['data_GB'],
            q=4,
            labels=['light', 'moderate', 'heavy', 'unlimited'],
            duplicates='drop'
        ).astype(str)

        # Value for money score
        plans_enhanced['value_score'] = (
            plans_enhanced['data_GB'] / (plans_enhanced['price_tnd'] + 1)
        )

        plans_enhanced['value_category'] = pd.qcut(
            plans_enhanced['value_score'],
            q=3,
            labels=['low_value', 'medium_value', 'high_value'],
            duplicates='drop'
        ).astype(str)

        # Service bundle complexity
        plans_enhanced['bundle_richness'] = (
            plans_enhanced['data_GB'] > plans_enhanced['data_GB'].median()
        ).astype(int) + (
            plans_enhanced['call_minutes'] > plans_enhanced['call_minutes'].median()
        ).astype(int) + (
            plans_enhanced['sms_count'] > plans_enhanced['sms_count'].median()
        ).astype(int)

        plans_enhanced['bundle_type'] = pd.cut(
            plans_enhanced['bundle_richness'],
            bins=[-1, 0, 1, 3],
            labels=['basic', 'standard', 'complete']
        ).astype(str)

        return plans_enhanced

    def create_interaction_features(self, clients_enhanced, subscriptions, plans_enhanced):
        """
        Create user-item interaction features
        """
        # Get latest subscription for each client
        latest_sub = subscriptions.sort_values("end_date").groupby("client_id").tail(1)

        interactions_df = latest_sub.merge(clients_enhanced, on="client_id", how="left")
        interactions_df = interactions_df.merge(plans_enhanced, on="plan_id", how="left")

        # Calculate upgrade/downgrade potential
        interactions_df['current_price'] = interactions_df['price_tnd']
        interactions_df['price_flexibility'] = interactions_df['value_score_x'] / (interactions_df['current_price'] + 1)

        return interactions_df

    def build_dataset_with_features(self, interactions_df, clients_enhanced, plans_enhanced, feature_config):
        """
        Build LightFM dataset with selected features
        """
        dataset = Dataset()

        # Fit users and items
        dataset.fit(
            (str(x) for x in interactions_df["client_id"]),
            (str(x) for x in interactions_df["plan_id"])
        )

        # User features based on configuration
        user_features_list = []
        if feature_config.get('use_segment', True):
            user_features_list.append('segment')
        if feature_config.get('use_data_intensity', True):
            user_features_list.append('data_intensity')
        if feature_config.get('use_call_intensity', True):
            user_features_list.append('call_intensity')
        if feature_config.get('use_usage_stability', True):
            user_features_list.append('usage_stability')
        if feature_config.get('use_lifecycle_stage', True):
            user_features_list.append('lifecycle_stage')

        # Collect all unique feature values
        all_user_features = []
        for feat in user_features_list:
            all_user_features.extend(clients_enhanced[feat].unique().tolist())

        dataset.fit_partial(
            users=(str(x) for x in interactions_df["client_id"]),
            user_features=all_user_features
        )

        # Item features based on configuration
        item_features_list = []
        if feature_config.get('use_plan_type', True):
            item_features_list.append('plan_type')
        if feature_config.get('use_price_tier', True):
            item_features_list.append('price_tier')
        if feature_config.get('use_data_tier', True):
            item_features_list.append('data_tier')
        if feature_config.get('use_value_category', True):
            item_features_list.append('value_category')
        if feature_config.get('use_bundle_type', True):
            item_features_list.append('bundle_type')

        # Collect all unique item feature values
        all_item_features = []
        for feat in item_features_list:
            all_item_features.extend(plans_enhanced[feat].unique().tolist())

        dataset.fit_partial(
            items=(str(x) for x in interactions_df["plan_id"]),
            item_features=all_item_features
        )

        # Build interactions
        interactions, weights = dataset.build_interactions([
            (str(row["client_id"]), str(row["plan_id"]))
            for _, row in interactions_df.iterrows()
        ])

        # Build user features
        user_features_data = []
        for _, row in clients_enhanced.iterrows():
            features = []
            for feat in user_features_list:
                if feat in row and pd.notna(row[feat]):
                    features.append(str(row[feat]))
            user_features_data.append((str(row["client_id"]), features))

        user_features = dataset.build_user_features(user_features_data)

        # Build item features
        item_features_data = []
        for _, row in plans_enhanced.iterrows():
            features = []
            for feat in item_features_list:
                if feat in row and pd.notna(row[feat]):
                    features.append(str(row[feat]))
            item_features_data.append((str(row["plan_id"]), features))

        item_features = dataset.build_item_features(item_features_data)

        return dataset, interactions, user_features, item_features

    def train_model(self, model_name, interactions, user_features, item_features, num_threads=4):
        """
        Train a specific model variant
        """
        config = self.model_config[model_name]

        model = LightFM(
            loss=config['loss'],
            no_components=config['no_components'],
            learning_rate=config['learning_rate'],
            item_alpha=config.get('item_alpha', 0.0),
            user_alpha=config.get('user_alpha', 0.0),
            max_sampled=config.get('max_sampled', 10),
            k=config.get('k', 5),
            random_state=42
        )

        model.fit(
            interactions,
            user_features=user_features,
            item_features=item_features,
            epochs=config['epochs'],
            num_threads=num_threads,
            verbose=True
        )

        self.models[model_name] = model
        return model

    def evaluate_model(self, model, interactions, user_features, item_features, k=5):
        """
        Evaluate model with multiple metrics
        """
        metrics = {
            'precision@k': precision_at_k(
                model, interactions, user_features=user_features,
                item_features=item_features, k=k
            ).mean(),
            'recall@k': recall_at_k(
                model, interactions, user_features=user_features,
                item_features=item_features, k=k
            ).mean(),
            'auc': auc_score(
                model, interactions, user_features=user_features,
                item_features=item_features
            ).mean()
        }
        return metrics

    def get_recommendations(self, model, dataset, user_id, user_features, item_features, n=5):
        """
        Get top N recommendations for a user
        """
        try:
            uid = dataset.mapping()[0][str(user_id)]
            n_items = len(dataset.mapping()[2])

            scores = model.predict(
                uid,
                np.arange(n_items),
                user_features=user_features,
                item_features=item_features
            )

            top_items = np.argsort(-scores)[:n]

            reverse_item_map = {v: k for k, v in dataset.mapping()[2].items()}
            recommendations = [
                {
                    'plan_id': reverse_item_map[idx],
                    'score': float(scores[idx]),
                    'rank': i + 1
                }
                for i, idx in enumerate(top_items)
            ]

            return recommendations
        except KeyError:
            return []

    def ensemble_predictions(self, dataset, user_id, user_features, item_features,
                            model_weights=None, n=5):
        """
        Ensemble multiple models for robust predictions
        """
        if model_weights is None:
            model_weights = {name: 1.0 for name in self.models.keys()}

        try:
            uid = dataset.mapping()[0][str(user_id)]
            n_items = len(dataset.mapping()[2])

            # Weighted average of predictions
            ensemble_scores = np.zeros(n_items)
            total_weight = sum(model_weights.values())

            for model_name, model in self.models.items():
                weight = model_weights.get(model_name, 1.0)
                scores = model.predict(
                    uid,
                    np.arange(n_items),
                    user_features=user_features,
                    item_features=item_features
                )
                ensemble_scores += (weight / total_weight) * scores

            top_items = np.argsort(-ensemble_scores)[:n]

            reverse_item_map = {v: k for k, v in dataset.mapping()[2].items()}
            recommendations = [
                {
                    'plan_id': reverse_item_map[idx],
                    'score': float(ensemble_scores[idx]),
                    'rank': i + 1
                }
                for i, idx in enumerate(top_items)
            ]

            return recommendations
        except KeyError:
            return []

    def cold_start_recommendation(self, user_features_dict, plans_enhanced, dataset,
                                  item_features, n=5):
        """
        Handle cold start users using content-based features
        """
        # Use average user embedding for cold start
        # Recommend based on item features that match user profile

        # Simple heuristic: recommend high-value plans for the user segment
        segment = user_features_dict.get('segment', 'default')
        data_intensity = user_features_dict.get('data_intensity', 'medium')

        # Filter plans based on user profile
        filtered_plans = plans_enhanced.copy()

        # Score plans based on feature matching
        filtered_plans['match_score'] = 0

        if data_intensity in ['high', 'very_high']:
            filtered_plans['match_score'] += (filtered_plans['data_tier'] == 'unlimited').astype(int) * 2
            filtered_plans['match_score'] += (filtered_plans['data_tier'] == 'heavy').astype(int)
        else:
            filtered_plans['match_score'] += (filtered_plans['price_tier'] == 'budget').astype(int)

        # Get top N based on match score and value
        filtered_plans['final_score'] = (
            filtered_plans['match_score'] * 0.6 +
            filtered_plans['value_score'] * 0.4
        )

        top_plans = filtered_plans.nlargest(n, 'final_score')

        recommendations = [
            {
                'plan_id': str(row['plan_id']),
                'score': float(row['final_score']),
                'rank': i + 1,
                'method': 'cold_start'
            }
            for i, (_, row) in enumerate(top_plans.iterrows())
        ]

        return recommendations


def compare_models(recommender, interactions, user_features, item_features, k_values=[3, 5, 10]):
    """
    Compare all trained models across different metrics
    """
    results = []

    for model_name, model in recommender.models.items():
        for k in k_values:
            metrics = recommender.evaluate_model(
                model, interactions, user_features, item_features, k=k
            )
            results.append({
                'model': model_name,
                'k': k,
                'precision@k': metrics['precision@k'],
                'recall@k': metrics['recall@k'],
                'auc': metrics['auc']
            })

    return pd.DataFrame(results)
