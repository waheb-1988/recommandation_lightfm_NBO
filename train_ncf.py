"""
Training script for Neural Collaborative Filtering (NCF) model

Integrates with existing telco data pipeline and compares with LightFM baseline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import warnings
warnings.filterwarnings('ignore')

from ncf_model import NCFModel, create_telco_ncf_model
from advanced_lightfm_models import AdvancedTelcoRecommender

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_telco_data():
    """Load and prepare telco dataset"""
    logger.info("Loading telco data...")

    clients = pd.read_csv("clients.csv")
    plans = pd.read_csv("plans.csv")
    subscriptions = pd.read_csv("subscriptions.csv")
    usage = pd.read_csv("usage.csv")

    logger.info(f"Loaded {len(clients)} clients, {len(plans)} plans, "
                f"{len(subscriptions)} subscriptions")

    return clients, plans, subscriptions, usage


def prepare_ncf_features(clients, plans, usage):
    """
    Prepare user and item features for NCF

    Args:
        clients: Clients DataFrame
        plans: Plans DataFrame
        usage: Usage DataFrame

    Returns:
        Tuple of (user_features_df, item_features_df)
    """
    logger.info("Engineering features for NCF...")

    # Use AdvancedTelcoRecommender for feature engineering
    recommender = AdvancedTelcoRecommender()

    # Engineer user features
    clients_enhanced = recommender.engineer_user_features(clients, usage)

    # Select numerical features for NCF
    user_feature_cols = [
        'data_used_GB_mean', 'data_used_GB_std',
        'call_minutes_mean', 'call_minutes_std',
        'sms_sent_mean', 'sms_sent_std',
        'value_score'
    ]

    user_features = clients_enhanced[['client_id'] + user_feature_cols].copy()

    # Normalize user features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    user_features[user_feature_cols] = scaler.fit_transform(
        user_features[user_feature_cols]
    )

    # Engineer item features
    plans_enhanced = recommender.engineer_item_features(plans)

    item_feature_cols = [
        'data_GB', 'call_minutes', 'sms_count', 'value_score', 'bundle_richness'
    ]

    item_features = plans_enhanced[['plan_id'] + item_feature_cols].copy()

    # Normalize item features
    item_scaler = StandardScaler()
    item_features[item_feature_cols] = item_scaler.fit_transform(
        item_features[item_feature_cols]
    )

    logger.info(f"User features shape: {user_features.shape}")
    logger.info(f"Item features shape: {item_features.shape}")

    return user_features, item_features, scaler, item_scaler


def train_ncf_model(
    subscriptions,
    user_features,
    item_features,
    embedding_size=64,
    epochs=20,
    batch_size=256,
    negative_sampling_ratio=4
):
    """
    Train NCF model

    Args:
        subscriptions: Subscriptions DataFrame
        user_features: User features DataFrame
        item_features: Item features DataFrame
        embedding_size: Size of embeddings
        epochs: Number of epochs
        batch_size: Batch size
        negative_sampling_ratio: Ratio of negative to positive samples

    Returns:
        Trained NCF model and training history
    """
    logger.info("=" * 60)
    logger.info("Training Neural Collaborative Filtering Model")
    logger.info("=" * 60)

    # Get initial unique counts (will be updated by prepare_data)
    num_users = subscriptions['client_id'].nunique()
    num_items = subscriptions['plan_id'].nunique()

    logger.info(f"Initial number of users: {num_users}")
    logger.info(f"Initial number of items: {num_items}")

    # Create NCF model with initial dimensions
    # These will be updated when prepare_data is called
    ncf = create_telco_ncf_model(
        num_users=num_users,
        num_items=num_items,
        embedding_size=embedding_size,
        use_deep=True
    )

    # Prepare data with negative sampling (this updates num_users and num_items)
    train_data = ncf.prepare_data(
        subscriptions,
        user_col='client_id',
        item_col='plan_id',
        negative_sampling_ratio=negative_sampling_ratio
    )

    logger.info(f"After encoding - Users: {ncf.num_users}, Items: {ncf.num_items}")

    # Split into train and validation
    train_split, val_split = train_test_split(
        train_data,
        test_size=0.2,
        random_state=42,
        stratify=train_data['label']
    )

    # Prepare features for training
    # Map client_id and plan_id to features
    train_user_features = None
    train_item_features = None

    if user_features is not None:
        # Create mapping from user_idx to features
        user_id_to_idx = dict(zip(
            ncf.user_encoder.classes_,
            range(len(ncf.user_encoder.classes_))
        ))
        train_user_features = user_features.copy()
        train_user_features['user_idx'] = train_user_features['client_id'].map(
            lambda x: user_id_to_idx.get(x, -1)
        )
        train_user_features = train_user_features[
            train_user_features['user_idx'] != -1
        ]

    if item_features is not None:
        item_id_to_idx = dict(zip(
            ncf.item_encoder.classes_,
            range(len(ncf.item_encoder.classes_))
        ))
        train_item_features = item_features.copy()
        train_item_features['item_idx'] = train_item_features['plan_id'].map(
            lambda x: item_id_to_idx.get(x, -1)
        )
        train_item_features = train_item_features[
            train_item_features['item_idx'] != -1
        ]

    # Train model
    start_time = datetime.now()

    history = ncf.train(
        train_data=train_split,
        user_features=train_user_features,
        item_features=train_item_features,
        validation_split=0,  # We already split
        epochs=epochs,
        batch_size=batch_size
    )

    training_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"Training completed in {training_time:.2f} seconds")

    # Evaluate on validation set
    val_metrics = ncf.evaluate(
        val_split,
        user_features=train_user_features,
        item_features=train_item_features,
        k_values=[3, 5, 10]
    )

    return ncf, history, val_metrics, training_time


def compare_with_lightfm(
    clients,
    plans,
    subscriptions,
    usage,
    ncf_model,
    ncf_metrics
):
    """
    Compare NCF with LightFM baseline

    Args:
        clients: Clients DataFrame
        plans: Plans DataFrame
        subscriptions: Subscriptions DataFrame
        usage: Usage DataFrame
        ncf_model: Trained NCF model
        ncf_metrics: NCF evaluation metrics

    Returns:
        Comparison DataFrame
    """
    logger.info("=" * 60)
    logger.info("Comparing NCF with LightFM Baseline")
    logger.info("=" * 60)

    # Train LightFM for comparison
    recommender = AdvancedTelcoRecommender()

    clients_enhanced = recommender.engineer_user_features(clients, usage)
    plans_enhanced = recommender.engineer_item_features(plans)
    interactions = recommender.create_interactions_matrix(
        subscriptions, clients_enhanced, plans_enhanced
    )

    # Build dataset
    feature_config = {
        'use_segment': True,
        'use_data_intensity': True,
        'use_call_intensity': True,
        'use_lifecycle_stage': True,
        'use_usage_stability': True,
        'use_plan_type': True,
        'use_price_tier': True,
        'use_data_tier': True,
        'use_value_category': True,
        'use_bundle_type': True
    }

    dataset, user_features, item_features = recommender.build_dataset_with_features(
        interactions, clients_enhanced, plans_enhanced, feature_config
    )

    # Train WARP model (best performing LightFM variant)
    logger.info("Training LightFM WARP model for comparison...")
    start_time = datetime.now()

    warp_model = recommender.train_model(
        'warp',
        dataset,
        user_features,
        item_features,
        epochs=30
    )

    lightfm_training_time = (datetime.now() - start_time).total_seconds()

    # Evaluate LightFM
    lightfm_metrics = recommender.evaluate_model(
        warp_model,
        dataset,
        user_features,
        item_features,
        k=5
    )

    # Create comparison table
    comparison = pd.DataFrame({
        'Model': ['LightFM (WARP)', 'NCF (Neural CF)'],
        'Precision@5': [
            lightfm_metrics.get('precision@5', 0),
            ncf_metrics.get('precision@5', 0)
        ],
        'Recall@5': [
            lightfm_metrics.get('recall@5', 0),
            ncf_metrics.get('recall@5', 0)
        ],
        'AUC': [
            lightfm_metrics.get('auc', 0),
            ncf_metrics.get('auc', 0)
        ],
        'Training Time (s)': [
            lightfm_training_time,
            ncf_metrics.get('training_time', 0)
        ]
    })

    # Calculate improvements
    precision_improvement = (
        (ncf_metrics['precision@5'] - lightfm_metrics['precision@5'])
        / lightfm_metrics['precision@5'] * 100
    )
    recall_improvement = (
        (ncf_metrics['recall@5'] - lightfm_metrics['recall@5'])
        / lightfm_metrics['recall@5'] * 100
    )
    auc_improvement = (
        (ncf_metrics['auc'] - lightfm_metrics['auc'])
        / lightfm_metrics['auc'] * 100
    )

    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)
    print(comparison.to_string(index=False))
    logger.info("\n" + "=" * 60)
    logger.info("IMPROVEMENTS")
    logger.info("=" * 60)
    logger.info(f"Precision@5: {precision_improvement:+.2f}%")
    logger.info(f"Recall@5: {recall_improvement:+.2f}%")
    logger.info(f"AUC: {auc_improvement:+.2f}%")
    logger.info("=" * 60)

    return comparison


def plot_training_history(history, save_path='ncf_training_history.png'):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = ['loss', 'auc', 'precision', 'recall']
    titles = ['Loss', 'AUC', 'Precision', 'Recall']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]

        if metric in history.history:
            ax.plot(history.history[metric], label=f'Train {title}')
        if f'val_{metric}' in history.history:
            ax.plot(history.history[f'val_{metric}'], label=f'Val {title}')

        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.set_title(f'Training {title}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"Training history plot saved to {save_path}")
    plt.close()


def main():
    """Main training pipeline"""
    logger.info("=" * 60)
    logger.info("NCF Training Pipeline for Telco Recommendations")
    logger.info("=" * 60)

    # Load data
    clients, plans, subscriptions, usage = load_telco_data()

    # Prepare features
    user_features, item_features, user_scaler, item_scaler = prepare_ncf_features(
        clients, plans, usage
    )

    # Train NCF model
    ncf_model, history, ncf_metrics, training_time = train_ncf_model(
        subscriptions,
        user_features,
        item_features,
        embedding_size=64,
        epochs=20,
        batch_size=256,
        negative_sampling_ratio=4
    )

    # Add training time to metrics
    ncf_metrics['training_time'] = training_time

    # Plot training history
    plot_training_history(history.history)

    # Save model
    ncf_model.save('ncf_telco_model')
    logger.info("NCF model saved successfully")

    # Compare with LightFM
    comparison = compare_with_lightfm(
        clients, plans, subscriptions, usage,
        ncf_model, ncf_metrics
    )

    # Save comparison
    comparison.to_csv('ncf_vs_lightfm_comparison.csv', index=False)
    logger.info("Comparison saved to ncf_vs_lightfm_comparison.csv")

    logger.info("\n" + "=" * 60)
    logger.info("NCF Training Pipeline Completed Successfully!")
    logger.info("=" * 60)

    return ncf_model, ncf_metrics, comparison


if __name__ == "__main__":
    ncf_model, metrics, comparison = main()
