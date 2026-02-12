"""
Quick demo script for Neural Collaborative Filtering

Run this to quickly test NCF on a small subset of data.
"""

import pandas as pd
import numpy as np
from ncf_model import create_telco_ncf_model
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_ncf_demo():
    """Run a quick demo of NCF"""

    logger.info("=" * 60)
    logger.info("NCF Quick Demo")
    logger.info("=" * 60)

    # Load data
    logger.info("Loading data...")
    subscriptions = pd.read_csv("subscriptions.csv")

    # Take a sample for quick demo
    sample_size = 1000
    sample_subs = subscriptions.sample(n=min(sample_size, len(subscriptions)), random_state=42)

    logger.info(f"Using {len(sample_subs)} interactions for demo")

    # Get initial unique counts (will be updated by prepare_data)
    num_users = sample_subs['client_id'].nunique()
    num_items = sample_subs['plan_id'].nunique()

    logger.info(f"Initial counts - Users: {num_users}, Items: {num_items}")

    # Create NCF model with placeholder dimensions
    # These will be updated when prepare_data is called
    logger.info("Creating NCF model...")
    ncf = create_telco_ncf_model(
        num_users=num_users,
        num_items=num_items,
        embedding_size=32,  # Smaller for demo
        use_deep=False
    )

    # Prepare data (this will update num_users and num_items in the model)
    logger.info("Preparing data with negative sampling...")
    train_data = ncf.prepare_data(
        sample_subs,
        user_col='client_id',
        item_col='plan_id',
        negative_sampling_ratio=2  # Lower for demo
    )

    logger.info(f"After encoding - Users: {ncf.num_users}, Items: {ncf.num_items}")

    # Train model (fewer epochs for demo)
    logger.info("Training NCF model...")
    ncf.train(
        train_data=train_data,
        epochs=5,  # Quick training
        batch_size=128,
        validation_split=0.2
    )

    # Evaluate
    logger.info("Evaluating model...")
    metrics = ncf.evaluate(
        train_data.sample(min(200, len(train_data))),  # Sample for quick eval
        k_values=[3, 5]
    )

    logger.info("\n" + "=" * 60)
    logger.info("Demo Results:")
    logger.info("=" * 60)
    for metric, value in metrics.items():
        logger.info(f"{metric}: {value:.4f}")

    # Get sample recommendations
    logger.info("\n" + "=" * 60)
    logger.info("Sample Recommendations:")
    logger.info("=" * 60)

    sample_user = sample_subs['client_id'].iloc[0]
    recommendations = ncf.recommend(
        user_id=sample_user,
        n=5
    )

    logger.info(f"\nTop 5 recommendations for user {sample_user}:")
    for rank, (plan_id, score) in enumerate(recommendations, 1):
        logger.info(f"  {rank}. Plan {plan_id}: {score:.4f}")

    logger.info("\n" + "=" * 60)
    logger.info("Demo completed successfully!")
    logger.info("For full training, run: python train_ncf.py")
    logger.info("=" * 60)


if __name__ == "__main__":
    run_ncf_demo()
