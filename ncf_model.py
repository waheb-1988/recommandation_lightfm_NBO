"""
Neural Collaborative Filtering (NCF) Implementation for Telco Recommendations

Based on the paper: "Neural Collaborative Filtering" (He et al., 2017)
Implements both GMF (Generalized Matrix Factorization) and MLP (Multi-Layer Perceptron) paths.

Enhanced for telco use case with user/item features integration.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NCFModel:
    """
    Neural Collaborative Filtering model with GMF and MLP paths

    Architecture:
    - GMF path: Element-wise product of user and item embeddings
    - MLP path: Concatenated embeddings through multiple dense layers
    - Final layer: Combines both paths for prediction
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_size: int = 64,
        mlp_layers: List[int] = [128, 64, 32],
        learning_rate: float = 0.001,
        l2_reg: float = 0.0001,
        use_features: bool = True
    ):
        """
        Initialize NCF model

        Args:
            num_users: Number of unique users
            num_items: Number of unique items
            embedding_size: Size of embedding vectors
            mlp_layers: List of layer sizes for MLP path
            learning_rate: Learning rate for Adam optimizer
            l2_reg: L2 regularization strength
            use_features: Whether to incorporate side features
        """
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.mlp_layers = mlp_layers
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.use_features = use_features

        self.model = None
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.history = None

    def build_model(
        self,
        user_feature_dim: int = 0,
        item_feature_dim: int = 0
    ) -> keras.Model:
        """
        Build the NCF model architecture

        Args:
            user_feature_dim: Dimension of user features (if using features)
            item_feature_dim: Dimension of item features (if using features)

        Returns:
            Compiled Keras model
        """
        # Input layers
        user_input = layers.Input(shape=(1,), name='user_input', dtype='int32')
        item_input = layers.Input(shape=(1,), name='item_input', dtype='int32')

        # GMF (Generalized Matrix Factorization) path
        gmf_user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_size,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name='gmf_user_embedding'
        )(user_input)
        gmf_user_embedding = layers.Flatten()(gmf_user_embedding)

        gmf_item_embedding = layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_size,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name='gmf_item_embedding'
        )(item_input)
        gmf_item_embedding = layers.Flatten()(gmf_item_embedding)

        # Element-wise product for GMF
        gmf_vector = layers.Multiply()([gmf_user_embedding, gmf_item_embedding])

        # MLP (Multi-Layer Perceptron) path
        mlp_user_embedding = layers.Embedding(
            input_dim=self.num_users,
            output_dim=self.embedding_size,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name='mlp_user_embedding'
        )(user_input)
        mlp_user_embedding = layers.Flatten()(mlp_user_embedding)

        mlp_item_embedding = layers.Embedding(
            input_dim=self.num_items,
            output_dim=self.embedding_size,
            embeddings_regularizer=regularizers.l2(self.l2_reg),
            name='mlp_item_embedding'
        )(item_input)
        mlp_item_embedding = layers.Flatten()(mlp_item_embedding)

        # Concatenate embeddings for MLP
        mlp_vector = layers.Concatenate()([mlp_user_embedding, mlp_item_embedding])

        # Add side features if available
        inputs = [user_input, item_input]
        if self.use_features and (user_feature_dim > 0 or item_feature_dim > 0):
            if user_feature_dim > 0:
                user_features_input = layers.Input(
                    shape=(user_feature_dim,),
                    name='user_features'
                )
                inputs.append(user_features_input)
                mlp_vector = layers.Concatenate()([mlp_vector, user_features_input])

            if item_feature_dim > 0:
                item_features_input = layers.Input(
                    shape=(item_feature_dim,),
                    name='item_features'
                )
                inputs.append(item_features_input)
                mlp_vector = layers.Concatenate()([mlp_vector, item_features_input])

        # MLP layers
        for idx, layer_size in enumerate(self.mlp_layers):
            mlp_vector = layers.Dense(
                layer_size,
                activation='relu',
                kernel_regularizer=regularizers.l2(self.l2_reg),
                name=f'mlp_layer_{idx}'
            )(mlp_vector)
            mlp_vector = layers.BatchNormalization()(mlp_vector)
            mlp_vector = layers.Dropout(0.2)(mlp_vector)

        # Concatenate GMF and MLP paths
        combined = layers.Concatenate()([gmf_vector, mlp_vector])

        # Final prediction layer
        output = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform',
            name='prediction'
        )(combined)

        # Build and compile model
        model = models.Model(inputs=inputs, outputs=output)

        model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='binary_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.AUC(name='auc'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall')
            ]
        )

        self.model = model
        logger.info(f"NCF model built successfully")
        logger.info(f"Total parameters: {model.count_params():,}")

        return model

    def prepare_data(
        self,
        interactions: pd.DataFrame,
        user_col: str = 'client_id',
        item_col: str = 'plan_id',
        rating_col: str = None,
        negative_sampling_ratio: int = 4
    ) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Prepare data for NCF training with negative sampling

        Args:
            interactions: DataFrame with user-item interactions
            user_col: Column name for user IDs
            item_col: Column name for item IDs
            rating_col: Column name for ratings (if None, assumes implicit feedback)
            negative_sampling_ratio: Number of negative samples per positive sample

        Returns:
            Tuple of (train_data, val_data, labels)
        """
        logger.info("Preparing data for NCF...")

        # Encode users and items
        interactions = interactions.copy()
        interactions['user_idx'] = self.user_encoder.fit_transform(
            interactions[user_col]
        )
        interactions['item_idx'] = self.item_encoder.fit_transform(
            interactions[item_col]
        )

        # Update num_users and num_items based on actual encoded data
        self.num_users = len(self.user_encoder.classes_)
        self.num_items = len(self.item_encoder.classes_)

        logger.info(f"Encoded {self.num_users} unique users and {self.num_items} unique items")

        # Create positive interactions
        if rating_col is not None:
            # Explicit feedback: threshold ratings
            threshold = interactions[rating_col].median()
            positive_interactions = interactions[
                interactions[rating_col] >= threshold
            ].copy()
        else:
            # Implicit feedback: all interactions are positive
            positive_interactions = interactions.copy()

        positive_interactions['label'] = 1

        # Generate negative samples
        logger.info(f"Generating negative samples (ratio: {negative_sampling_ratio}:1)...")
        negative_samples = []

        all_items = set(interactions['item_idx'].unique())
        user_items = interactions.groupby('user_idx')['item_idx'].apply(set).to_dict()

        for user_idx in positive_interactions['user_idx'].unique():
            user_positives = user_items.get(user_idx, set())
            user_negatives = list(all_items - user_positives)

            # Sample negative items
            num_negatives = len(
                positive_interactions[positive_interactions['user_idx'] == user_idx]
            ) * negative_sampling_ratio

            if len(user_negatives) < num_negatives:
                sampled_negatives = user_negatives * (num_negatives // len(user_negatives) + 1)
                sampled_negatives = sampled_negatives[:num_negatives]
            else:
                sampled_negatives = np.random.choice(
                    user_negatives,
                    size=num_negatives,
                    replace=False
                )

            for item_idx in sampled_negatives:
                negative_samples.append({
                    'user_idx': user_idx,
                    'item_idx': item_idx,
                    'label': 0
                })

        negative_df = pd.DataFrame(negative_samples)

        # Combine positive and negative samples
        all_samples = pd.concat([
            positive_interactions[['user_idx', 'item_idx', 'label']],
            negative_df
        ], ignore_index=True)

        # Shuffle
        all_samples = all_samples.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Total samples: {len(all_samples):,}")
        logger.info(f"Positive samples: {(all_samples['label'] == 1).sum():,}")
        logger.info(f"Negative samples: {(all_samples['label'] == 0).sum():,}")

        # Verify indices are in correct range
        logger.info(f"User indices range: [{all_samples['user_idx'].min()}, {all_samples['user_idx'].max()}]")
        logger.info(f"Item indices range: [{all_samples['item_idx'].min()}, {all_samples['item_idx'].max()}]")
        logger.info(f"Expected: users [0, {self.num_users}), items [0, {self.num_items})")

        # Sanity check
        if all_samples['user_idx'].max() >= self.num_users:
            logger.error(f"ERROR: user_idx max ({all_samples['user_idx'].max()}) >= num_users ({self.num_users})")
        if all_samples['item_idx'].max() >= self.num_items:
            logger.error(f"ERROR: item_idx max ({all_samples['item_idx'].max()}) >= num_items ({self.num_items})")

        return all_samples

    def train(
        self,
        train_data: pd.DataFrame,
        user_features: pd.DataFrame = None,
        item_features: pd.DataFrame = None,
        validation_split: float = 0.2,
        epochs: int = 20,
        batch_size: int = 256,
        callbacks: List = None
    ):
        """
        Train the NCF model

        Args:
            train_data: DataFrame with user_idx, item_idx, and label columns
            user_features: Optional user features DataFrame
            item_features: Optional item features DataFrame
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
            callbacks: List of Keras callbacks
        """
        # Verify data indices are in valid range
        max_user_idx = train_data['user_idx'].max()
        max_item_idx = train_data['item_idx'].max()
        min_user_idx = train_data['user_idx'].min()
        min_item_idx = train_data['item_idx'].min()

        logger.info(f"Data index ranges: users [{min_user_idx}, {max_user_idx}], items [{min_item_idx}, {max_item_idx}]")
        logger.info(f"Model expects: users [0, {self.num_users}), items [0, {self.num_items})")

        if max_user_idx >= self.num_users or max_item_idx >= self.num_items:
            logger.error(f"Data indices out of range! Max user idx: {max_user_idx} (expected < {self.num_users}), "
                        f"Max item idx: {max_item_idx} (expected < {self.num_items})")
            raise ValueError(f"Data indices exceed model capacity. "
                           f"Max indices in data: user={max_user_idx}, item={max_item_idx}. "
                           f"Model expects: user<{self.num_users}, item<{self.num_items}")

        # Force rebuild if dimensions changed
        if self.model is not None:
            logger.warning("Model already exists, will be replaced with new dimensions")
            self.model = None

        if self.model is None:
            # Determine feature dimensions
            user_feature_dim = len(user_features.columns) - 1 if user_features is not None else 0
            item_feature_dim = len(item_features.columns) - 1 if item_features is not None else 0

            logger.info(f"Building model with {self.num_users} users and {self.num_items} items")
            self.build_model(user_feature_dim, item_feature_dim)

        # Prepare inputs as a list (Keras functional API expects list for multiple inputs)
        X = [
            train_data['user_idx'].values.astype('int32'),
            train_data['item_idx'].values.astype('int32')
        ]

        # Add features if available
        if self.use_features:
            if user_features is not None:
                user_feat_array = user_features.set_index('user_idx').loc[
                    train_data['user_idx']
                ].values.astype('float32')
                X.append(user_feat_array)

            if item_features is not None:
                item_feat_array = item_features.set_index('item_idx').loc[
                    train_data['item_idx']
                ].values.astype('float32')
                X.append(item_feat_array)

        y = train_data['label'].values.astype('float32')

        # Default callbacks
        if callbacks is None:
            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                ),
                keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=2,
                    min_lr=1e-6
                )
            ]

        # Train model
        logger.info(f"Training NCF model for {epochs} epochs...")
        try:
            self.history = self.model.fit(
                X,
                y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=2  # Changed to 2 to avoid Unicode progress bar issues on Windows
            )
        except UnicodeEncodeError:
            # Fallback to silent training if console encoding issues
            logger.warning("Console encoding issue detected, training silently...")
            self.history = self.model.fit(
                X,
                y,
                validation_split=validation_split,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0
            )

        logger.info("Training completed!")

        return self.history

    def predict(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray,
        user_features: pd.DataFrame = None,
        item_features: pd.DataFrame = None
    ) -> np.ndarray:
        """
        Predict scores for user-item pairs

        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
            user_features: Optional user features
            item_features: Optional item features

        Returns:
            Array of predicted scores
        """
        # Encode IDs
        user_indices = self.user_encoder.transform(user_ids).astype('int32')
        item_indices = self.item_encoder.transform(item_ids).astype('int32')

        X = [user_indices, item_indices]

        # Add features if available
        if self.use_features:
            if user_features is not None:
                X.append(user_features.set_index('client_id').loc[user_ids].values.astype('float32'))
            if item_features is not None:
                X.append(item_features.set_index('plan_id').loc[item_ids].values.astype('float32'))

        predictions = self.model.predict(X, verbose=0)
        return predictions.flatten()

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        user_features: pd.DataFrame = None,
        item_features: pd.DataFrame = None,
        exclude_seen: bool = True,
        seen_items: set = None
    ) -> List[Tuple[int, float]]:
        """
        Generate top-N recommendations for a user

        Args:
            user_id: User ID to generate recommendations for
            n: Number of recommendations
            user_features: Optional user features
            item_features: Optional item features
            exclude_seen: Whether to exclude already seen items
            seen_items: Set of items the user has already interacted with

        Returns:
            List of (item_id, score) tuples
        """
        # Get all items
        all_items = self.item_encoder.classes_

        # Exclude seen items if requested
        if exclude_seen and seen_items is not None:
            candidate_items = [item for item in all_items if item not in seen_items]
        else:
            candidate_items = all_items

        # Prepare inputs for prediction
        user_ids = np.array([user_id] * len(candidate_items))
        item_ids = np.array(candidate_items)

        # Get predictions
        scores = self.predict(user_ids, item_ids, user_features, item_features)

        # Sort by score and get top N
        top_indices = np.argsort(scores)[::-1][:n]

        recommendations = [
            (candidate_items[idx], float(scores[idx]))
            for idx in top_indices
        ]

        return recommendations

    def evaluate(
        self,
        test_data: pd.DataFrame,
        user_features: pd.DataFrame = None,
        item_features: pd.DataFrame = None,
        k_values: List[int] = [3, 5, 10]
    ) -> Dict[str, float]:
        """
        Evaluate model performance

        Args:
            test_data: Test dataset with user_idx, item_idx, label
            user_features: Optional user features
            item_features: Optional item features
            k_values: List of K values for precision@K and recall@K

        Returns:
            Dictionary of evaluation metrics
        """
        logger.info("Evaluating NCF model...")

        # Get predictions
        X = [
            test_data['user_idx'].values.astype('int32'),
            test_data['item_idx'].values.astype('int32')
        ]

        if self.use_features:
            if user_features is not None:
                X.append(user_features.set_index('user_idx').loc[
                    test_data['user_idx']
                ].values.astype('float32'))
            if item_features is not None:
                X.append(item_features.set_index('item_idx').loc[
                    test_data['item_idx']
                ].values.astype('float32'))

        y_true = test_data['label'].values
        y_pred = self.model.predict(X, verbose=0).flatten()

        # Overall metrics
        from sklearn.metrics import roc_auc_score, average_precision_score

        metrics = {
            'auc': float(roc_auc_score(y_true, y_pred)),
            'average_precision': float(average_precision_score(y_true, y_pred))
        }

        # Precision@K and Recall@K
        for k in k_values:
            precision_k, recall_k = self._precision_recall_at_k(
                test_data, y_pred, k
            )
            metrics[f'precision@{k}'] = precision_k
            metrics[f'recall@{k}'] = recall_k

        logger.info("Evaluation results:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        return metrics

    def _precision_recall_at_k(
        self,
        test_data: pd.DataFrame,
        predictions: np.ndarray,
        k: int
    ) -> Tuple[float, float]:
        """
        Calculate Precision@K and Recall@K

        Args:
            test_data: Test data with user_idx, item_idx, label
            predictions: Model predictions
            k: Number of top recommendations to consider

        Returns:
            Tuple of (precision@k, recall@k)
        """
        test_data = test_data.copy()
        test_data['pred'] = predictions

        precisions = []
        recalls = []

        for user_idx in test_data['user_idx'].unique():
            user_data = test_data[test_data['user_idx'] == user_idx]

            # Get top K predictions
            top_k = user_data.nlargest(k, 'pred')

            # Calculate precision and recall
            relevant = user_data[user_data['label'] == 1]
            recommended_relevant = top_k[top_k['label'] == 1]

            if len(top_k) > 0:
                precision = len(recommended_relevant) / len(top_k)
                precisions.append(precision)

            if len(relevant) > 0:
                recall = len(recommended_relevant) / len(relevant)
                recalls.append(recall)

        return np.mean(precisions), np.mean(recalls)

    def save(self, filepath: str):
        """Save model and encoders"""
        self.model.save(f"{filepath}_model.h5")

        with open(f"{filepath}_encoders.pkl", 'wb') as f:
            pickle.dump({
                'user_encoder': self.user_encoder,
                'item_encoder': self.item_encoder,
                'num_users': self.num_users,
                'num_items': self.num_items,
                'embedding_size': self.embedding_size,
                'mlp_layers': self.mlp_layers
            }, f)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model and encoders"""
        self.model = keras.models.load_model(f"{filepath}_model.h5")

        with open(f"{filepath}_encoders.pkl", 'rb') as f:
            data = pickle.load(f)
            self.user_encoder = data['user_encoder']
            self.item_encoder = data['item_encoder']
            self.num_users = data['num_users']
            self.num_items = data['num_items']
            self.embedding_size = data['embedding_size']
            self.mlp_layers = data['mlp_layers']

        logger.info(f"Model loaded from {filepath}")


def create_telco_ncf_model(
    num_users: int,
    num_items: int,
    embedding_size: int = 64,
    use_deep: bool = True
) -> NCFModel:
    """
    Create NCF model optimized for telco recommendations

    Args:
        num_users: Number of unique users
        num_items: Number of unique items
        embedding_size: Size of embeddings
        use_deep: Whether to use deep MLP layers

    Returns:
        Configured NCF model
    """
    if use_deep:
        mlp_layers = [256, 128, 64, 32]
        learning_rate = 0.0005
    else:
        mlp_layers = [128, 64, 32]
        learning_rate = 0.001

    model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_size=embedding_size,
        mlp_layers=mlp_layers,
        learning_rate=learning_rate,
        l2_reg=0.00001,
        use_features=True
    )

    return model
