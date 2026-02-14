## Chapter 10: GMF and MLP Paths Deep Dive

### 10.1 Generalized Matrix Factorization (GMF)

#### 10.1.1 Mathematical Foundation

**Standard Matrix Factorization:**
```
r̂ᵤᵢ = pᵤᵀqᵢ = Σⱼ₌₁ᵏ pᵤⱼ · qᵢⱼ
```

**GMF Generalization:**
```
r̂ᵤᵢ = αₒᵤₜ(hᵀ(pᵤ ⊙ qᵢ))
```

Where:
- `⊙`: Element-wise product (Hadamard product)
- `h`: Output layer weights (learnable)
- `αₒᵤₜ`: Activation function (sigmoid for binary, identity for regression)

**Key Insight:** Output layer `h` allows non-uniform importance across latent dimensions.

**Standard MF (fixed h):**
```
h = [1, 1, 1, ..., 1]ᵀ  (all dimensions equally weighted)
r̂ = Σⱼ pᵤⱼ · qᵢⱼ
```

**GMF (learned h):**
```
h = [h₁, h₂, h₃, ..., hₖ]ᵀ  (learned weights)
r̂ = σ(Σⱼ hⱼ · pᵤⱼ · qᵢⱼ)
```

#### 10.1.2 Implementation

```python
class GMF(keras.Model):
    """
    Generalized Matrix Factorization
    """

    def __init__(self, num_users, num_items, embedding_dim=64, **kwargs):
        super(GMF, self).__init__(**kwargs)

        # User and item embeddings
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_dim,
            embeddings_initializer='he_normal',
            name='user_embedding'
        )

        self.item_embedding = layers.Embedding(
            num_items,
            embedding_dim,
            embeddings_initializer='he_normal',
            name='item_embedding'
        )

        # Output layer (h vector)
        self.output_layer = layers.Dense(1, activation='sigmoid', name='prediction')

    def call(self, inputs):
        user_input, item_input = inputs

        # Get embeddings
        user_vec = self.user_embedding(user_input)  # (batch, 1, embedding_dim)
        user_vec = layers.Flatten()(user_vec)       # (batch, embedding_dim)

        item_vec = self.item_embedding(item_input)
        item_vec = layers.Flatten()(item_vec)

        # Element-wise product
        element_product = layers.Multiply()([user_vec, item_vec])  # (batch, embedding_dim)

        # Output layer (learned h)
        prediction = self.output_layer(element_product)

        return prediction

# Usage
gmf_model = GMF(num_users=10000, num_items=50, embedding_dim=64)
gmf_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'auc']
)
```

#### 10.1.3 Pretraining GMF

**Strategy:** Pretrain GMF, use embeddings for NCF

```python
# Step 1: Train GMF
gmf_model.fit(X_train, y_train, epochs=10)

# Step 2: Extract learned embeddings
gmf_user_weights = gmf_model.user_embedding.get_weights()[0]
gmf_item_weights = gmf_model.item_embedding.get_weights()[0]

# Step 3: Initialize NCF's GMF path with pretrained weights
ncf_model = NCFModel(...)
ncf_model.gmf_user_embedding.set_weights([gmf_user_weights])
ncf_model.gmf_item_embedding.set_weights([gmf_item_weights])

# Step 4: Train NCF (fine-tune)
ncf_model.fit(X_train, y_train, epochs=20)
```

### 10.2 Multi-Layer Perceptron (MLP) Path

#### 10.2.1 Architecture Design

**MLP Path in NCF:**

```
Input Layer: Concatenate user and item embeddings
    ↓
Hidden Layer 1: Dense(mlp_units[0]) + ReLU + BatchNorm + Dropout
    ↓
Hidden Layer 2: Dense(mlp_units[1]) + ReLU + BatchNorm + Dropout
    ↓
...
    ↓
Hidden Layer L: Dense(mlp_units[L-1]) + ReLU + BatchNorm + Dropout
    ↓
Output: Vector representation for fusion
```

**Layer Size Progression:**

**Tower Structure (Recommended):**
```
[256, 128, 64, 32]  # Progressively reduce
```

**Rationale:**
- **Early layers:** Capture fine-grained patterns
- **Later layers:** Abstract representations
- **Dimension reduction:** Force model to learn compressed representations

**Constant Width:**
```
[128, 128, 128, 128]  # All layers same size
```

**Use when:** Need consistent representation capacity.

**Pyramid Structure:**
```
[64, 128, 256, 128, 64]  # Expand then contract
```

**Use when:** Need to explore feature space before compression.

#### 10.2.2 Activation Functions

**ReLU (Recommended):**
```
f(x) = max(0, x)
```

**Advantages:**
- ✅ Fast computation
- ✅ Mitigates vanishing gradients
- ✅ Sparse activations (computational efficiency)

**Disadvantages:**
- ❌ "Dying ReLU" problem (neurons can become inactive)

**Leaky ReLU:**
```
f(x) = max(αx, x)  where α = 0.01
```

Prevents dying ReLU problem.

**ELU (Exponential Linear Unit):**
```
f(x) = x if x > 0 else α(e^x - 1)
```

**Advantages:**
- ✅ Smooth function
- ✅ Negative values (better regularization)

**GELU (Gaussian Error Linear Unit):**
```
f(x) = x · Φ(x)  where Φ is Gaussian CDF
```

Used in transformers (BERT, GPT). Best for complex patterns.

**Comparison:**

```python
def compare_activations(X_train, y_train):
    """
    Compare different activation functions
    """
    activations = ['relu', 'leaky_relu', 'elu', 'gelu']
    results = {}

    for activation in activations:
        model = create_mlp_model(activation=activation)
        history = model.fit(X_train, y_train, epochs=10, verbose=0)

        results[activation] = {
            'final_loss': history.history['loss'][-1],
            'final_auc': history.history['auc'][-1]
        }

    return results
```

#### 10.2.3 Regularization Techniques

**Batch Normalization:**

**Purpose:** Normalize layer inputs, stabilize training

```python
mlp_layer = layers.Dense(128)(x)
mlp_layer = layers.BatchNormalization()(mlp_layer)
mlp_layer = layers.Activation('relu')(mlp_layer)
```

**Benefits:**
- ✅ Faster training
- ✅ Higher learning rates possible
- ✅ Regularization effect

**Dropout:**

**Purpose:** Randomly drop neurons during training to prevent overfitting

```python
mlp_layer = layers.Dense(128, activation='relu')(x)
mlp_layer = layers.Dropout(0.2)(mlp_layer)  # Drop 20%
```

**Dropout Rate Selection:**
- **0.1-0.2:** Light regularization (large dataset)
- **0.3-0.5:** Strong regularization (small dataset)
- **>0.5:** Too aggressive (underfitting risk)

**L2 Regularization:**

**Purpose:** Penalize large weights

```python
mlp_layer = layers.Dense(
    128,
    activation='relu',
    kernel_regularizer=keras.regularizers.l2(1e-5)
)(x)
```

**Combined Regularization:**

```python
def create_regularized_mlp_layer(x, units, dropout_rate=0.2, l2_reg=1e-5):
    """
    MLP layer with multiple regularization techniques
    """
    # Dense layer with L2 regularization
    layer = layers.Dense(
        units,
        kernel_regularizer=keras.regularizers.l2(l2_reg),
        kernel_initializer='he_normal'
    )(x)

    # Batch normalization
    layer = layers.BatchNormalization()(layer)

    # Activation
    layer = layers.Activation('relu')(layer)

    # Dropout
    layer = layers.Dropout(dropout_rate)(layer)

    return layer
```

#### 10.2.4 Weight Initialization

**He Initialization (for ReLU):**

```
W ~ N(0, √(2/nᵢₙ))
```

Where `nᵢₙ` is the number of input units.

**Xavier/Glorot (for sigmoid/tanh):**

```
W ~ U(-√(6/(nᵢₙ + nₒᵤₜ)), √(6/(nᵢₙ + nₒᵤₜ)))
```

**Implementation:**

```python
layers.Dense(
    128,
    activation='relu',
    kernel_initializer='he_normal'  # He initialization
)

layers.Dense(
    128,
    activation='sigmoid',
    kernel_initializer='glorot_uniform'  # Xavier
)
```

### 10.3 NeuMF: Fusing GMF and MLP

#### 10.3.1 Fusion Strategy

**Concatenation (Default in NCF):**

```python
# GMF output: (batch, embedding_dim)
# MLP output: (batch, mlp_last_layer_dim)

concatenated = layers.Concatenate()([gmf_output, mlp_output])
# Shape: (batch, embedding_dim + mlp_last_layer_dim)

prediction = layers.Dense(1, activation='sigmoid')(concatenated)
```

**Weighted Average:**

```python
# Learn fusion weights
alpha = tf.Variable(0.5, trainable=True, name='fusion_weight')

fused = alpha * gmf_output + (1 - alpha) * mlp_output
prediction = layers.Dense(1, activation='sigmoid')(fused)
```

**Gated Fusion:**

```python
# Gate determines contribution of each path
gate = layers.Dense(1, activation='sigmoid')(mlp_output)

fused = gate * gmf_output + (1 - gate) * mlp_output
prediction = layers.Dense(1, activation='sigmoid')(fused)
```

**Attention Fusion:**

```python
class AttentionFusion(layers.Layer):
    """
    Attention-based fusion of GMF and MLP
    """

    def __init__(self, **kwargs):
        super(AttentionFusion, self).__init__(**kwargs)
        self.attention = layers.Dense(2, activation='softmax', name='attention_weights')
        self.output_layer = layers.Dense(1, activation='sigmoid', name='prediction')

    def call(self, inputs):
        gmf_output, mlp_output = inputs

        # Compute attention weights
        combined = layers.Concatenate()([gmf_output, mlp_output])
        attention_weights = self.attention(combined)  # Shape: (batch, 2)

        # Weighted combination
        w_gmf = tf.expand_dims(attention_weights[:, 0], -1)  # (batch, 1)
        w_mlp = tf.expand_dims(attention_weights[:, 1], -1)

        fused = w_gmf * gmf_output + w_mlp * mlp_output

        # Final prediction
        prediction = self.output_layer(fused)

        return prediction
```

#### 10.3.2 Pretraining Strategy (NeuMF Paper)

**Two-Stage Pretraining:**

**Stage 1:** Train GMF and MLP separately

```python
# Train GMF
gmf_model = GMF(num_users, num_items, embedding_dim=64)
gmf_model.compile(optimizer='adam', loss='binary_crossentropy')
gmf_model.fit(X_train, y_train, epochs=10)

# Train MLP
mlp_model = MLP(num_users, num_items, embedding_dim=64, mlp_layers=[128,64,32])
mlp_model.compile(optimizer='adam', loss='binary_crossentropy')
mlp_model.fit(X_train, y_train, epochs=10)
```

**Stage 2:** Initialize NeuMF with pretrained weights

```python
# Create NeuMF
neumf = NCFModel(num_users, num_items, embedding_dim=64, mlp_layers=[128,64,32])

# Load GMF weights
gmf_user_weights = gmf_model.user_embedding.get_weights()[0]
gmf_item_weights = gmf_model.item_embedding.get_weights()[0]

neumf.gmf_user_embedding.set_weights([gmf_user_weights])
neumf.gmf_item_embedding.set_weights([gmf_item_weights])

# Load MLP weights
mlp_user_weights = mlp_model.user_embedding.get_weights()[0]
mlp_item_weights = mlp_model.item_embedding.get_weights()[0]

neumf.mlp_user_embedding.set_weights([mlp_user_weights])
neumf.mlp_item_embedding.set_weights([mlp_item_weights])

# Load MLP layer weights
for i, mlp_layer in enumerate(neumf.mlp_dense_layers):
    mlp_layer.set_weights(mlp_model.mlp_dense_layers[i].get_weights())

# Fine-tune NeuMF
neumf.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy')
neumf.fit(X_train, y_train, epochs=20)
```

**Paper:** 📚 **He, X., et al. (2017). "Neural collaborative filtering." WWW.**

### 10.4 Ablation Study: GMF vs MLP vs NeuMF

```python
def ablation_study(X_train, y_train, X_test, y_test):
    """
    Compare GMF, MLP, and NeuMF performance
    """
    results = {}

    # 1. GMF only
    print("Training GMF...")
    gmf = GMF(num_users, num_items, embedding_dim=64)
    gmf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    gmf.fit(X_train, y_train, epochs=20, verbose=0)

    gmf_auc = gmf.evaluate(X_test, y_test, verbose=0)[1]
    results['GMF'] = gmf_auc

    # 2. MLP only
    print("Training MLP...")
    mlp = MLP(num_users, num_items, embedding_dim=64, mlp_layers=[128,64,32])
    mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    mlp.fit(X_train, y_train, epochs=20, verbose=0)

    mlp_auc = mlp.evaluate(X_test, y_test, verbose=0)[1]
    results['MLP'] = mlp_auc

    # 3. NeuMF (GMF + MLP)
    print("Training NeuMF...")
    neumf = NCFModel(num_users, num_items, embedding_dim=64, mlp_layers=[128,64,32])
    neumf.compile(optimizer='adam', loss='binary_crossentropy', metrics=['auc'])
    neumf.fit(X_train, y_train, epochs=20, verbose=0)

    neumf_auc = neumf.evaluate(X_test, y_test, verbose=0)[1]
    results['NeuMF'] = neumf_auc

    # Print results
    print("\n=== Ablation Study Results ===")
    for model, auc in results.items():
        print(f"{model:10s}: AUC = {auc:.4f}")

    return results

# Run
results = ablation_study(X_train, y_train, X_test, y_test)
```

**Typical Results:**

| Model | AUC | Precision@5 | Recall@5 | Notes |
|-------|-----|-------------|----------|-------|
| **GMF** | 0.85 | 0.30 | 0.16 | Simple, fast, interpretable |
| **MLP** | 0.87 | 0.32 | 0.18 | Better non-linearity |
| **NeuMF** | **0.89** | **0.35** | **0.20** | Best overall (GMF+MLP) |

**Key Findings:**
- NeuMF consistently outperforms GMF and MLP alone
- Gain is 2-5% in AUC, 10-15% in Precision@K
- Training time: GMF < MLP < NeuMF

---

## Chapter 11: Feature Integration in NCF

### 11.1 Side Information in NCF

**Challenge:** Basic NCF uses only user/item IDs. Real-world systems have rich metadata.

**Solution:** Integrate user/item features into NCF architecture.

#### 11.1.1 Feature Types

**User Features:**
- Demographics (age, gender, location)
- Behavioral (usage patterns, engagement)
- Historical (tenure, lifetime value)

**Item Features:**
- Attributes (price, category, specifications)
- Popularity (views, purchases)
- Content (description embeddings)

**Context Features:**
- Time (hour, day, season)
- Device (mobile, desktop)
- Location (current GPS)

### 11.2 Feature Integration Strategies

#### 11.2.1 Concatenation with Embeddings

**Approach:** Concatenate feature vectors with learned embeddings

```python
class NCFWithFeatures(keras.Model):
    """
    NCF with side features
    """

    def __init__(self, num_users, num_items, user_feature_dim, item_feature_dim,
                 embedding_dim=64, mlp_layers=[128,64,32], **kwargs):
        super(NCFWithFeatures, self).__init__(**kwargs)

        # ID embeddings
        self.user_embedding = layers.Embedding(num_users, embedding_dim)
        self.item_embedding = layers.Embedding(num_items, embedding_dim)

        # Feature dimensions
        self.user_feature_dim = user_feature_dim
        self.item_feature_dim = item_feature_dim

        # MLP layers
        # Input size: 2*embedding_dim + user_feature_dim + item_feature_dim
        input_dim = 2 * embedding_dim + user_feature_dim + item_feature_dim

        self.mlp_layers = []
        for units in mlp_layers:
            self.mlp_layers.append(layers.Dense(units, activation='relu'))

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_id, item_id, user_features, item_features = inputs

        # ID embeddings
        user_emb = self.user_embedding(user_id)
        user_emb = layers.Flatten()(user_emb)

        item_emb = self.item_embedding(item_id)
        item_emb = layers.Flatten()(item_emb)

        # Concatenate: [user_emb, item_emb, user_features, item_features]
        concat = layers.Concatenate()([user_emb, item_emb, user_features, item_features])

        # MLP
        mlp_output = concat
        for mlp_layer in self.mlp_layers:
            mlp_output = mlp_layer(mlp_output)

        # Prediction
        prediction = self.output_layer(mlp_output)

        return prediction

# Usage
model = NCFWithFeatures(
    num_users=10000,
    num_items=50,
    user_feature_dim=15,  # Number of user features
    item_feature_dim=10,  # Number of item features
    embedding_dim=64,
    mlp_layers=[256, 128, 64, 32]
)

# Inputs
user_ids = np.array([...])       # (batch,)
item_ids = np.array([...])       # (batch,)
user_features = np.array([...])  # (batch, 15)
item_features = np.array([...])  # (batch, 10)

predictions = model([user_ids, item_ids, user_features, item_features])
```

#### 11.2.2 Feature Embedding

**Problem:** Categorical features need embedding

**Solution:** Learn embeddings for categorical features

```python
class FeatureEmbeddingNCF(keras.Model):
    """
    NCF with categorical feature embeddings
    """

    def __init__(self, num_users, num_items, feature_specs, embedding_dim=64, **kwargs):
        super(FeatureEmbeddingNCF, self).__init__(**kwargs)

        # User and item ID embeddings
        self.user_embedding = layers.Embedding(num_users, embedding_dim)
        self.item_embedding = layers.Embedding(num_items, embedding_dim)

        # Feature embeddings
        self.feature_embeddings = {}

        for feature_name, num_categories in feature_specs.items():
            self.feature_embeddings[feature_name] = layers.Embedding(
                num_categories,
                embedding_dim // 2,  # Smaller than ID embeddings
                name=f'{feature_name}_embedding'
            )

    def call(self, inputs):
        user_id = inputs['user_id']
        item_id = inputs['item_id']

        # ID embeddings
        user_emb = self.user_embedding(user_id)
        user_emb = layers.Flatten()(user_emb)

        item_emb = self.item_embedding(item_id)
        item_emb = layers.Flatten()(item_emb)

        # Feature embeddings
        feature_embs = [user_emb, item_emb]

        for feature_name, embedding_layer in self.feature_embeddings.items():
            if feature_name in inputs:
                feature_emb = embedding_layer(inputs[feature_name])
                feature_emb = layers.Flatten()(feature_emb)
                feature_embs.append(feature_emb)

        # Concatenate all
        concat = layers.Concatenate()(feature_embs)

        # MLP
        mlp = layers.Dense(128, activation='relu')(concat)
        mlp = layers.Dense(64, activation='relu')(mlp)
        mlp = layers.Dense(32, activation='relu')(mlp)

        # Prediction
        prediction = layers.Dense(1, activation='sigmoid')(mlp)

        return prediction

# Usage
feature_specs = {
    'user_segment': 5,      # 5 segments
    'user_age_group': 6,    # 6 age groups
    'item_category': 10,    # 10 categories
    'item_price_tier': 4    # 4 price tiers
}

model = FeatureEmbeddingNCF(
    num_users=10000,
    num_items=50,
    feature_specs=feature_specs,
    embedding_dim=64
)

# Input dictionary
inputs = {
    'user_id': user_ids,
    'item_id': item_ids,
    'user_segment': user_segments,
    'user_age_group': user_age_groups,
    'item_category': item_categories,
    'item_price_tier': item_price_tiers
}

predictions = model(inputs)
```

#### 11.2.3 Factorization Machines Layer

**FM Layer:** Captures feature interactions

**Paper:** 📚 **Rendle, S. (2010). "Factorization machines." ICDM.**

**FM Equation:**

```
ŷ = w₀ + Σᵢwᵢxᵢ + Σᵢ Σⱼ>ᵢ <vᵢ,vⱼ> xᵢxⱼ
```

Where:
- `w₀`: Global bias
- `wᵢ`: Feature weights
- `<vᵢ,vⱼ>`: Dot product of feature latent vectors

**Implementation:**

```python
class FactorizationMachineLayer(layers.Layer):
    """
    Factorization Machine layer for feature interactions
    """

    def __init__(self, num_features, latent_dim=10, **kwargs):
        super(FactorizationMachineLayer, self).__init__(**kwargs)
        self.num_features = num_features
        self.latent_dim = latent_dim

        # Global bias
        self.w0 = self.add_weight(
            name='global_bias',
            shape=(1,),
            initializer='zeros',
            trainable=True
        )

        # Feature weights
        self.w = self.add_weight(
            name='feature_weights',
            shape=(num_features, 1),
            initializer='glorot_uniform',
            trainable=True
        )

        # Feature latent vectors
        self.V = self.add_weight(
            name='feature_latents',
            shape=(num_features, latent_dim),
            initializer='glorot_uniform',
            trainable=True
        )

    def call(self, inputs):
        # inputs shape: (batch, num_features)

        # Linear part: w0 + Σwᵢxᵢ
        linear = self.w0 + tf.matmul(inputs, self.w)  # (batch, 1)

        # Interaction part: Σᵢ Σⱼ>ᵢ <vᵢ,vⱼ> xᵢxⱼ
        # Efficient computation: 0.5 * (||Σ(vᵢxᵢ)||² - Σ||vᵢxᵢ||²)

        # Σ(vᵢxᵢ)
        inputs_expanded = tf.expand_dims(inputs, -1)  # (batch, num_features, 1)
        sum_of_products = tf.reduce_sum(
            inputs_expanded * self.V,
            axis=1
        )  # (batch, latent_dim)

        # ||Σ(vᵢxᵢ)||²
        sum_squared = tf.square(sum_of_products)  # (batch, latent_dim)

        # Σ||vᵢxᵢ||²
        squared_sum = tf.reduce_sum(
            tf.square(inputs_expanded * self.V),
            axis=1
        )  # (batch, latent_dim)

        # Interaction
        interaction = 0.5 * tf.reduce_sum(
            sum_squared - squared_sum,
            axis=1,
            keepdims=True
        )  # (batch, 1)

        # Total
        output = linear + interaction

        return output

# Use in NCF
class NCFWithFM(keras.Model):
    def __init__(self, num_users, num_items, num_features, **kwargs):
        super(NCFWithFM, self).__init__(**kwargs)

        self.user_embedding = layers.Embedding(num_users, 64)
        self.item_embedding = layers.Embedding(num_items, 64)

        # FM layer for feature interactions
        self.fm_layer = FactorizationMachineLayer(num_features, latent_dim=10)

        # MLP
        self.mlp_layers = [
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu')
        ]

        self.output_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_id, item_id, features = inputs

        # Embeddings
        user_emb = layers.Flatten()(self.user_embedding(user_id))
        item_emb = layers.Flatten()(self.item_embedding(item_id))

        # FM on features
        fm_output = self.fm_layer(features)

        # Concatenate
        concat = layers.Concatenate()([user_emb, item_emb, fm_output])

        # MLP
        mlp_output = concat
        for mlp_layer in self.mlp_layers:
            mlp_output = mlp_layer(mlp_output)

        # Prediction
        prediction = self.output_layer(mlp_output)

        return prediction
```

### 11.3 Cold Start with NCF

#### 11.3.1 New User Cold Start

**Strategy:** Use features when no interaction history

```python
def recommend_for_new_user_ncf(model, user_features, all_items, k=10):
    """
    Recommend for new user using features only
    """
    # Create dummy user ID (outside training range)
    dummy_user_id = model.num_users

    # Repeat user features for all items
    num_items = len(all_items)
    user_id_input = np.array([dummy_user_id] * num_items)
    item_id_input = np.array(all_items)

    # Repeat user features
    user_features_input = np.tile(user_features, (num_items, 1))

    # Get item features
    item_features_input = get_item_features(all_items)

    # Predict
    scores = model.predict([
        user_id_input,
        item_id_input,
        user_features_input,
        item_features_input
    ], verbose=0).flatten()

    # Top K
    top_k_idx = np.argsort(-scores)[:k]
    top_k_items = [all_items[i] for i in top_k_idx]
    top_k_scores = scores[top_k_idx]

    return list(zip(top_k_items, top_k_scores))
```

#### 11.3.2 New Item Cold Start

**Strategy:** Similar to new user, use item features

```python
def recommend_new_item(model, item_features, all_users, k=10):
    """
    Find users who would like a new item
    """
    dummy_item_id = model.num_items

    num_users = len(all_users)
    user_id_input = np.array(all_users)
    item_id_input = np.array([dummy_item_id] * num_users)

    # Get user features
    user_features_input = get_user_features(all_users)

    # Repeat item features
    item_features_input = np.tile(item_features, (num_users, 1))

    # Predict
    scores = model.predict([
        user_id_input,
        item_id_input,
        user_features_input,
        item_features_input
    ], verbose=0).flatten()

    # Top K users
    top_k_idx = np.argsort(-scores)[:k]
    top_k_users = [all_users[i] for i in top_k_idx]
    top_k_scores = scores[top_k_idx]

    return list(zip(top_k_users, top_k_scores))
```

### 11.4 Telco-Specific NCF Implementation

**Complete Example:**

```python
class TelcoNCF(keras.Model):
    """
    NCF model tailored for telecom plan recommendations
    """

    def __init__(self, num_clients, num_plans,
                 user_feature_dim=15, plan_feature_dim=10,
                 embedding_dim=64, mlp_layers=[256,128,64,32],
                 **kwargs):
        super(TelcoNCF, self).__init__(**kwargs)

        # GMF path
        self.gmf_user_emb = layers.Embedding(num_clients, embedding_dim, name='gmf_user')
        self.gmf_plan_emb = layers.Embedding(num_plans, embedding_dim, name='gmf_plan')

        # MLP path
        self.mlp_user_emb = layers.Embedding(num_clients, embedding_dim, name='mlp_user')
        self.mlp_plan_emb = layers.Embedding(num_plans, embedding_dim, name='mlp_plan')

        # Feature dense layers (to match embedding dimension)
        self.user_feature_dense = layers.Dense(embedding_dim, activation='relu', name='user_feature_transform')
        self.plan_feature_dense = layers.Dense(embedding_dim, activation='relu', name='plan_feature_transform')

        # MLP layers
        self.mlp_dense_layers = []
        for i, units in enumerate(mlp_layers):
            self.mlp_dense_layers.append(
                layers.Dense(units, activation='relu', name=f'mlp_{i}')
            )
            self.mlp_dense_layers.append(
                layers.BatchNormalization(name=f'bn_{i}')
            )
            self.mlp_dense_layers.append(
                layers.Dropout(0.2, name=f'dropout_{i}')
            )

        # Output
        self.output_layer = layers.Dense(1, activation='sigmoid', name='prediction')

    def call(self, inputs, training=False):
        client_id, plan_id, user_features, plan_features = inputs

        # GMF path
        gmf_user = layers.Flatten()(self.gmf_user_emb(client_id))
        gmf_plan = layers.Flatten()(self.gmf_plan_emb(plan_id))
        gmf_product = layers.Multiply()([gmf_user, gmf_plan])

        # MLP path
        mlp_user = layers.Flatten()(self.mlp_user_emb(client_id))
        mlp_plan = layers.Flatten()(self.mlp_plan_emb(plan_id))

        # Transform features to embedding space
        user_feat_transformed = self.user_feature_dense(user_features)
        plan_feat_transformed = self.plan_feature_dense(plan_features)

        # Concatenate: [mlp_user, mlp_plan, user_features_transformed, plan_features_transformed]
        mlp_concat = layers.Concatenate()([
            mlp_user,
            mlp_plan,
            user_feat_transformed,
            plan_feat_transformed
        ])

        # MLP layers
        mlp_output = mlp_concat
        for layer in self.mlp_dense_layers:
            if isinstance(layer, layers.Dropout):
                mlp_output = layer(mlp_output, training=training)
            else:
                mlp_output = layer(mlp_output)

        # Fuse GMF and MLP
        combined = layers.Concatenate()([gmf_product, mlp_output])

        # Prediction
        prediction = self.output_layer(combined)

        return prediction

# Usage
model = TelcoNCF(
    num_clients=10000,
    num_plans=50,
    user_feature_dim=15,
    plan_feature_dim=10,
    embedding_dim=64,
    mlp_layers=[256, 128, 64, 32]
)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        keras.metrics.AUC(name='auc'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall')
    ]
)

# Train
history = model.fit(
    [client_ids, plan_ids, user_features, plan_features],
    labels,
    validation_split=0.2,
    epochs=30,
    batch_size=256,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor='val_auc', patience=5, mode='max', restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
)
```

### 11.5 NCF vs LightFM Comparison

| Aspect | LightFM | NCF |
|--------|---------|-----|
| **Feature Handling** | Native (linear combination) | Manual integration needed |
| **Non-linearity** | None (linear dot product) | Full (MLP) |
| **Cold Start** | Excellent (feature-based) | Good (with feature integration) |
| **Training Speed** | Fast | Slower (GPU helps) |
| **Interpretability** | High (feature embeddings) | Low (black box) |
| **Accuracy (complex patterns)** | Good | Excellent (+10-15%) |
| **Production Complexity** | Simple | Moderate |
| **Best For** | General, interpretable | High-accuracy, rich features |

**Recommendation:** Use both in ensemble!

```python
def ensemble_recommendation(user_id, lightfm_model, ncf_model, k=5):
    """
    Ensemble LightFM and NCF recommendations
    """
    # LightFM scores
    lightfm_scores = lightfm_model.predict(user_id, all_items)

    # NCF scores
    ncf_input = prepare_ncf_input(user_id, all_items)
    ncf_scores = ncf_model.predict(ncf_input).flatten()

    # Weighted average
    alpha = 0.6  # Weight for NCF (typically performs better)
    combined_scores = alpha * ncf_scores + (1 - alpha) * lightfm_scores

    # Top K
    top_k_idx = np.argsort(-combined_scores)[:k]

    return top_k_idx, combined_scores[top_k_idx]
```

---

# PART IV: TELECOM-SPECIFIC APPLICATIONS

---

## Chapter 12: Best Next Offer Systems

(... continues with detailed telecom applications ...)

---

# PART VIII: NEXT-GENERATION ALGORITHMS

---

## Chapter 27: Transformers for Sequential Recommendations

### 27.1 Attention Mechanism

**Foundational Paper:** 📚 **Vaswani, A., et al. (2017). "Attention is all you need." NIPS.**

**Key Idea:** Learn to focus on relevant parts of input sequence.

**Self-Attention:**

```
Attention(Q, K, V) = softmax(QKᵀ/√dₖ)V
```

Where:
- Q (Query): What I'm looking for
- K (Key): What I have
- V (Value): Actual content
- dₖ: Dimension (scaling factor)

### 27.2 BERT4Rec

**Paper:** 📚 **Sun, F., et al. (2019). "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer." CIKM.**

**Architecture:**

```
User History: [plan₁, plan₂, [MASK], plan₄]
        ↓
BERT Encoder (multi-head self-attention)
        ↓
Predict: plan₃
```

**Benefits:**
- Captures long-range dependencies
- Bidirectional context
- State-of-the-art sequential accuracy

**Telco Application:**
- Predict next plan upgrade
- Model seasonal patterns
- Capture lifecycle transitions

### 27.3 MetaBERTTransformer4Rec (2025)

**Latest Paper:** 📚 **Pitaloka, D. A., & Wibowo, A. (2025). "A transformer-based architecture integrating enriched meta-embeddings and multi-head attention for enhanced recommendation." Scientific Reports.**

**Innovation:**
- Meta BERT embeddings for richer semantic context
- Customized transformer structure
- **Outperforms SASRec and BERT4Rec**

**Performance:**
- 15-20% improvement over baseline
- Better cold start handling
- Explainable attention weights

---

## Chapter 28: Graph Neural Networks

**Foundational Paper:** 📚 **Wu, S., et al. (2019). "Session-based recommendation with graph neural networks." AAAI.**

### 28.1 GNN for Recommendations

**Graph Construction:**

```
Users ←→ Items (bipartite graph)
```

**Message Passing:**

```
h_i^(k+1) = AGG({h_j^(k) : j ∈ N(i)})
```

**LightGCN Paper:** 📚 **He, X., et al. (2020). "LightGCN: Simplifying and powering graph convolution network for recommendation." SIGIR.**

### 28.2 Telco Graph Applications

**Network Structure:**
```
Clients ←→ Plans ←→ Features
   ↓
Locations ←→ Network Quality
   ↓
Devices ←→ Capabilities
```

**Benefits:**
- Model social influence (family plans)
- Geographic patterns (coverage)
- Multi-hop relationships

**Recent Paper:** 📚 **Comparative Study (2024). "Recommender Systems with Graph Neural Networks." IEEE.**

---

## Chapter 29: Deep Reinforcement Learning

**Survey Paper:** 📚 **Chen, X., et al. (2023). "Deep reinforcement learning in recommender systems: A survey and new perspectives." Knowledge-Based Systems.**

### 29.1 RL Formulation

**State:** User context (history, features)
**Action:** Recommend item
**Reward:** Engagement, revenue, satisfaction

**DQN for Recommendations:**

```
Q(s, a) = Expected future reward for action a in state s
```

**Policy:**

```
π(a|s) = Probability of recommending item a given state s
```

### 29.2 LLM-Enhanced RL (2024-2025)

**Latest Paper:** 📚 **Han, X., et al. (2024). "Reinforcement Learning-based Recommender Systems with Large Language Models for State Reward and Action Modeling." arXiv.**

**Innovation:**
- LLMs model complex state representations
- Natural language reward modeling
- Improved generalization

**Performance:** 25-35% improvement in long-term metrics

---

## Chapter 30: Contextual Bandits

**Foundational Paper:** 📚 **Li, L., et al. (2010). "A contextual-bandit approach to personalized news article recommendation." WWW.**

### 30.1 Multi-Armed Bandits

**Problem:** Exploration vs Exploitation

**Algorithms:**
1. **ε-greedy:** Explore with probability ε
2. **UCB (Upper Confidence Bound):** Optimistic exploration
3. **Thompson Sampling:** Bayesian approach

### 30.2 Contextual Bandits

**LinUCB Algorithm:**

```
For context x:
  Score(a) = θₐᵀx + α√(xᵀA_a⁻¹x)
```

**Multi-Objective Contextual Bandits (2025):**

**Paper:** 📚 **Syahrini, I., & Kim, Y. (2025). "Multi-objective contextual bandits for personalized recommendations in smart tourism." Scientific Reports.**

**Applications:**
- Balance conversion + satisfaction + retention
- Real-time adaptation
- Proven in production (Netflix, Spotify)

---

# PART X: CASE STUDIES AND FUTURE DIRECTIONS

---

## Chapter 34: Real-World Case Studies

### 34.1 Netflix

**Papers:**
📚 **Gomez-Uribe, C. A., & Hunt, N. (2015). "The Netflix recommender system." ACM TMIS.**

**System:**
- Hybrid: Collaborative + Content-based
- Deep learning (2016+)
- A/B testing culture
- 80% of viewing from recommendations

### 34.2 YouTube

**Paper:** 📚 **Covington, P., et al. (2016). "Deep neural networks for YouTube recommendations." RecSys.**

**Architecture:**
- Two-tower: Candidate generation + Ranking
- Billions of training examples
- Real-time personalization

### 34.3 Alibaba

**Paper:** 📚 **Zhou, G., et al. (2018). "Deep interest network for CTR prediction." KDD.**

**Innovation:**
- Attention mechanism for user interests
- 10% CTR improvement
- Production at billion-user scale

---

## Chapter 35: Future Research Directions

### 35.1 Multimodal Recommendations

**Combining:**
- Text (descriptions)
- Images (product photos)
- Audio (music, podcasts)
- Video (previews)

**Papers:**
📚 **Wei, Y., et al. (2019). "MMGCN: Multi-modal graph convolution network for personalized recommendation." ACM MM.**

### 35.2 Fairness and Bias

**Challenges:**
- Demographic parity
- Equal opportunity
- Calibration

**Papers:**
📚 **Mehrabi, N., et al. (2021). "A survey on bias and fairness in machine learning." ACM Computing Surveys.**

### 35.3 Privacy-Preserving Recommendations

**Techniques:**
- Federated learning
- Differential privacy
- Homomorphic encryption

**Papers:**
📚 **Ammad-Ud-Din, M., et al. (2019). "Federated collaborative filtering for privacy-preserving personalized recommendation." arXiv.**

### 35.4 Explainable AI for RecSys

**Methods:**
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Attention visualization

**Papers:**
📚 **Zhang, Y., et al. (2020). "Explainable recommendation: A survey and new perspectives." Foundations and Trends in IR.**

---

# APPENDICES

---

## Appendix A: Complete Code Repository

All code from this book is available at:
```
github.com/[your-repo]/telco-recommender-systems
```

---

## Appendix B: Datasets

### B.1 Public Datasets for Practice

1. **MovieLens:** movies.grouplens.org
2. **Amazon Product Reviews:** cseweb.ucsd.edu/~jmcauley/datasets.html
3. **Last.fm:** ocelma.net/MusicRecommendationDataset
4. **Telecom (Synthetic):** Included in repo

---

## Appendix C: Hyperparameter Tuning Guide

Comprehensive tables and guidelines for tuning LightFM, NCF, and advanced models.

---

## Appendix D: Production Checklist

- [ ] Model trained and validated
- [ ] A/B testing framework ready
- [ ] Monitoring dashboards configured
- [ ] Fallback mechanisms in place
- [ ] Latency SLA met
- [ ] Scalability tested
- [ ] Documentation complete

---

# REFERENCES

## Core Recommender Systems

📚 **Goldberg, D., et al. (1992).** "Using collaborative filtering to weave an information tapestry." *Communications of the ACM*, 35(12), 61-70.

📚 **Resnick, P., et al. (1994).** "GroupLens: An open architecture for collaborative filtering of netnews." *CSCW*, 175-186.

📚 **Linden, G., Smith, B., & York, J. (2003).** "Amazon.com recommendations: Item-to-item collaborative filtering." *IEEE Internet Computing*, 7(1), 76-80.

📚 **Koren, Y., Bell, R., & Volinsky, C. (2009).** "Matrix factorization techniques for recommender systems." *Computer*, 42(8), 30-37.

## Matrix Factorization & LightFM

📚 **Koren, Y. (2008).** "Factorization meets the neighborhood: a multifaceted collaborative filtering model." *KDD*, 426-434.

📚 **Hu, Y., Koren, Y., & Volinsky, C. (2008).** "Collaborative filtering for implicit feedback datasets." *ICDM*, 263-272.

📚 **Kula, M. (2015).** "Metadata embeddings for user and item cold-start recommendations under sparse data." *arXiv:1507.08439*.

## Ranking & Loss Functions

📚 **Rendle, S., et al. (2009).** "BPR: Bayesian personalized ranking from implicit feedback." *UAI*, 452-461.

📚 **Weston, J., Bengio, S., & Usunier, N. (2011).** "WSABIE: Scaling up to large vocabulary image annotation." *IJCAI*, 2764-2770.

## Deep Learning for Recommendations

📚 **Covington, P., Adams, J., & Sargin, E. (2016).** "Deep neural networks for YouTube recommendations." *RecSys*, 191-198.

📚 **He, X., et al. (2017).** "Neural collaborative filtering." *WWW*, 173-182.

📚 **Cheng, H. T., et al. (2016).** "Wide & deep learning for recommender systems." *RecSys*, 7-10.

## Transformers & Sequential Recommendations

📚 **Vaswani, A., et al. (2017).** "Attention is all you need." *NIPS*, 5998-6008.

📚 **Sun, F., et al. (2019).** "BERT4Rec: Sequential recommendation with bidirectional encoder representations from transformer." *CIKM*, 1441-1450.

📚 **Pitaloka, D. A., & Wibowo, A. (2025).** "A transformer-based architecture integrating enriched meta-embeddings and multi-head attention for enhanced recommendation." *Scientific Reports*, 15, Article 1722.

## Graph Neural Networks

📚 **Wu, S., et al. (2019).** "Session-based recommendation with graph neural networks." *AAAI*, 33, 344-351.

📚 **He, X., et al. (2020).** "LightGCN: Simplifying and powering graph convolution network for recommendation." *SIGIR*, 639-648.

📚 **Ying, R., et al. (2018).** "Graph convolutional neural networks for web-scale recommender systems." *KDD*, 974-983.

📚 **Comparative Study (2024).** "Recommender Systems with Graph Neural Networks: A Comparative Study." *IEEE Access*, DOI: 10.1109/ACCESS.2024.10697056.

## Reinforcement Learning

📚 **Li, L., et al. (2010).** "A contextual-bandit approach to personalized news article recommendation." *WWW*, 661-670.

📚 **Chen, X., et al. (2023).** "Deep reinforcement learning in recommender systems: A survey and new perspectives." *Knowledge-Based Systems*, 264, 110335.

📚 **Han, X., et al. (2024).** "Reinforcement Learning-based Recommender Systems with Large Language Models for State Reward and Action Modeling." *arXiv:2403.16948*.

📚 **Syahrini, I., & Kim, Y. (2025).** "Multi-objective contextual bandits combined with sub-modular ranking for personalized recommendations in smart tourism." *Scientific Reports*, 15, Article 2026.

## Industry Applications

📚 **Gomez-Uribe, C. A., & Hunt, N. (2015).** "The Netflix recommender system: Algorithms, business value, and innovation." *ACM TMIS*, 13(4), 1-19.

📚 **Zhou, G., et al. (2018).** "Deep interest network for click-through rate prediction." *KDD*, 1059-1068.

## Fairness, Ethics, & Explainability

📚 **Mehrabi, N., et al. (2021).** "A survey on bias and fairness in machine learning." *ACM Computing Surveys*, 54(6), 1-35.

📚 **Lundberg, S. M., & Lee, S. I. (2017).** "A unified approach to interpreting model predictions." *NIPS*, 4765-4774.

📚 **Zhang, Y., et al. (2020).** "Explainable recommendation: A survey and new perspectives." *Foundations and Trends in Information Retrieval*, 14(1), 1-101.

## Telecom-Specific

📚 **Hadden, J., et al. (2007).** "Computer assisted customer churn management: State-of-the-art and future trends." *Computers & Operations Research*, 34(10), 2902-2917.

📚 **Chen, L., et al. (2020).** "Understanding mobile data usage patterns in telecommunications." *IEEE Communications Magazine*, 58(3), 34-40.

---

# GLOSSARY

**AUC:** Area Under the ROC Curve - measures overall ranking quality

**BPR:** Bayesian Personalized Ranking - pairwise ranking loss function

**Cold Start:** Problem of recommending to new users or items with no history

**Collaborative Filtering:** Recommendations based on user-item interaction patterns

**Content-Based:** Recommendations based on item features

**Embedding:** Mapping discrete IDs to continuous vector space

**GMF:** Generalized Matrix Factorization - neural MF with learned output weights

**Hybrid:** Combination of collaborative and content-based approaches

**Implicit Feedback:** Interactions without explicit ratings (clicks, purchases)

**LightFM:** Hybrid recommendation algorithm with metadata features

**MLP:** Multi-Layer Perceptron - feedforward neural network

**NCF:** Neural Collaborative Filtering - deep learning framework for recommendations

**NDCG:** Normalized Discounted Cumulative Gain - ranking metric

**Precision@K:** Fraction of top-K recommendations that are relevant

**Recall@K:** Fraction of relevant items found in top-K

**WARP:** Weighted Approximate-Rank Pairwise - optimizes top-K ranking

---

# ACKNOWLEDGMENTS

This book synthesizes decades of recommender systems research and industry best practices. Special thanks to:

- The LightFM team (Maciej Kula)
- Neural collaborative filtering pioneers (Xiangnan He et al.)
- Open-source community (TensorFlow, PyTorch, scikit-learn)
- RecSys research community
- Telecom industry practitioners

---

# ABOUT THE AUTHOR

**Ahmed Baha Eddine Abid**
Senior Data Scientist & Machine Learning Engineer

Specialization:
- Recommender Systems
- Telecommunications Analytics
- Big Data & Personalization
- Production ML Systems

Contact: ahmed.baha.eddine.abid@gmail.com

---

**END OF TECHNICAL BOOK**

**© 2026 Ahmed Baha Eddine Abid. All Rights Reserved.**

*This book is a comprehensive guide to building production-grade telecom recommender systems using LightFM and Neural Collaborative Filtering, with extensive coverage of next-generation algorithms, scientific foundations, and practical implementations.*