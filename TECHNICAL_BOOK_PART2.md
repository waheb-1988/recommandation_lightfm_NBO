## Chapter 7: Feature Engineering for Hybrid Models

### 7.1 Importance of Feature Engineering

**Quote:** "Coming up with features is difficult, time-consuming, requires expert knowledge. 'Applied machine learning' is basically feature engineering." - Andrew Ng

In hybrid recommendation systems like LightFM, **feature quality directly determines model quality**.

### 7.2 User Features for Telecom

#### 7.2.1 Demographic Features

```python
# Basic demographics
demographic_features = [
    'age_group:18-25',
    'age_group:26-35',
    'age_group:36-50',
    'age_group:51-65',
    'age_group:65+',
    'location:urban',
    'location:suburban',
    'location:rural',
    'account_type:individual',
    'account_type:family',
    'account_type:business'
]
```

**Research shows:**
- Age is strong predictor for data usage (younger → more data)
- Location affects network quality perception
- Business accounts have different needs (reliability > price)

**Paper:** 📚 **Chen, L., et al. (2020). "Understanding mobile data usage patterns in telecommunications." IEEE Communications.**

#### 7.2.2 Usage-Based Features

**Data Intensity:**

```python
def create_data_intensity_features(usage_df):
    """
    Bin data usage into intensity levels
    """
    # Quantile-based binning
    usage_df['data_intensity'] = pd.qcut(
        usage_df['avg_data_GB'],
        q=5,
        labels=['very_low', 'low', 'medium', 'high', 'very_high'],
        duplicates='drop'
    )

    # Alternative: Domain-knowledge based
    usage_df['data_category'] = pd.cut(
        usage_df['avg_data_GB'],
        bins=[0, 1, 5, 15, 50, np.inf],
        labels=['minimal', 'light', 'moderate', 'heavy', 'power_user']
    )

    return usage_df
```

**Voice Usage Patterns:**

```python
def create_call_features(call_logs):
    """
    Extract calling behavior patterns
    """
    features = {}

    # Call intensity
    features['call_intensity'] = pd.qcut(
        call_logs.groupby('client_id')['duration'].sum(),
        q=5,
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )

    # Time patterns
    call_logs['hour'] = pd.to_datetime(call_logs['timestamp']).dt.hour
    business_hours = call_logs[call_logs['hour'].between(9, 17)]

    features['calling_pattern'] = np.where(
        business_hours.groupby('client_id').size() /
        call_logs.groupby('client_id').size() > 0.6,
        'business_hours',
        'personal_hours'
    )

    # International calling
    features['international_caller'] = np.where(
        call_logs[call_logs['international'] == True]
        .groupby('client_id').size() > 5,
        'yes',
        'no'
    )

    return features
```

#### 7.2.3 Behavioral Features

**Engagement Metrics:**

```python
def create_engagement_features(app_logs, support_tickets):
    """
    Customer engagement and interaction features
    """
    engagement = {}

    # App engagement
    engagement['app_active_user'] = np.where(
        app_logs.groupby('client_id')['session_id'].nunique() > 10,
        'yes',
        'no'
    )

    # Self-service capability
    support_online = support_tickets[support_tickets['channel'] == 'online']
    total_support = support_tickets.groupby('client_id').size()
    online_ratio = support_online.groupby('client_id').size() / total_support

    engagement['self_service_user'] = np.where(
        online_ratio > 0.7,
        'yes',
        'no'
    )

    # Feature exploration
    features_used = app_logs.groupby('client_id')['feature'].nunique()
    engagement['feature_explorer'] = pd.qcut(
        features_used,
        q=3,
        labels=['basic', 'moderate', 'power']
    )

    return engagement
```

#### 7.2.4 Derived/Composite Features

**Usage Stability:**

```python
def create_stability_features(usage_history):
    """
    Measure usage pattern stability
    Coefficient of Variation (CV) = std / mean
    """
    usage_stats = usage_history.groupby('client_id')['data_GB'].agg([
        ('mean_usage', 'mean'),
        ('std_usage', 'std')
    ])

    # Coefficient of variation
    usage_stats['cv'] = usage_stats['std_usage'] / (usage_stats['mean_usage'] + 1)

    # Categorize
    usage_stats['usage_stability'] = pd.cut(
        usage_stats['cv'],
        bins=[0, 0.3, 0.6, np.inf],
        labels=['stable', 'moderate', 'variable']
    )

    return usage_stats['usage_stability']
```

**Lifecycle Stage:**

```python
def create_lifecycle_features(clients, usage, tenure_months):
    """
    Customer lifecycle stage based on tenure and value
    """
    # Value score (composite)
    value_score = (
        0.4 * normalize(usage['data_GB']) +
        0.3 * normalize(usage['call_minutes']) +
        0.2 * normalize(clients['account_value']) +
        0.1 * normalize(tenure_months)
    )

    # Lifecycle tier
    lifecycle_tier = pd.qcut(
        value_score,
        q=4,
        labels=['bronze', 'silver', 'gold', 'platinum']
    )

    # Tenure stage
    tenure_stage = pd.cut(
        tenure_months,
        bins=[0, 3, 12, 36, np.inf],
        labels=['new', 'growing', 'mature', 'veteran']
    )

    return lifecycle_tier, tenure_stage
```

**Growth Pattern:**

```python
def create_growth_features(usage_time_series):
    """
    Trend in usage over time
    """
    growth = {}

    # 3-month moving average trend
    for client_id, ts in usage_time_series.groupby('client_id'):
        recent_avg = ts.tail(3)['data_GB'].mean()
        previous_avg = ts.iloc[-6:-3]['data_GB'].mean() if len(ts) >= 6 else recent_avg

        trend = (recent_avg - previous_avg) / (previous_avg + 1)

        if trend > 0.15:
            growth[client_id] = 'growing'
        elif trend < -0.15:
            growth[client_id] = 'declining'
        else:
            growth[client_id] = 'stable'

    return growth
```

#### 7.2.5 Churn Risk Features

**Churn Indicators:**

```python
def create_churn_risk_features(clients, usage, support, payments):
    """
    Features predictive of churn
    """
    churn_signals = {}

    # Usage decline
    usage_decline = (usage['recent_3mo'] - usage['previous_3mo']) / usage['previous_3mo']
    churn_signals['usage_declining'] = (usage_decline < -0.2).astype(int)

    # Support escalation
    recent_complaints = support[support['type'] == 'complaint'].groupby('client_id').size()
    churn_signals['high_complaints'] = (recent_complaints > 2).astype(int)

    # Payment issues
    late_payments = payments[payments['status'] == 'late'].groupby('client_id').size()
    churn_signals['payment_issues'] = (late_payments > 1).astype(int)

    # Contract ending soon
    churn_signals['contract_ending'] = (
        (clients['contract_end_date'] - pd.Timestamp.now()).dt.days < 90
    ).astype(int)

    # Composite churn risk score
    churn_risk_score = (
        0.3 * churn_signals['usage_declining'] +
        0.25 * churn_signals['high_complaints'] +
        0.25 * churn_signals['payment_issues'] +
        0.2 * churn_signals['contract_ending']
    )

    churn_signals['churn_risk_level'] = pd.cut(
        churn_risk_score,
        bins=[0, 0.3, 0.6, 1.0],
        labels=['low', 'medium', 'high']
    )

    return churn_signals
```

**Research Paper:** 📚 **Hadden, J., et al. (2007). "Computer assisted customer churn management: State-of-the-art and future trends." Computers & Operations Research.**

### 7.3 Item (Plan) Features

#### 7.3.1 Plan Attributes

```python
def create_plan_features(plans):
    """
    Basic and derived plan features
    """
    features = {}

    # Price tier (quantile-based)
    features['price_tier'] = pd.qcut(
        plans['monthly_price'],
        q=4,
        labels=['budget', 'standard', 'premium', 'luxury']
    )

    # Data tier
    features['data_tier'] = pd.qcut(
        plans['data_GB'],
        q=4,
        labels=['light', 'moderate', 'heavy', 'unlimited'],
        duplicates='drop'
    )

    # Voice tier
    features['voice_tier'] = pd.qcut(
        plans['call_minutes'],
        q=3,
        labels=['limited', 'ample', 'unlimited'],
        duplicates='drop'
    )

    # Plan type
    features['plan_type'] = plans['type']  # residential/business/prepaid/postpaid

    # Contract length
    features['contract_length'] = pd.cut(
        plans['contract_months'],
        bins=[0, 1, 12, 24, np.inf],
        labels=['no_contract', 'short', 'standard', 'long']
    )

    return features
```

#### 7.3.2 Value Features

**Price-to-Benefit Ratio:**

```python
def create_value_features(plans):
    """
    Value proposition features
    """
    # Data value (GB per dollar)
    plans['data_value'] = plans['data_GB'] / (plans['monthly_price'] + 1)

    # Voice value
    plans['voice_value'] = plans['call_minutes'] / (plans['monthly_price'] + 1)

    # Composite value score
    plans['value_score'] = (
        0.5 * normalize(plans['data_value']) +
        0.3 * normalize(plans['voice_value']) +
        0.2 * normalize(plans['sms_included'])
    )

    # Value category
    plans['value_category'] = pd.qcut(
        plans['value_score'],
        q=3,
        labels=['low_value', 'medium_value', 'high_value']
    )

    return plans
```

#### 7.3.3 Bundle Richness

```python
def create_bundle_features(plans):
    """
    Service bundle comprehensiveness
    """
    # Count included services
    plans['bundle_score'] = (
        (plans['data_GB'] > plans['data_GB'].median()).astype(int) +
        (plans['call_minutes'] > plans['call_minutes'].median()).astype(int) +
        (plans['sms_included'] > plans['sms_included'].median()).astype(int) +
        plans['international_included'].astype(int) +
        plans['roaming_included'].astype(int) +
        plans['streaming_bundle'].astype(int)
    )

    # Bundle richness category
    plans['bundle_type'] = pd.cut(
        plans['bundle_score'],
        bins=[0, 2, 4, 6],
        labels=['basic', 'standard', 'complete']
    )

    return plans
```

### 7.4 Interaction Features

#### 7.4.1 Plan Fit Score

```python
def create_fit_features(user_usage, plan_allowances):
    """
    How well does plan match user needs?
    """
    fit = {}

    # Data fit
    fit['data_utilization'] = user_usage['data_GB'] / plan_allowances['data_GB']

    # Over-users (need upgrade)
    fit['data_constrained'] = (fit['data_utilization'] > 0.9).astype(int)

    # Under-users (overpaying)
    fit['data_wasteful'] = (fit['data_utilization'] < 0.3).astype(int)

    # Voice fit
    fit['voice_utilization'] = user_usage['call_minutes'] / plan_allowances['call_minutes']

    # Overall efficiency
    fit['plan_efficiency'] = (
        0.6 * fit['data_utilization'] +
        0.3 * fit['voice_utilization'] +
        0.1 * (user_usage['sms_sent'] / plan_allowances['sms_included'])
    )

    # Efficiency category
    fit['efficiency_level'] = pd.cut(
        fit['plan_efficiency'],
        bins=[0, 0.5, 0.8, 1.2, np.inf],
        labels=['underutilized', 'good_fit', 'well_matched', 'overutilized']
    )

    return fit
```

#### 7.4.2 Upgrade/Downgrade Potential

```python
def create_migration_features(current_plan, user_tier, plan_tier):
    """
    Potential for plan changes
    """
    migration = {}

    # Tier gap
    tier_mapping = {'budget': 0, 'standard': 1, 'premium': 2, 'luxury': 3}
    current_tier_num = tier_mapping[current_plan['price_tier']]
    user_value_num = tier_mapping[user_tier]

    migration['upgrade_potential'] = max(0, user_value_num - current_tier_num)
    migration['downgrade_risk'] = max(0, current_tier_num - user_value_num)

    # Can afford upgrade?
    migration['can_afford_upgrade'] = (
        (user_tier in ['premium', 'luxury']) or
        (migration['upgrade_potential'] == 1)  # One tier only
    ).astype(int)

    return migration
```

### 7.5 Feature Selection and Dimensionality Reduction

#### 7.5.1 Correlation Analysis

```python
def remove_correlated_features(feature_matrix, threshold=0.9):
    """
    Remove highly correlated features
    """
    import pandas as pd

    # Calculate correlation matrix
    corr_matrix = feature_matrix.corr().abs()

    # Upper triangle of correlations
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation > threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]

    print(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")

    return feature_matrix.drop(columns=to_drop)
```

#### 7.5.2 Feature Importance (Model-Based)

```python
def feature_importance_lightfm(model, dataset, feature_names):
    """
    Estimate feature importance from LightFM embeddings
    """
    # Get feature embeddings
    item_embeddings = model.item_embeddings
    user_embeddings = model.user_embeddings

    # Compute L2 norm of each feature's embedding
    user_importance = {
        feature_names[i]: np.linalg.norm(user_embeddings[i])
        for i in range(len(feature_names))
    }

    item_importance = {
        feature_names[i]: np.linalg.norm(item_embeddings[i])
        for i in range(len(feature_names))
    }

    # Sort by importance
    user_importance_sorted = sorted(
        user_importance.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return user_importance_sorted, item_importance
```

#### 7.5.3 Ablation Study

```python
def ablation_study(features_list, train_data, test_data):
    """
    Remove one feature at a time and measure impact
    """
    baseline_model = train_model(features_list, train_data)
    baseline_score = evaluate(baseline_model, test_data)

    results = {}

    for feature in features_list:
        # Remove feature
        features_subset = [f for f in features_list if f != feature]

        # Train without this feature
        model = train_model(features_subset, train_data)
        score = evaluate(model, test_data)

        # Impact
        impact = baseline_score - score
        results[feature] = impact

        print(f"Feature: {feature}, Impact: {impact:.4f}")

    # Sort by impact (most important first)
    important_features = sorted(
        results.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return important_features
```

**Paper:** 📚 **Melis, G., et al. (2018). "On the state of the art of evaluation in neural language models." ICLR.**

### 7.6 Feature Engineering Best Practices

#### 7.6.1 Domain Knowledge First

**Principle:** Features should encode domain understanding, not just statistical patterns.

**Example:**
```python
# Bad: Just bin everything into quantiles
data_tier = pd.qcut(data_GB, q=5)

# Good: Use domain thresholds
def data_tier_domain(data_GB):
    if data_GB < 1:
        return 'minimal'  # Just messaging/browsing
    elif data_GB < 5:
        return 'light'    # Occasional video
    elif data_GB < 15:
        return 'moderate' # Regular streaming
    elif data_GB < 50:
        return 'heavy'    # Daily HD video
    else:
        return 'power'    # 4K, gaming, hotspot
```

#### 7.6.2 Feature Interaction

**Cross Features:**

```python
def create_interaction_features(user_features, item_features):
    """
    Create meaningful feature interactions
    """
    interactions = {}

    # Segment × Plan Type (strong signal)
    interactions['segment_plan_match'] = (
        (user_features['segment'] == 'business') &
        (item_features['plan_type'] == 'business')
    ).astype(int)

    # High data user × Unlimited plan
    interactions['data_user_unlimited'] = (
        (user_features['data_intensity'] == 'very_high') &
        (item_features['data_tier'] == 'unlimited')
    ).astype(int)

    # Low usage × Budget plan
    interactions['light_user_budget'] = (
        (user_features['data_intensity'] == 'low') &
        (item_features['price_tier'] == 'budget')
    ).astype(int)

    return interactions
```

**Note:** LightFM learns feature interactions implicitly through latent factors, but explicit interactions can help.

#### 7.6.3 Temporal Features

**Seasonality:**

```python
def create_temporal_features(date):
    """
    Time-based features for seasonal patterns
    """
    temporal = {}

    temporal['month'] = date.month
    temporal['quarter'] = f'Q{date.quarter}'
    temporal['is_summer'] = (date.month in [6, 7, 8]).astype(int)
    temporal['is_holiday_season'] = (date.month in [11, 12]).astype(int)

    # Day of week (for usage patterns)
    temporal['day_of_week'] = date.dayofweek
    temporal['is_weekend'] = (date.dayofweek >= 5).astype(int)

    # Days to contract end (for retention offers)
    temporal['days_to_contract_end'] = (
        contract_end_date - date
    ).dt.days

    temporal['in_renewal_window'] = (
        (temporal['days_to_contract_end'] >= 0) &
        (temporal['days_to_contract_end'] <= 90)
    ).astype(int)

    return temporal
```

**Paper:** 📚 **Koren, Y. (2009). "Collaborative filtering with temporal dynamics." KDD.**

#### 7.6.4 Normalization and Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def normalize_features(features, method='standard'):
    """
    Normalize numerical features
    """
    if method == 'standard':
        # Zero mean, unit variance
        scaler = StandardScaler()
    elif method == 'minmax':
        # Scale to [0, 1]
        scaler = MinMaxScaler()

    normalized = scaler.fit_transform(features)

    return normalized, scaler
```

**When to normalize:**
- ✅ For gradient-based methods (neural networks)
- ✅ When features have different scales
- ❌ For tree-based methods (not needed)
- ❌ For LightFM (optional, embedding learning is scale-invariant)

#### 7.6.5 Handling Missing Values

```python
def handle_missing_values(df, strategy='median'):
    """
    Impute missing values
    """
    if strategy == 'median':
        # Use median for numerical
        df_filled = df.fillna(df.median())

    elif strategy == 'mode':
        # Use mode for categorical
        df_filled = df.fillna(df.mode().iloc[0])

    elif strategy == 'indicator':
        # Create missingness indicator
        for col in df.columns:
            if df[col].isnull().any():
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                df[col] = df[col].fillna(df[col].median())

    elif strategy == 'model':
        # Use model-based imputation
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        df_filled = pd.DataFrame(
            imputer.fit_transform(df),
            columns=df.columns
        )

    return df_filled
```

### 7.7 Feature Engineering for Cold Start

#### 7.7.1 New User Features

**Collect at Signup:**

```python
new_user_features = {
    'demographics': [
        'age_group',
        'location',
        'occupation'
    ],
    'stated_preferences': [
        'expected_data_usage',  # Ask: "How much do you stream?"
        'calling_needs',        # "Mostly calling or data?"
        'budget_range'          # "Price range you're comfortable with"
    ],
    'device_info': [
        'device_type',
        'device_capabilities'   # 5G-capable, etc.
    ]
}
```

**Inferred from Early Behavior:**

```python
def infer_new_user_features(first_week_usage):
    """
    Learn from first 7 days
    """
    inferred = {}

    # Quick learners of usage patterns
    if first_week_usage['data_GB'] > 5:
        inferred['likely_data_intensity'] = 'high'
    elif first_week_usage['data_GB'] > 1:
        inferred['likely_data_intensity'] = 'medium'
    else:
        inferred['likely_data_intensity'] = 'low'

    # Peak usage time
    if first_week_usage['peak_hour'] in range(9, 17):
        inferred['usage_pattern'] = 'business_hours'
    else:
        inferred['usage_pattern'] = 'personal'

    return inferred
```

#### 7.7.2 New Plan Features

**Plan Launch Strategies:**

```python
def new_plan_features(plan_specs, similar_plans):
    """
    Features for newly launched plans
    """
    features = {
        # Explicit characteristics
        'price_tier': categorize_price(plan_specs['price']),
        'data_tier': categorize_data(plan_specs['data']),
        'plan_type': plan_specs['type'],

        # Similarity to existing successful plans
        'similar_to_popular': find_most_similar(plan_specs, popular_plans),

        # Competitive positioning
        'price_vs_market': plan_specs['price'] - market_average_price,
        'value_vs_market': plan_specs['value_score'] - market_average_value
    }

    return features
```

### 7.8 Complete Feature Engineering Pipeline

```python
class TelecomFeatureEngineer:
    """
    End-to-end feature engineering for telco recommendations
    """

    def __init__(self):
        self.user_scalers = {}
        self.item_scalers = {}
        self.feature_names = {
            'user': [],
            'item': []
        }

    def engineer_user_features(self, clients, usage, interactions):
        """
        Create comprehensive user feature set
        """
        features = {}

        # 1. Demographics
        features['segment'] = clients['segment']
        features['age_group'] = pd.cut(
            clients['age'],
            bins=[0, 25, 35, 50, 65, 100],
            labels=['18-25', '26-35', '36-50', '51-65', '65+']
        )
        features['location'] = clients['location_type']

        # 2. Usage intensity
        features['data_intensity'] = pd.qcut(
            usage['avg_data_GB'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )

        features['call_intensity'] = pd.qcut(
            usage['avg_call_minutes'],
            q=5,
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )

        # 3. Behavioral
        features['usage_stability'] = self._compute_stability(usage)
        features['growth_pattern'] = self._compute_growth(usage)

        # 4. Lifecycle
        features['tenure_stage'] = pd.cut(
            clients['tenure_months'],
            bins=[0, 3, 12, 36, np.inf],
            labels=['new', 'growing', 'mature', 'veteran']
        )

        # 5. Value tier
        value_score = self._compute_value_score(usage, clients)
        features['lifecycle_tier'] = pd.qcut(
            value_score,
            q=4,
            labels=['bronze', 'silver', 'gold', 'platinum']
        )

        # 6. Churn risk
        features['churn_risk'] = self._compute_churn_risk(
            clients, usage, interactions
        )

        # Store feature names
        self.feature_names['user'] = list(features.keys())

        return pd.DataFrame(features)

    def engineer_item_features(self, plans):
        """
        Create comprehensive plan feature set
        """
        features = {}

        # 1. Basic attributes
        features['plan_type'] = plans['type']
        features['contract_length'] = pd.cut(
            plans['contract_months'],
            bins=[0, 1, 12, 24, np.inf],
            labels=['no_contract', 'short', 'standard', 'long']
        )

        # 2. Price tier
        features['price_tier'] = pd.qcut(
            plans['monthly_price'],
            q=4,
            labels=['budget', 'standard', 'premium', 'luxury'],
            duplicates='drop'
        )

        # 3. Data tier
        features['data_tier'] = pd.qcut(
            plans['data_GB'],
            q=4,
            labels=['light', 'moderate', 'heavy', 'unlimited'],
            duplicates='drop'
        )

        # 4. Value category
        value_score = plans['data_GB'] / (plans['monthly_price'] + 1)
        features['value_category'] = pd.qcut(
            value_score,
            q=3,
            labels=['low_value', 'medium_value', 'high_value'],
            duplicates='drop'
        )

        # 5. Bundle richness
        features['bundle_type'] = self._compute_bundle_richness(plans)

        # Store feature names
        self.feature_names['item'] = list(features.keys())

        return pd.DataFrame(features)

    def create_lightfm_features(self, user_features, item_features):
        """
        Convert to LightFM format
        """
        # User features as list of strings
        user_features_list = []
        for idx, row in user_features.iterrows():
            features = [
                f"{col}:{val}"
                for col, val in row.items()
                if pd.notna(val)
            ]
            user_features_list.append((idx, features))

        # Item features as list of strings
        item_features_list = []
        for idx, row in item_features.iterrows():
            features = [
                f"{col}:{val}"
                for col, val in row.items()
                if pd.notna(val)
            ]
            item_features_list.append((idx, features))

        return user_features_list, item_features_list

    def _compute_stability(self, usage):
        """Helper: Compute usage stability"""
        cv = usage['std_data'] / (usage['mean_data'] + 1)
        return pd.cut(
            cv,
            bins=[0, 0.3, 0.6, np.inf],
            labels=['stable', 'moderate', 'variable']
        )

    def _compute_growth(self, usage):
        """Helper: Compute growth pattern"""
        trend = (usage['recent_3mo'] - usage['previous_3mo']) / (usage['previous_3mo'] + 1)
        return np.where(
            trend > 0.15, 'growing',
            np.where(trend < -0.15, 'declining', 'stable')
        )

    def _compute_value_score(self, usage, clients):
        """Helper: Compute customer value score"""
        return (
            0.4 * self._normalize(usage['avg_data_GB']) +
            0.3 * self._normalize(usage['avg_call_minutes']) +
            0.2 * self._normalize(clients['tenure_months']) +
            0.1 * self._normalize(clients['account_value'])
        )

    def _compute_churn_risk(self, clients, usage, interactions):
        """Helper: Compute churn risk level"""
        # Simplified churn risk computation
        risk_score = 0
        if usage['recent_usage'] < usage['historical_avg'] * 0.7:
            risk_score += 0.3
        if clients['support_tickets'] > 2:
            risk_score += 0.3
        if clients['days_to_contract_end'] < 90:
            risk_score += 0.4

        return pd.cut(
            pd.Series(risk_score),
            bins=[0, 0.3, 0.6, 1.0],
            labels=['low', 'medium', 'high']
        )

    def _compute_bundle_richness(self, plans):
        """Helper: Compute bundle richness"""
        score = (
            (plans['data_GB'] > plans['data_GB'].median()).astype(int) +
            (plans['call_minutes'] > plans['call_minutes'].median()).astype(int) +
            plans['international_included'].astype(int) +
            plans['streaming_bundle'].astype(int)
        )
        return pd.cut(
            score,
            bins=[0, 1, 3, 4],
            labels=['basic', 'standard', 'complete']
        )

    def _normalize(self, series):
        """Helper: Min-max normalization"""
        return (series - series.min()) / (series.max() - series.min() + 1e-8)
```

### 7.9 Feature Engineering Research

**Key Papers:**

📚 **Domingos, P. (2012). "A few useful things to know about machine learning." Communications of the ACM, 55(10), 78-87.**
- Feature engineering is often more important than algorithm choice
- Domain knowledge trumps fancy algorithms

📚 **Zheng, A., & Casari, A. (2018). "Feature Engineering for Machine Learning." O'Reilly Media.**
- Comprehensive guide to feature engineering techniques

📚 **Ke, G., et al. (2017). "LightGBM: A highly efficient gradient boosting decision tree." NIPS.**
- Automated feature engineering with tree-based methods

### 7.10 Summary

**Key Takeaways:**

1. **Quality over Quantity:** 10 well-engineered features > 100 random features
2. **Domain Knowledge:** Use telecom expertise to create meaningful features
3. **Iterative Process:** Feature engineering is iterative—start simple, add complexity
4. **Validation:** Always validate feature importance through ablation studies
5. **Cold Start Focus:** Design features that work for new users/items

**Recommended Feature Set for Telecom:**

**Users (10-15 features):**
- Segment (residential/business)
- Data intensity (5 levels)
- Call intensity (5 levels)
- Usage stability
- Lifecycle tier
- Tenure stage
- Churn risk
- Growth pattern

**Items (8-12 features):**
- Plan type
- Price tier
- Data tier
- Voice tier
- Value category
- Bundle type
- Contract length

**Total: ~20-25 features (sweet spot for LightFM)**

---

# PART III: NEURAL COLLABORATIVE FILTERING

---

## Chapter 8: Deep Learning for Recommendations

### 8.1 From Matrix Factorization to Neural Networks

#### 8.1.1 Limitations of Linear Models

**Matrix Factorization Assumption:**

```
r̂ᵤᵢ = pᵤ · qᵢᵀ = Σⱼ pᵤⱼ qᵢⱼ
```

This is a **linear** interaction between user and item latent factors.

**Problem:** Real-world user-item interactions are often **non-linear**.

**Example:**
- User likes action movies AND comedy movies
- But dislikes action-comedy movies
- Linear MF cannot capture this XOR-like pattern

#### 8.1.2 Neural Networks for Non-Linearity

**Neural Network Approach:**

```
User Embedding ──┐
                 ├──> Neural Network ──> Prediction
Item Embedding ──┘
```

**Advantages:**
- ✅ Learn complex, non-linear patterns
- ✅ Arbitrary function approximation
- ✅ Can incorporate rich features naturally
- ✅ State-of-the-art performance

**Key Insight:** Replace dot product with neural network

**Foundational Paper:** 📚 **Covington, P., Adams, J., & Sargin, E. (2016). "Deep neural networks for YouTube recommendations." RecSys.**

### 8.2 Deep Learning Success Stories

#### 8.2.1 YouTube (2016)

**Architecture:**
- Two-tower network (candidate generation + ranking)
- Hundreds of features
- Billions of training examples

**Results:**
- Significant improvement in watch time
- Better cold start handling
- Powers 70% of YouTube viewing

**Paper:** 📚 **Covington et al. (2016). RecSys.**

#### 8.2.2 Pinterest (2017)

**PinSage - Graph Convolutional Network:**
- 3 billion pins, 18 billion edges
- GCN on massive graph
- Random walk + embeddings

**Results:**
- 150% increase in engagement
- Better content discovery

**Paper:** 📚 **Ying, R., et al. (2018). "Graph convolutional neural networks for web-scale recommender systems." KDD.**

#### 8.2.3 Alibaba (2018)

**Deep Interest Network (DIN):**
- Attention mechanism for user history
- Captures evolving interests
- E-commerce product recommendations

**Results:**
- 10% CTR improvement
- Deployed in production serving billions

**Paper:** 📚 **Zhou, G., et al. (2018). "Deep interest network for click-through rate prediction." KDD.**

### 8.3 Neural Network Architectures for RecSys

#### 8.3.1 Multi-Layer Perceptron (MLP)

**Basic Architecture:**

```
Input: [user_embedding, item_embedding]
   ↓
Layer 1: Dense(256) + ReLU
   ↓
Layer 2: Dense(128) + ReLU
   ↓
Layer 3: Dense(64) + ReLU
   ↓
Output: Dense(1) + Sigmoid → Prediction
```

**Code:**

```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def create_mlp_recommender(num_users, num_items, embedding_dim=64):
    """
    Simple MLP-based recommender
    """
    # Inputs
    user_input = layers.Input(shape=(1,), name='user_input')
    item_input = layers.Input(shape=(1,), name='item_input')

    # Embeddings
    user_embedding = layers.Embedding(
        num_users,
        embedding_dim,
        name='user_embedding'
    )(user_input)
    user_embedding = layers.Flatten()(user_embedding)

    item_embedding = layers.Embedding(
        num_items,
        embedding_dim,
        name='item_embedding'
    )(item_input)
    item_embedding = layers.Flatten()(item_embedding)

    # Concatenate
    concat = layers.Concatenate()([user_embedding, item_embedding])

    # MLP layers
    dense1 = layers.Dense(256, activation='relu')(concat)
    dropout1 = layers.Dropout(0.2)(dense1)

    dense2 = layers.Dense(128, activation='relu')(dropout1)
    dropout2 = layers.Dropout(0.2)(dense2)

    dense3 = layers.Dense(64, activation='relu')(dropout2)

    # Output
    output = layers.Dense(1, activation='sigmoid')(dense3)

    # Model
    model = keras.Model(
        inputs=[user_input, item_input],
        outputs=output,
        name='mlp_recommender'
    )

    return model
```

#### 8.3.2 Wide & Deep Networks

**Architecture:** Combine memorization (wide) and generalization (deep)

**Paper:** 📚 **Cheng, H. T., et al. (2016). "Wide & deep learning for recommender systems." RecSys.**

```
Wide Part (Linear):
  Cross-product features ──> Linear Model ──┐
                                             ├──> Combined Output
Deep Part (Non-linear):                      │
  Embeddings ──> Deep Neural Network ────────┘
```

**Implementation:**

```python
def create_wide_and_deep(num_users, num_items, num_features):
    """
    Wide & Deep architecture
    """
    # Inputs
    user_input = layers.Input(shape=(1,), name='user')
    item_input = layers.Input(shape=(1,), name='item')
    features_input = layers.Input(shape=(num_features,), name='features')

    # Deep part
    user_emb = layers.Embedding(num_users, 64)(user_input)
    user_emb = layers.Flatten()(user_emb)

    item_emb = layers.Embedding(num_items, 64)(item_input)
    item_emb = layers.Flatten()(item_emb)

    deep_concat = layers.Concatenate()([user_emb, item_emb, features_input])
    deep = layers.Dense(128, activation='relu')(deep_concat)
    deep = layers.Dense(64, activation='relu')(deep)

    # Wide part (linear on cross-features)
    wide = layers.Dense(1, activation=None)(features_input)

    # Combine
    combined = layers.Concatenate()([deep, wide])
    output = layers.Dense(1, activation='sigmoid')(combined)

    model = keras.Model(
        inputs=[user_input, item_input, features_input],
        outputs=output
    )

    return model
```

#### 8.3.3 Autoencoders

**Collaborative Denoising Autoencoder:**

**Paper:** 📚 **Wu, Y., et al. (2016). "Collaborative denoising auto-encoders for top-n recommender systems." WSDM.**

```
User-Item Vector → Encoder → Latent → Decoder → Reconstructed Vector
```

**Use Case:** Learn user representations from sparse interaction matrix

```python
def create_autoencoder_recommender(num_items, latent_dim=64):
    """
    Autoencoder for collaborative filtering
    """
    # Input: user's item interaction vector (sparse)
    input_layer = layers.Input(shape=(num_items,), name='user_items')

    # Encoder
    encoded = layers.Dense(256, activation='relu')(input_layer)
    encoded = layers.Dropout(0.5)(encoded)  # Denoising
    encoded = layers.Dense(128, activation='relu')(encoded)
    encoded = layers.Dense(latent_dim, activation='relu', name='latent')(encoded)

    # Decoder
    decoded = layers.Dense(128, activation='relu')(encoded)
    decoded = layers.Dense(256, activation='relu')(decoded)
    decoded = layers.Dense(num_items, activation='sigmoid')(decoded)

    # Model
    autoencoder = keras.Model(inputs=input_layer, outputs=decoded)

    return autoencoder
```

### 8.4 Neural Collaborative Filtering (NCF)

#### 8.4.1 Introduction

**NCF Framework:** Generalized framework for neural recommendation models.

**Foundational Paper:** 📚 **He, X., et al. (2017). "Neural collaborative filtering." WWW.**

**Key Idea:** Replace inner product with neural architecture

**Traditional MF:**
```
ŷᵤᵢ = pᵤᵀqᵢ
```

**NCF:**
```
ŷᵤᵢ = f(pᵤ, qᵢ | Θ)
```

Where `f` is a neural network with parameters `Θ`.

#### 8.4.2 NCF Instantiations

**1. Generalized Matrix Factorization (GMF)**

Learns element-wise product in latent space:

```
h_GMF = pᵤ ⊙ qᵢ
ŷᵤᵢ = α(hᵀ_GMF · h)
```

Where `⊙` is element-wise product, `α` is activation (e.g., sigmoid).

**2. Multi-Layer Perceptron (MLP)**

Concatenate embeddings and pass through MLP:

```
z₀ = [pᵤ, qᵢ]
zₗ = α(Wₗᵀzₗ₋₁ + bₗ)
ŷᵤᵢ = σ(hᵀzₗ)
```

**3. Neural Matrix Factorization (NeuMF)**

Combines GMF and MLP:

```
GMF Output: pᵤᴳ ⊙ qᵢᴳ
MLP Output: φ(concat(pᵤᴹ, qᵢᴹ))

ŷᵤᵢ = σ(hᵀ[pᵤᴳ ⊙ qᵢᴳ, φ(...)])
```

#### 8.4.3 NCF Architecture Diagram

```
User ID ──> User Embedding (GMF) ──────────┐
                                           ⊙ ──┐
Item ID ──> Item Embedding (GMF) ──────────┘   │
                                               │
User ID ──> User Embedding (MLP) ──┐           │
                                   Concat       │
Item ID ──> Item Embedding (MLP) ──┘           │
                 ↓                              │
            MLP Layers                          │
                 ↓                              │
            MLP Output ─────────────────────────┤
                                                │
                                            Concatenate
                                                │
                                           Prediction Layer
                                                │
                                              ŷᵤᵢ
```

#### 8.4.4 Why NCF Works

**1. Non-linearity:** Captures complex patterns MF misses

**2. Flexibility:** Can incorporate various features

**3. Expressiveness:** Universal approximation theorem applies

**4. Empirical Success:** 10-15% improvement over MF on many datasets

### 8.5 Training Deep Recommenders

#### 8.5.1 Loss Functions

**Binary Cross-Entropy (Implicit Feedback):**

```
L = -Σᵤ,ᵢ [yᵤᵢ log ŷᵤᵢ + (1-yᵤᵢ) log(1-ŷᵤᵢ)]
```

**Mean Squared Error (Explicit Ratings):**

```
L = Σᵤ,ᵢ (rᵤᵢ - ŷᵤᵢ)²
```

**Bayesian Personalized Ranking (Pairwise):**

```
L = -Σᵤ,ᵢ,ⱼ log σ(ŷᵤᵢ - ŷᵤⱼ)
```

#### 8.5.2 Optimization

**Adam Optimizer:** Adaptive learning rate, momentum

```python
optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-7
)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC()]
)
```

**Learning Rate Scheduling:**

```python
lr_schedule = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-6
)
```

#### 8.5.3 Regularization

**1. Dropout:**

```python
layers.Dropout(0.2)  # Drop 20% of neurons randomly
```

**2. L2 Regularization:**

```python
layers.Dense(128, activation='relu',
             kernel_regularizer=keras.regularizers.l2(1e-5))
```

**3. Batch Normalization:**

```python
layers.BatchNormalization()
```

**4. Early Stopping:**

```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
```

#### 8.5.4 Negative Sampling

**Challenge:** Implicit feedback has many more negatives than positives.

**Strategy:** Sample negatives during training.

```python
def sample_negatives(user_items, num_items, ratio=4):
    """
    For each positive, sample 'ratio' negatives
    """
    samples = []

    for user, items in user_items.items():
        positives = set(items)
        all_items = set(range(num_items))
        negatives = list(all_items - positives)

        for pos_item in positives:
            # Positive sample
            samples.append((user, pos_item, 1))

            # Negative samples
            neg_items = np.random.choice(negatives, size=ratio, replace=False)
            for neg_item in neg_items:
                samples.append((user, neg_item, 0))

    return samples
```

### 8.6 Advantages and Challenges

#### 8.6.1 Advantages

✅ **Superior Accuracy:** 10-20% improvement over MF on complex datasets

✅ **Flexibility:** Easy to incorporate diverse features

✅ **Non-linear Patterns:** Captures interactions MF cannot

✅ **Transfer Learning:** Pre-trained embeddings

✅ **Multi-modal:** Can combine text, images, metadata

#### 8.6.2 Challenges

❌ **Computational Cost:** Training is slower (GPU needed)

❌ **Data Hungry:** Needs more data than MF to avoid overfitting

❌ **Hyperparameter Tuning:** Many hyperparameters to optimize

❌ **Interpretability:** "Black box" model

❌ **Deployment Complexity:** Larger model size, inference latency

### 8.7 When to Use Deep Learning

**Use Deep Learning When:**

✅ Large dataset (100K+ interactions)
✅ Rich features (user/item metadata, context)
✅ Complex patterns in data
✅ Accuracy is critical (even 1-2% gain matters)
✅ Have GPU resources

**Stick with LightFM/MF When:**

✅ Small dataset (<10K interactions)
✅ Limited features
✅ Need interpretability
✅ CPU-only deployment
✅ Training speed critical

**Hybrid Approach (Best of Both):**

Use LightFM for cold start, NCF for warm users → Ensemble

---

## Chapter 9: Neural Collaborative Filtering Architecture

### 9.1 NCF Framework Overview

**Full NCF Architecture:**

```python
class NCFModel(keras.Model):
    """
    Neural Collaborative Filtering with GMF and MLP paths
    """

    def __init__(self, num_users, num_items, embedding_size=64,
                 mlp_layers=[128, 64, 32], **kwargs):
        super(NCFModel, self).__init__(**kwargs)

        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.mlp_layers = mlp_layers

        # GMF Embeddings
        self.gmf_user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            name='gmf_user_embedding'
        )
        self.gmf_item_embedding = layers.Embedding(
            num_items,
            embedding_size,
            name='gmf_item_embedding'
        )

        # MLP Embeddings (separate from GMF)
        self.mlp_user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            name='mlp_user_embedding'
        )
        self.mlp_item_embedding = layers.Embedding(
            num_items,
            embedding_size,
            name='mlp_item_embedding'
        )

        # MLP Layers
        self.mlp_dense_layers = []
        for units in mlp_layers:
            self.mlp_dense_layers.append(
                layers.Dense(units, activation='relu')
            )

        # Final prediction layer
        self.prediction_layer = layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        user_input, item_input = inputs

        # GMF Part
        gmf_user_latent = self.gmf_user_embedding(user_input)
        gmf_user_latent = layers.Flatten()(gmf_user_latent)

        gmf_item_latent = self.gmf_item_embedding(item_input)
        gmf_item_latent = layers.Flatten()(gmf_item_latent)

        gmf_vector = layers.Multiply()([gmf_user_latent, gmf_item_latent])

        # MLP Part
        mlp_user_latent = self.mlp_user_embedding(user_input)
        mlp_user_latent = layers.Flatten()(mlp_user_latent)

        mlp_item_latent = self.mlp_item_embedding(item_input)
        mlp_item_latent = layers.Flatten()(mlp_item_latent)

        mlp_vector = layers.Concatenate()([mlp_user_latent, mlp_item_latent])

        # Pass through MLP layers
        for dense_layer in self.mlp_dense_layers:
            mlp_vector = dense_layer(mlp_vector)

        # Concatenate GMF and MLP outputs
        combined = layers.Concatenate()([gmf_vector, mlp_vector])

        # Final prediction
        prediction = self.prediction_layer(combined)

        return prediction
```

### 9.2 Component Analysis

#### 9.2.1 Embedding Layers

**Purpose:** Map discrete IDs to continuous vector space

**User Embedding:**
```
user_id (integer) → user_vector (ℝᵈ)

Example:
user_12345 → [0.23, -0.15, 0.67, ..., 0.42]  (64-dim)
```

**Initialization:**

```python
# Random normal initialization (He initialization)
embedding = layers.Embedding(
    input_dim=num_users,
    output_dim=embedding_dim,
    embeddings_initializer='he_normal'
)

# Pretrained (from LightFM, Word2Vec, etc.)
pretrained_weights = load_pretrained_embeddings()
embedding = layers.Embedding(
    input_dim=num_users,
    output_dim=embedding_dim,
    weights=[pretrained_weights],
    trainable=True  # Fine-tune or freeze
)
```

#### 9.2.2 GMF (Generalized Matrix Factorization) Path

**Mathematical Formulation:**

```
h_GMF = pᵤᴳᴹᶠ ⊙ qᵢᴳᴹᶠ
```

Where:
- `⊙`: Element-wise multiplication (Hadamard product)
- `pᵤᴳᴹᶠ`: GMF user embedding
- `qᵢᴳᴹᶠ`: GMF item embedding

**Why Element-wise Product?**

Traditional MF uses inner product:
```
r̂ = pᵤ · qᵢ = Σⱼ pᵤⱼ qᵢⱼ
```

Element-wise product preserves dimension:
```
h = pᵤ ⊙ qᵢ = [pᵤ₁qᵢ₁, pᵤ₂qᵢ₂, ..., pᵤₖqᵢₖ]
```

This allows the model to learn non-uniform weights for each latent dimension.

**Implementation:**

```python
# GMF computation
gmf_user = gmf_user_embedding(user_id)  # Shape: (batch, embedding_dim)
gmf_item = gmf_item_embedding(item_id)  # Shape: (batch, embedding_dim)

gmf_output = layers.Multiply()([gmf_user, gmf_item])  # Element-wise product
```

#### 9.2.3 MLP (Multi-Layer Perceptron) Path

**Mathematical Formulation:**

```
z₀ = [pᵤᴹᴸᴾ ; qᵢᴹᴸᴾ]  (concatenation)

z₁ = α₁(W₁ᵀz₀ + b₁)
z₂ = α₂(W₂ᵀz₁ + b₂)
...
zₗ = αₗ(WₗᵀzₗWhat₋₁ + bₗ)
```

Where:
- `αₗ`: Activation function (ReLU)
- `Wₗ, bₗ`: Weights and biases of layer L

**Layer Design:**

**Tower Structure:** Progressively reduce dimensions

```
Input: 128 (64 + 64 concatenated)
  ↓
Layer 1: 128 units
  ↓
Layer 2: 64 units
  ↓
Layer 3: 32 units
```

**Rationale:** Higher layers learn more abstract representations.

**Implementation:**

```python
# MLP computation
mlp_user = mlp_user_embedding(user_id)
mlp_item = mlp_item_embedding(item_id)

mlp_concat = layers.Concatenate()([mlp_user, mlp_item])  # Shape: (batch, 2*embedding_dim)

# MLP layers
mlp_layer1 = layers.Dense(128, activation='relu')(mlp_concat)
mlp_layer2 = layers.Dense(64, activation='relu')(mlp_layer1)
mlp_layer3 = layers.Dense(32, activation='relu')(mlp_layer2)

mlp_output = mlp_layer3
```

#### 9.2.4 Fusion Layer

**Combine GMF and MLP:**

```python
# Concatenate GMF and MLP outputs
combined = layers.Concatenate()([gmf_output, mlp_output])
# Shape: (batch, embedding_dim + 32)

# Final prediction
prediction = layers.Dense(1, activation='sigmoid')(combined)
```

**Alternative Fusion Strategies:**

**1. Weighted Sum:**
```python
prediction = alpha * gmf_output + (1 - alpha) * mlp_output
```

**2. Attention:**
```python
attention_weights = layers.Dense(2, activation='softmax')([gmf_output, mlp_output])
combined = attention_weights[0] * gmf_output + attention_weights[1] * mlp_output
```

**3. Gating:**
```python
gate = layers.Dense(1, activation='sigmoid')(mlp_output)
combined = gate * gmf_output + (1 - gate) * mlp_output
```

### 9.3 Training the NCF Model

#### 9.3.1 Data Preparation

```python
def prepare_ncf_data(interactions, num_items, negative_ratio=4):
    """
    Prepare training data with negative sampling
    """
    user_ids = []
    item_ids = []
    labels = []

    # Group by user
    user_items = interactions.groupby('user_id')['item_id'].apply(set).to_dict()

    for user_id, pos_items in user_items.items():
        pos_items_set = set(pos_items)
        all_items = set(range(num_items))
        neg_items = list(all_items - pos_items_set)

        # Positive samples
        for item_id in pos_items:
            user_ids.append(user_id)
            item_ids.append(item_id)
            labels.append(1)

        # Negative samples
        num_neg = len(pos_items) * negative_ratio
        sampled_neg = np.random.choice(neg_items, size=min(num_neg, len(neg_items)), replace=False)

        for item_id in sampled_neg:
            user_ids.append(user_id)
            item_ids.append(item_id)
            labels.append(0)

    # Shuffle
    indices = np.arange(len(user_ids))
    np.random.shuffle(indices)

    return (
        np.array(user_ids)[indices],
        np.array(item_ids)[indices],
        np.array(labels)[indices]
    )

# Usage
user_ids, item_ids, labels = prepare_ncf_data(train_interactions, num_items)

# Split into train/val
from sklearn.model_selection import train_test_split
X_train = [user_ids, item_ids]
y_train = labels

X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)
```

#### 9.3.2 Model Compilation

```python
# Create model
model = NCFModel(
    num_users=num_users,
    num_items=num_items,
    embedding_size=64,
    mlp_layers=[128, 64, 32]
)

# Compile
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
```

#### 9.3.3 Training Loop

```python
# Callbacks
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_auc',
        patience=5,
        mode='max',
        restore_best_weights=True
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6
    ),
    keras.callbacks.ModelCheckpoint(
        'best_ncf_model.h5',
        monitor='val_auc',
        mode='max',
        save_best_only=True
    )
]

# Train
history = model.fit(
    [X_train_user, X_train_item],
    y_train,
    validation_data=([X_val_user, X_val_item], y_val),
    epochs=50,
    batch_size=256,
    callbacks=callbacks,
    verbose=1
)
```

#### 9.3.4 Training Monitoring

```python
import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Visualize training progress
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].set_title('Training and Validation Loss')

    # AUC
    axes[1].plot(history.history['auc'], label='Train AUC')
    axes[1].plot(history.history['val_auc'], label='Val AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].legend()
    axes[1].set_title('Training and Validation AUC')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

plot_training_history(history)
```

### 9.4 Inference and Recommendations

#### 9.4.1 Batch Prediction

```python
def recommend_for_users(model, user_ids, num_items, k=10):
    """
    Generate top-K recommendations for multiple users
    """
    recommendations = {}

    for user_id in user_ids:
        # Create input: user repeated, all items
        user_input = np.array([user_id] * num_items)
        item_input = np.arange(num_items)

        # Predict scores
        scores = model.predict([user_input, item_input], verbose=0).flatten()

        # Get top K
        top_k_idx = np.argsort(-scores)[:k]
        top_k_scores = scores[top_k_idx]

        recommendations[user_id] = list(zip(top_k_idx, top_k_scores))

    return recommendations
```

#### 9.4.2 Optimized Inference

**Problem:** Predicting for all items is slow for large catalogs.

**Solution 1: Approximate Nearest Neighbors (ANN)**

```python
import faiss

def build_item_index(model, num_items):
    """
    Build FAISS index for fast retrieval
    """
    # Get item embeddings (from GMF path)
    item_embeddings = model.gmf_item_embedding.get_weights()[0]

    # Create FAISS index
    dimension = item_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product search

    # Add items
    index.add(item_embeddings.astype('float32'))

    return index

def fast_recommend(user_id, model, item_index, k=10):
    """
    Fast top-K recommendation using ANN
    """
    # Get user embedding
    user_embedding = model.gmf_user_embedding(np.array([user_id]))
    user_embedding = user_embedding.numpy().astype('float32')

    # Search
    scores, item_ids = item_index.search(user_embedding, k)

    return list(zip(item_ids[0], scores[0]))
```

**Solution 2: Two-Stage Retrieval**

```python
def two_stage_recommendation(user_id, model, all_items, k=10, candidate_size=100):
    """
    Stage 1: Fast candidate generation (e.g., popularity, categories)
    Stage 2: Accurate ranking with NCF
    """
    # Stage 1: Get candidates (fast heuristic)
    candidates = get_popular_items(candidate_size)  # Or category-based, etc.

    # Stage 2: Rank with NCF
    user_input = np.array([user_id] * len(candidates))
    item_input = np.array(candidates)

    scores = model.predict([user_input, item_input], verbose=0).flatten()

    # Top K
    top_k_idx = np.argsort(-scores)[:k]
    top_k_items = [candidates[i] for i in top_k_idx]
    top_k_scores = scores[top_k_idx]

    return list(zip(top_k_items, top_k_scores))
```

### 9.5 Hyperparameter Tuning

#### 9.5.1 Key Hyperparameters

| Hyperparameter | Typical Range | Impact |
|----------------|---------------|--------|
| **embedding_size** | 32-256 | Model capacity |
| **mlp_layers** | [64,32] to [256,128,64,32] | Complexity |
| **learning_rate** | 0.0001-0.01 | Convergence speed |
| **batch_size** | 128-1024 | Training stability |
| **negative_ratio** | 1-10 | Class balance |
| **dropout_rate** | 0.0-0.5 | Regularization |
| **l2_reg** | 1e-7 to 1e-3 | Overfitting control |

#### 9.5.2 Grid Search

```python
from sklearn.model_selection import ParameterGrid

# Define grid
param_grid = {
    'embedding_size': [32, 64, 128],
    'mlp_layers': [[64, 32], [128, 64, 32], [256, 128, 64]],
    'learning_rate': [0.0001, 0.001, 0.01],
    'negative_ratio': [2, 4, 8]
}

best_auc = 0
best_params = None

for params in ParameterGrid(param_grid):
    print(f"Training with params: {params}")

    # Prepare data
    X_train, y_train = prepare_data(params['negative_ratio'])

    # Create model
    model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_size=params['embedding_size'],
        mlp_layers=params['mlp_layers']
    )

    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(params['learning_rate']),
        loss='binary_crossentropy',
        metrics=[keras.metrics.AUC()]
    )

    # Train
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=20,
        batch_size=256,
        verbose=0
    )

    # Evaluate
    val_auc = max(history.history['val_auc'])

    if val_auc > best_auc:
        best_auc = val_auc
        best_params = params

print(f"Best params: {best_params}")
print(f"Best AUC: {best_auc}")
```

#### 9.5.3 Bayesian Optimization

```python
from sklearn.model_selection import cross_val_score
from scipy.stats import uniform, randint
from skopt import BayesSearchCV  # scikit-optimize

# For Keras models, wrap in KerasClassifier
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

def create_model(embedding_size=64, learning_rate=0.001, mlp_units=64):
    model = NCFModel(
        num_users=num_users,
        num_items=num_items,
        embedding_size=embedding_size,
        mlp_layers=[mlp_units*2, mlp_units, mlp_units//2]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['auc']
    )

    return model

# Wrap model
keras_model = KerasClassifier(build_fn=create_model, epochs=10, batch_size=256, verbose=0)

# Search space
search_spaces = {
    'embedding_size': [32, 64, 128, 256],
    'learning_rate': (1e-4, 1e-2, 'log-uniform'),
    'mlp_units': [32, 64, 128]
}

# Bayesian search
bayes_search = BayesSearchCV(
    keras_model,
    search_spaces,
    n_iter=30,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=1
)

bayes_search.fit(X_train, y_train)

print(f"Best params: {bayes_search.best_params_}")
print(f"Best score: {bayes_search.best_score_}")
```

---

**(The book continues with Chapters 10-35, covering GMF/MLP implementation details, telecom-specific applications, production deployment, evaluation metrics, next-generation algorithms like Transformers, GNNs, and RL, complete code implementations, case studies, and extensive references. Due to length, I've provided a comprehensive foundation. Would you like me to continue with specific chapters?)**