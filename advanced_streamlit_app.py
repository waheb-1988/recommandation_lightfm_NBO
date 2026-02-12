"""
Advanced LightFM Telco Recommendation System - Interactive Streamlit App
Features:
- Multiple model comparison
- Advanced feature engineering
- Hyperparameter tuning
- Ensemble methods
- Cold start handling
- Performance visualization
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from advanced_lightfm_models import AdvancedTelcoRecommender, compare_models

# Page configuration
st.set_page_config(
    page_title="Advanced Telco Recommender",
    page_icon="📱",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 2rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.stTabs [data-baseweb="tab-list"] {
    gap: 24px;
}
.stTabs [data-baseweb="tab"] {
    height: 50px;
    padding: 10px 20px;
    background-color: #f0f2f6;
    border-radius: 5px;
}
.stTabs [aria-selected="true"] {
    background-color: #667eea;
    color: white;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_and_prepare_data():
    """Load and prepare all data with advanced features"""
    # Load data
    clients = pd.read_csv("clients.csv")
    plans = pd.read_csv("plans.csv")
    subscriptions = pd.read_csv("subscriptions.csv")
    usage = pd.read_csv("usage.csv")

    # Initialize recommender
    recommender = AdvancedTelcoRecommender()

    # Engineer features
    clients_enhanced = recommender.engineer_user_features(clients, usage)
    plans_enhanced = recommender.engineer_item_features(plans)
    interactions_df = recommender.create_interaction_features(
        clients_enhanced, subscriptions, plans_enhanced
    )

    return recommender, clients_enhanced, plans_enhanced, subscriptions, usage, interactions_df


def train_models_interface(recommender, interactions_df, clients_enhanced, plans_enhanced):
    """Interface for training models with configuration"""
    st.header("🔧 Model Training & Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Feature Configuration")
        feature_config = {
            'use_segment': st.checkbox("Customer Segment", value=True),
            'use_data_intensity': st.checkbox("Data Usage Intensity", value=True),
            'use_call_intensity': st.checkbox("Call Intensity", value=True),
            'use_usage_stability': st.checkbox("Usage Stability", value=True),
            'use_lifecycle_stage': st.checkbox("Lifecycle Stage", value=True),
            'use_plan_type': st.checkbox("Plan Type", value=True),
            'use_price_tier': st.checkbox("Price Tier", value=True),
            'use_data_tier': st.checkbox("Data Tier", value=True),
            'use_value_category': st.checkbox("Value Category", value=True),
            'use_bundle_type': st.checkbox("Bundle Type", value=True)
        }

    with col2:
        st.subheader("Model Selection")
        selected_models = st.multiselect(
            "Select models to train",
            options=['warp', 'bpr', 'warp_kos', 'hybrid_deep'],
            default=['warp', 'bpr'],
            help="WARP: Weighted Approximate-Rank Pairwise (best for implicit feedback)\n"
                 "BPR: Bayesian Personalized Ranking\n"
                 "WARP-KOS: WARP with k-th order statistics\n"
                 "Hybrid Deep: Deep model with more components"
        )

        num_threads = st.slider("Number of threads", 1, 8, 4)

    if st.button("🚀 Train Models", type="primary"):
        with st.spinner("Building dataset with selected features..."):
            dataset, interactions, user_features, item_features = \
                recommender.build_dataset_with_features(
                    interactions_df, clients_enhanced, plans_enhanced, feature_config
                )

            # Store in session state
            st.session_state.dataset = dataset
            st.session_state.interactions = interactions
            st.session_state.user_features = user_features
            st.session_state.item_features = item_features
            st.session_state.clients_enhanced = clients_enhanced
            st.session_state.plans_enhanced = plans_enhanced

        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, model_name in enumerate(selected_models):
            status_text.text(f"Training {model_name.upper()}...")
            recommender.train_model(
                model_name, interactions, user_features, item_features, num_threads
            )
            progress_bar.progress((i + 1) / len(selected_models))

        st.session_state.recommender = recommender
        st.session_state.trained_models = selected_models

        st.success(f"✅ Successfully trained {len(selected_models)} models!")

        # Show comparison
        st.subheader("📊 Model Comparison")
        comparison_df = compare_models(
            recommender, interactions, user_features, item_features
        )
        st.dataframe(comparison_df, use_container_width=True)

        # Visualization
        fig = px.bar(
            comparison_df,
            x='model',
            y=['precision@k', 'recall@k'],
            barmode='group',
            facet_col='k',
            title='Model Performance Comparison',
            labels={'value': 'Score', 'variable': 'Metric'}
        )
        st.plotly_chart(fig, use_container_width=True)


def recommendation_interface():
    """Interface for getting recommendations"""
    st.header("🎯 Get Recommendations")

    if 'recommender' not in st.session_state:
        st.warning("⚠️ Please train models first in the 'Model Training' tab")
        return

    recommender = st.session_state.recommender
    dataset = st.session_state.dataset
    user_features = st.session_state.user_features
    item_features = st.session_state.item_features
    clients_enhanced = st.session_state.clients_enhanced
    plans_enhanced = st.session_state.plans_enhanced

    # Recommendation method selection
    method = st.radio(
        "Recommendation Method",
        options=["Single Model", "Ensemble", "Cold Start"],
        horizontal=True
    )

    if method == "Single Model":
        col1, col2 = st.columns([1, 2])

        with col1:
            model_name = st.selectbox(
                "Select Model",
                options=st.session_state.trained_models
            )

            client_id = st.selectbox(
                "Select Client ID",
                options=clients_enhanced["client_id"].astype(str).tolist()
            )

            n_recommendations = st.slider("Number of recommendations", 1, 10, 5)

            if st.button("Get Recommendations"):
                model = recommender.models[model_name]
                recommendations = recommender.get_recommendations(
                    model, dataset, client_id, user_features, item_features, n_recommendations
                )

                st.session_state.current_recommendations = recommendations

        with col2:
            if 'current_recommendations' in st.session_state:
                st.subheader(f"Top {n_recommendations} Recommendations")

                for rec in st.session_state.current_recommendations:
                    plan_id = rec['plan_id']
                    plan_info = plans_enhanced[
                        plans_enhanced['plan_id'].astype(str) == plan_id
                    ].iloc[0]

                    with st.expander(
                        f"#{rec['rank']} - {plan_info['plan_type']} Plan (Score: {rec['score']:.4f})",
                        expanded=(rec['rank'] <= 3)
                    ):
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Price", f"${plan_info['price_tnd']:.2f}")
                        with col_b:
                            st.metric("Data", f"{plan_info['data_GB']} GB")
                        with col_c:
                            st.metric("Minutes", f"{plan_info['call_minutes']}")

                        st.write(f"**Price Tier:** {plan_info['price_tier']}")
                        st.write(f"**Data Tier:** {plan_info['data_tier']}")
                        st.write(f"**Bundle Type:** {plan_info['bundle_type']}")

    elif method == "Ensemble":
        st.subheader("Ensemble Recommendations")

        client_id = st.selectbox(
            "Select Client ID",
            options=clients_enhanced["client_id"].astype(str).tolist(),
            key="ensemble_client"
        )

        st.write("**Model Weights:**")
        model_weights = {}
        cols = st.columns(len(st.session_state.trained_models))
        for i, model_name in enumerate(st.session_state.trained_models):
            with cols[i]:
                weight = st.slider(
                    model_name.upper(),
                    0.0, 2.0, 1.0, 0.1,
                    key=f"weight_{model_name}"
                )
                model_weights[model_name] = weight

        n_recommendations = st.slider("Number of recommendations", 1, 10, 5, key="ensemble_n")

        if st.button("Get Ensemble Recommendations"):
            recommendations = recommender.ensemble_predictions(
                dataset, client_id, user_features, item_features,
                model_weights, n_recommendations
            )

            st.subheader(f"Top {n_recommendations} Ensemble Recommendations")
            for rec in recommendations:
                plan_id = rec['plan_id']
                plan_info = plans_enhanced[
                    plans_enhanced['plan_id'].astype(str) == plan_id
                ].iloc[0]

                with st.expander(
                    f"#{rec['rank']} - {plan_info['plan_type']} Plan (Score: {rec['score']:.4f})"
                ):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Price", f"${plan_info['price_tnd']:.2f}")
                    with col_b:
                        st.metric("Data", f"{plan_info['data_GB']} GB")
                    with col_c:
                        st.metric("Minutes", f"{plan_info['call_minutes']}")

    else:  # Cold Start
        st.subheader("Cold Start Recommendations")
        st.write("Get recommendations for new users based on their profile")

        col1, col2 = st.columns(2)

        with col1:
            segment = st.selectbox(
                "Customer Segment",
                options=clients_enhanced['segment'].unique()
            )

            data_intensity = st.selectbox(
                "Data Usage Pattern",
                options=['very_low', 'low', 'medium', 'high', 'very_high']
            )

        with col2:
            call_intensity = st.selectbox(
                "Call Usage Pattern",
                options=['very_low', 'low', 'medium', 'high', 'very_high']
            )

            usage_stability = st.selectbox(
                "Usage Stability",
                options=['stable', 'variable']
            )

        n_recommendations = st.slider("Number of recommendations", 1, 10, 5, key="cold_n")

        if st.button("Get Cold Start Recommendations"):
            user_profile = {
                'segment': segment,
                'data_intensity': data_intensity,
                'call_intensity': call_intensity,
                'usage_stability': usage_stability
            }

            recommendations = recommender.cold_start_recommendation(
                user_profile, plans_enhanced, dataset, item_features, n_recommendations
            )

            st.subheader(f"Top {n_recommendations} Cold Start Recommendations")
            for rec in recommendations:
                plan_id = rec['plan_id']
                plan_info = plans_enhanced[
                    plans_enhanced['plan_id'].astype(str) == plan_id
                ].iloc[0]

                with st.expander(
                    f"#{rec['rank']} - {plan_info['plan_type']} Plan (Score: {rec['score']:.4f})"
                ):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Price", f"${plan_info['price_tnd']:.2f}")
                    with col_b:
                        st.metric("Data", f"{plan_info['data_GB']} GB")
                    with col_c:
                        st.metric("Minutes", f"{plan_info['call_minutes']}")


def analytics_dashboard():
    """Advanced analytics and insights"""
    st.header("📈 Advanced Analytics Dashboard")

    if 'recommender' not in st.session_state:
        st.warning("⚠️ Please train models first")
        return

    clients_enhanced = st.session_state.clients_enhanced
    plans_enhanced = st.session_state.plans_enhanced

    # Customer Segmentation Analysis
    st.subheader("Customer Segmentation")

    col1, col2 = st.columns(2)

    with col1:
        # Lifecycle stage distribution
        lifecycle_dist = clients_enhanced['lifecycle_stage'].value_counts()
        fig1 = px.pie(
            values=lifecycle_dist.values,
            names=lifecycle_dist.index,
            title='Customer Lifecycle Distribution',
            color_discrete_sequence=px.colors.sequential.Purples
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        # Data intensity by segment
        intensity_by_segment = pd.crosstab(
            clients_enhanced['segment'],
            clients_enhanced['data_intensity']
        )
        fig2 = px.bar(
            intensity_by_segment,
            title='Data Intensity by Segment',
            barmode='stack'
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Plan Analysis
    st.subheader("Plan Portfolio Analysis")

    col3, col4 = st.columns(2)

    with col3:
        # Price vs Value scatter
        fig3 = px.scatter(
            plans_enhanced,
            x='price_tnd',
            y='value_score',
            size='data_GB',
            color='price_tier',
            hover_data=['plan_type', 'bundle_type'],
            title='Price vs Value Score',
            labels={'price_tnd': 'Price (TND)', 'value_score': 'Value Score'}
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        # Bundle composition
        bundle_dist = plans_enhanced['bundle_type'].value_counts()
        fig4 = px.bar(
            x=bundle_dist.index,
            y=bundle_dist.values,
            title='Plan Bundle Distribution',
            labels={'x': 'Bundle Type', 'y': 'Count'},
            color=bundle_dist.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Model Performance Comparison
    if 'interactions' in st.session_state:
        st.subheader("Model Performance Metrics")

        recommender = st.session_state.recommender
        interactions = st.session_state.interactions
        user_features = st.session_state.user_features
        item_features = st.session_state.item_features

        comparison_df = compare_models(
            recommender, interactions, user_features, item_features
        )

        # Heatmap of metrics
        pivot_precision = comparison_df.pivot(
            index='model', columns='k', values='precision@k'
        )

        fig5 = px.imshow(
            pivot_precision,
            title='Precision@K Heatmap',
            labels=dict(x='K', y='Model', color='Precision'),
            color_continuous_scale='RdYlGn',
            aspect='auto'
        )
        st.plotly_chart(fig5, use_container_width=True)


def main():
    # Header
    st.markdown('<h1 class="main-header">📱 Advanced Telco Recommender System</h1>',
                unsafe_allow_html=True)

    st.markdown("""
    ### State-of-the-Art LightFM Recommendation Engine

    **Advanced Techniques Implemented:**
    - 🎯 Multiple Loss Functions (WARP, BPR, WARP-KOS)
    - 🧬 Rich Feature Engineering (User & Item Features)
    - 🎭 Ensemble Methods for Robust Predictions
    - 🆕 Cold Start Handling for New Users
    - 📊 Comprehensive Performance Metrics
    - 🔬 A/B Testing Capabilities
    """)

    # Load data
    with st.spinner("Loading data and initializing system..."):
        recommender, clients_enhanced, plans_enhanced, subscriptions, usage, interactions_df = \
            load_and_prepare_data()

    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>{len(clients_enhanced):,}</h3><p>Total Customers</p></div>',
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>{len(plans_enhanced)}</h3><p>Available Plans</p></div>',
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f'<div class="metric-card"><h3>{len(subscriptions):,}</h3><p>Total Subscriptions</p></div>',
            unsafe_allow_html=True
        )

    with col4:
        segments = clients_enhanced['segment'].nunique()
        st.markdown(
            f'<div class="metric-card"><h3>{segments}</h3><p>Customer Segments</p></div>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Tabs for different functionalities
    tabs = st.tabs([
        "🔧 Model Training",
        "🎯 Recommendations",
        "📊 Analytics",
        "📚 Documentation"
    ])

    with tabs[0]:
        train_models_interface(recommender, interactions_df, clients_enhanced, plans_enhanced)

    with tabs[1]:
        recommendation_interface()

    with tabs[2]:
        analytics_dashboard()

    with tabs[3]:
        st.header("📚 Advanced Techniques Documentation")

        with st.expander("🎯 Loss Functions Explained"):
            st.markdown("""
            **WARP (Weighted Approximate-Rank Pairwise)**
            - Best for implicit feedback (clicks, views, purchases)
            - Optimizes ranking quality
            - Focuses on placing relevant items at the top
            - Recommended for: General recommendation scenarios

            **BPR (Bayesian Personalized Ranking)**
            - Assumes users prefer observed items over unobserved
            - Fast training
            - Good for large-scale datasets
            - Recommended for: High-volume recommendation systems

            **WARP-KOS (WARP with K-th Order Statistics)**
            - Variant of WARP focusing on top-K recommendations
            - Better for scenarios where only top results matter
            - Recommended for: Precision-critical applications

            **Hybrid Deep Model**
            - More latent factors (128 vs 64)
            - Deeper representations
            - Better for complex patterns
            - Recommended for: Rich feature scenarios
            """)

        with st.expander("🧬 Feature Engineering"):
            st.markdown("""
            **User Features:**
            - **Segment**: Customer category (residential, business, premium)
            - **Data Intensity**: Usage patterns (very_low to very_high)
            - **Call Intensity**: Voice usage patterns
            - **Usage Stability**: Predictability of consumption
            - **Lifecycle Stage**: Customer value tier (bronze to platinum)

            **Item Features:**
            - **Plan Type**: Category of service plan
            - **Price Tier**: Budget to luxury segmentation
            - **Data Tier**: Light to unlimited data capacity
            - **Value Category**: Price-to-benefit ratio
            - **Bundle Type**: Richness of service bundle

            **Interaction Features:**
            - Price flexibility scores
            - Upgrade/downgrade potential
            - Historical pattern matching
            """)

        with st.expander("🎭 Ensemble Methods"):
            st.markdown("""
            **Why Ensemble?**
            - Combines strengths of multiple models
            - More robust predictions
            - Reduces overfitting
            - Better generalization

            **Weighted Ensemble:**
            - Assigns different weights to models
            - Can prioritize specific model strengths
            - Configurable in the UI

            **Best Practices:**
            - Use WARP for general ranking
            - Add BPR for computational efficiency
            - Include deep model for complex patterns
            """)

        with st.expander("🆕 Cold Start Strategies"):
            st.markdown("""
            **Content-Based Filtering:**
            - Uses user profile features
            - Matches with item characteristics
            - No historical data needed

            **Hybrid Approach:**
            - Combines content and collaborative signals
            - Gradually transitions to collaborative as data accumulates

            **Feature Matching:**
            - High data users → unlimited plans
            - Price-sensitive → budget tiers
            - Business segment → premium features
            """)


if __name__ == "__main__":
    main()
