# 🚀 Unified App Guide: LightFM + NCF

## Overview

The **Unified App** combines all recommendation models in one interactive interface:
- ✅ **LightFM Models**: WARP, BPR, WARP-KOS, Hybrid Deep
- ✅ **NCF**: Neural Collaborative Filtering (Deep Learning)
- ✅ **Comparison**: Side-by-side performance metrics
- ✅ **Recommendations**: Test all models with your data

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# Build image
docker build -t telco-reco .

# Run unified app (includes all models)
docker run -p 8501:8501 telco-reco
```

Then open: http://localhost:8501

### Option 2: Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Run unified app
streamlit run unified_app.py
```

---

## 📖 User Guide

### Page 1: 🏠 Overview

**What you see:**
- Quick introduction to LightFM and NCF
- Dataset statistics
- Performance comparison table

**What to do:**
- Read the overview to understand the models
- Check your dataset size

---

### Page 2: 🎯 Model Training & Comparison

This page has 3 tabs:

#### Tab 1: LightFM Training

**Steps:**
1. Select models to train (e.g., WARP + BPR)
2. Adjust hyperparameters:
   - **Epochs**: 10-50 (default: 30)
   - **Components**: 32, 64, or 128 (default: 64)
3. Click **"🚀 Train LightFM Models"**
4. Wait 3-5 minutes
5. Review results table and chart

**Expected Output:**
```
Model    Precision@5   Recall@5   AUC
WARP     0.34          0.18       0.87
BPR      0.31          0.16       0.84
```

#### Tab 2: NCF Training

**Steps:**
1. Configure NCF:
   - **Epochs**: 10-50 (default: 20)
   - **Embedding Size**: 32, 64, or 128 (default: 64)
   - **Architecture**: Deep or Standard
   - **Negative Sampling**: 2-6 (default: 4)
2. Click **"🚀 Train NCF Model"**
3. Wait 2-3 minutes
4. Review performance metrics

**Expected Output:**
```
Precision@5: 0.43
Recall@5: 0.24
AUC: 0.92
```

#### Tab 3: Model Comparison

**What you see:**
- Combined results table (highlights best scores)
- Side-by-side bar charts
- Winner announcement (best model)

**Example:**
```
🏆 Best Precision@5: NCF (Deep Learning) (0.4300)
🏆 Best AUC: NCF (Deep Learning) (0.9200)
```

---

### Page 3: 📊 Recommendations

**Steps:**
1. Train at least one model (LightFM or NCF)
2. Select a model from dropdown
3. Choose a client ID
4. Set number of recommendations (3-10)
5. Click **"Get Recommendations"**

**Output:**
```
#1 - Postpaid Plan (Score: 0.8542)
    Price: 45.50 TND
    Data: 100 GB
    Minutes: 1000
    Price Tier: premium
    Data Tier: heavy
    Bundle: complete
```

**Try different models to compare:**
- LightFM - WARP
- LightFM - BPR
- NCF - Deep Learning

---

### Page 4: 📈 Analytics

**Features:**
1. **Performance Heatmap**: Visual comparison of all models
2. **Customer Insights**:
   - Lifecycle distribution (pie chart)
   - Data intensity by segment (stacked bar)

---

## 🎯 Recommended Workflow

### First Time Setup (10 minutes)

```
1. Navigate to "Model Training & Comparison"
2. Train LightFM models:
   - Select: WARP, BPR
   - Keep default settings
   - Click Train (wait 3-5 min)
3. Train NCF model:
   - Keep default settings
   - Click Train (wait 2-3 min)
4. Go to "Model Comparison" tab
   - Review which model performs best
```

### Getting Recommendations (2 minutes)

```
1. Navigate to "Recommendations"
2. Select best model (from comparison)
3. Choose a client
4. Get top 5 recommendations
5. Review plan details
```

### Comparing Models (5 minutes)

```
1. Get recommendations from LightFM WARP
2. Note the top 3 plans
3. Get recommendations from NCF
4. Compare the top 3 plans
5. Decide which model gives better results
```

---

## 📊 Performance Expectations

### LightFM Models

| Model | Precision@5 | Training Time | Best For |
|-------|-------------|---------------|----------|
| WARP | 0.34 | 45s | General use |
| BPR | 0.31 | 25s | Fast training |
| WARP-KOS | 0.36 | 50s | Top-K focus |
| Hybrid Deep | 0.37 | 120s | Complex patterns |

### NCF Model

| Config | Precision@5 | Training Time | Best For |
|--------|-------------|---------------|----------|
| Standard | 0.42 | 120s | Balanced |
| Deep | 0.43-0.45 | 180s | Best accuracy |

---

## ⚙️ Configuration Tips

### For Best Accuracy
```
LightFM:
- Models: All 4 (WARP, BPR, WARP-KOS, Hybrid Deep)
- Epochs: 50
- Components: 128

NCF:
- Epochs: 30
- Embedding: 128
- Architecture: Deep
- Negative Sampling: 6
```

### For Fast Training
```
LightFM:
- Models: BPR only
- Epochs: 20
- Components: 64

NCF:
- Epochs: 10
- Embedding: 32
- Architecture: Standard
- Negative Sampling: 2
```

### For Production
```
LightFM:
- Models: WARP + BPR
- Epochs: 30
- Components: 64

NCF:
- Epochs: 20
- Embedding: 64
- Architecture: Deep
- Negative Sampling: 4
```

---

## 🐛 Troubleshooting

### Issue: "Train models first" warning

**Solution:**
- Go to "Model Training & Comparison" page
- Train at least one LightFM model OR NCF model
- Then return to Recommendations/Analytics

### Issue: Training takes too long

**Solution:**
- Reduce epochs (try 10-20)
- Use fewer LightFM models (just WARP + BPR)
- For NCF, use Standard instead of Deep

### Issue: Out of memory

**Solution:**
- Reduce embedding size to 32
- Reduce negative sampling ratio to 2
- Train one model at a time

### Issue: Models not in dropdown

**Solution:**
- Models are stored in session state
- If you refresh the page, you need to retrain
- Check browser console for errors

---

## 📈 Interpreting Results

### Precision@5
- **What it means**: Of 5 recommendations, how many are relevant?
- **Good score**: > 0.30
- **Excellent score**: > 0.40
- **Your goal**: Maximize this

### Recall@5
- **What it means**: Of all relevant items, how many are in top 5?
- **Good score**: > 0.15
- **Excellent score**: > 0.20
- **Note**: Always lower than Precision

### AUC (Area Under Curve)
- **What it means**: Overall ranking quality
- **Good score**: > 0.85
- **Excellent score**: > 0.90
- **Range**: 0.5 (random) to 1.0 (perfect)

---

## 🎓 Decision Guide

### Which Model Should I Use?

```
Do you need the BEST accuracy?
├─ YES → Use NCF (Deep)
│       Expected: Precision@5 ≈ 0.43-0.45
│       Cost: 3 minutes training
└─ NO → Continue

Do you need FAST training?
├─ YES → Use LightFM BPR
│       Expected: Precision@5 ≈ 0.31
│       Cost: 25 seconds training
└─ NO → Continue

Do you want BALANCED performance?
├─ YES → Use LightFM WARP or NCF Standard
│       WARP: Precision@5 ≈ 0.34 (45s)
│       NCF: Precision@5 ≈ 0.42 (120s)
└─ NO → Continue

Do you want BEST of both worlds?
└─ YES → Train multiple models
        Use NCF for high-value customers
        Use LightFM for scale
```

---

## 🔄 Integration with Your System

### API Endpoint (Example)

```python
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load trained models
ncf_model = pickle.load(open('ncf_model.pkl', 'rb'))
lightfm_model = pickle.load(open('lightfm_model.pkl', 'rb'))

@app.route('/recommend', methods=['POST'])
def recommend():
    client_id = request.json['client_id']
    model_type = request.json.get('model', 'ncf')  # default to NCF
    n = request.json.get('n', 5)

    if model_type == 'ncf':
        recs = ncf_model.recommend(client_id, n=n)
    else:
        recs = lightfm_model.get_recommendations(client_id, n=n)

    return jsonify({'recommendations': recs})
```

### Batch Processing

```python
# Generate recommendations for all users
all_users = clients['client_id'].tolist()
batch_recommendations = {}

for user_id in all_users:
    # Use best model (NCF in this case)
    recs = ncf_model.recommend(user_id, n=5)
    batch_recommendations[user_id] = recs

# Save to database or CSV
pd.DataFrame(batch_recommendations).to_csv('all_recommendations.csv')
```

---

## 🚀 Next Steps

### After Initial Training

1. ✅ Compare all models
2. ✅ Pick the best performer
3. ✅ Test with real users
4. ✅ Collect feedback
5. ✅ Retrain weekly

### For Production Deployment

1. ✅ Train with full dataset
2. ✅ Save best model
3. ✅ Set up API endpoint
4. ✅ Implement A/B testing
5. ✅ Monitor performance

### For Optimization

1. ✅ Hyperparameter tuning
2. ✅ Feature engineering
3. ✅ Ensemble methods
4. ✅ Regular retraining

---

## 📞 Support

### Questions?

- Check documentation: [ADVANCED_TECHNIQUES.md](ADVANCED_TECHNIQUES.md)
- NCF guide: [NCF_QUICKSTART.md](NCF_QUICKSTART.md)
- Algorithm comparison: [NEXT_GENERATION_ALGORITHMS.md](NEXT_GENERATION_ALGORITHMS.md)

### Common Questions

**Q: Which is better, LightFM or NCF?**
A: NCF typically gives 10-15% better accuracy but takes longer to train. For production, test both and use what works best for your data.

**Q: Can I use multiple models together?**
A: Yes! Train all models and use different ones for different scenarios (e.g., NCF for premium customers, LightFM for scale).

**Q: How often should I retrain?**
A: Weekly for production systems. Monthly for stable datasets.

**Q: Do I need a GPU?**
A: No, but it helps. NCF trains 2-3x faster with GPU. LightFM doesn't benefit much from GPU.

---

**Ready to compare all models?**

Run: `docker run -p 8501:8501 telco-reco`

Then open: http://localhost:8501

🚀 **Let the best model win!**
