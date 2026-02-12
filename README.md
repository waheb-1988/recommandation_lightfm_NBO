# 📦 Advanced LightFM Recommendation System for Telecom Operators

This project implements a **state-of-the-art hybrid recommendation system** using [LightFM](https://making.lyst.com/lightfm/docs/home.html), designed for **telecom operators** to automatically suggest the most suitable mobile plans to clients — based on their usage patterns and similarities with other users.

---

## 🌟 New: Advanced Features

### 🚀 Two Applications Available:

1. **Basic App** (`streamlit_app.py`) - Simple recommendations
2. **Advanced App** (`advanced_streamlit_app.py`) - Full-featured with:
   - 🎯 Multiple Loss Functions (WARP, BPR, WARP-KOS, Hybrid Deep)
   - 🧬 Rich Feature Engineering (15+ user & item features)
   - 🎭 Ensemble Methods for robust predictions
   - 🆕 Advanced Cold Start handling
   - 📊 Interactive Model Comparison
   - 📈 Real-time Analytics Dashboard
   - ⚡ A/B Testing Capabilities

**👉 See [ADVANCED_TECHNIQUES.md](ADVANCED_TECHNIQUES.md) for detailed documentation**
**👉 See [QUICKSTART_ADVANCED.md](QUICKSTART_ADVANCED.md) for quick start guide**

---

## 🚀 Features

### Basic Features
* ✅ Hybrid recommendation system (collaborative + content-based)
* 📊 Personalized plan suggestions for each client
* 🧠 Handles **cold-start** for new clients via JSON input
* 💿 Fully **Dockerized setup** — no local Python install needed
* 🧩 Outputs both CSV and JSON recommendation files

### Advanced Features ⭐ NEW
* 🎯 **4 Loss Functions**: WARP, BPR, WARP-KOS, Hybrid Deep
* 🧬 **Advanced Feature Engineering**: Data intensity, usage stability, lifecycle stages
* 🎭 **Ensemble Methods**: Weighted model combination
* 📊 **Model Comparison**: Side-by-side performance metrics
* 🆕 **Enhanced Cold Start**: Content-based with feature matching
* 📈 **Interactive Dashboard**: Real-time analytics and visualizations
* ⚡ **Hyperparameter Tuning**: Configurable model parameters

### Deep Learning Features 🧠 NEWEST
* 🔬 **Neural Collaborative Filtering (NCF)**: Deep learning-based recommendations
* 📊 **10-15% Performance Improvement** over LightFM baseline
* 🎯 **Dual Architecture**: GMF + MLP paths for better accuracy
* ⚡ **Feature Integration**: Incorporates user/item side features
* 📈 **Better Cold Start**: Leverages features for new users
* 🔄 **Easy Comparison**: Side-by-side with LightFM

**👉 See [NCF_QUICKSTART.md](NCF_QUICKSTART.md) for NCF guide**
**👉 See [NEXT_GENERATION_ALGORITHMS.md](NEXT_GENERATION_ALGORITHMS.md) for algorithm comparison**

---

## 🧮 Project Structure

```
📁 recommandation_lightfm/
 ┣ 📜 Dockerfile
 ┣ 📜 requirements.txt
 ┣ 📜 lightfm_reco.py
 ┣ 📜 clients.csv
 ┣ 📜 plans.csv
 ┣ 📜 subscriptions.csv
 ┣ 📜 usage.csv
 ┣ 📜 new_clients.json
 ┗ 📜 README.md
```

---

## ⚙️ How to Run This Project (with Docker)

### 1️⃣ Clone or Download the Repository

```bash
git clone https://github.com/ahmedbahaeddineabid/recommandation_lightfm.git
cd recommandation_lightfm
```

—or download the ZIP and open a terminal (PowerShell, CMD, or VS Code) inside the project folder.

---

### 2️⃣ Build the Docker Image

```bash
docker build -t recommandation_lightfm .
```

---

### 3️⃣ Run the Container

**Option A: Basic Batch Processing**
```bash
docker run --rm -v ${PWD}:/app recommandation_lightfm
```
💡 This runs the basic recommendation script and generates output files.

**Option B: Interactive Streamlit App (Basic)**
```bash
docker run -p 8501:8501 recommandation_lightfm streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```
💡 Access at http://localhost:8501

**Option C: Advanced Streamlit App ⭐ RECOMMENDED**
```bash
docker run -p 8501:8501 recommandation_lightfm streamlit run advanced_streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```
💡 Full-featured interface with model comparison, ensemble methods, and analytics!

---

## 🎯 Quick Start (Advanced App)

### Local Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run advanced app
streamlit run advanced_streamlit_app.py
```

### Using the Interface

1. **Model Training Tab**
   - Select features to use
   - Choose models (WARP, BPR, WARP-KOS, Hybrid Deep)
   - Train and compare performance

2. **Recommendations Tab**
   - Single Model: Get recommendations from one model
   - Ensemble: Combine multiple models with custom weights
   - Cold Start: Handle new users with no history

3. **Analytics Tab**
   - Customer segmentation insights
   - Plan portfolio analysis
   - Model performance comparison

**📚 See [QUICKSTART_ADVANCED.md](QUICKSTART_ADVANCED.md) for detailed guide**

---

## 🗂️ Output Files

After running, two new files will be generated automatically:

* `recommendations.csv` → top plan recommendations for **existing clients**
* `cold_start_recommendations.json` → top plan recommendations for **new clients** (from `new_clients.json`)

---

## 📊 Dataset Overview

| File                  | Description                                              |
| --------------------- | -------------------------------------------------------- |
| **clients.csv**       | Basic client data including segments and demographics    |
| **plans.csv**         | Available mobile plans (IDs, names, prices, types, etc.) |
| **subscriptions.csv** | Client subscriptions linking clients to plans            |
| **usage.csv**         | Aggregated client usage data (data, calls, SMS)          |
| **new_clients.json**  | Input file for cold-start predictions                    |

---

## 🧩 Requirements

* [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running
* No need for local Python or libraries — **everything runs inside Docker**

---

## 👨‍💻 Author

**Ahmed Baha Eddine Abid**
📧 [ahmed.baha.eddine.abid@gmail.com](mailto:ahmed.baha.eddine.abid@gmail.com)
🧠 Data Science & BI | Machine Learning | Telecom Analytics
