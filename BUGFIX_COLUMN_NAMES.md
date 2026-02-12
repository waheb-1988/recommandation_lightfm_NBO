# Bug Fix: Column Name Mismatches

## Issue
The advanced LightFM application was failing with `KeyError: 'price'` because the code was expecting column names that didn't match the actual CSV files.

## Root Cause
The plans.csv file uses `price_tnd` (Tunisian Dinar) instead of `price`, and `sms_count` instead of `sms_included`.

## Files Fixed

### 1. advanced_lightfm_models.py
**Changes:**
- Line 139: `plans_enhanced['price']` → `plans_enhanced['price_tnd']`
- Line 155: `plans_enhanced['price']` → `plans_enhanced['price_tnd']`
- Line 171: `plans_enhanced['sms_included']` → `plans_enhanced['sms_count']`
- Line 193: `interactions_df['price']` → `interactions_df['price_tnd']`

### 2. advanced_streamlit_app.py
**Changes:**
- Line 232, 283, 343: `plan_info['price']` → `plan_info['price_tnd']` (all occurrences)
- Line 399: scatter plot x-axis changed from `'price'` to `'price_tnd'`
- Line 405: label updated from `'price': 'Price ($)'` to `'price_tnd': 'Price (TND)'`

## Verified Column Names

### plans.csv
✅ Correct columns:
- `plan_id`, `plan_name`, `plan_type`, `price_tnd`, `data_GB`, `call_minutes`, `sms_count`, `valid_from`, `valid_to`, `recommended_segment`

### usage.csv
✅ Correct columns:
- `usage_id`, `client_id`, `month`, `year`, `data_used_GB`, `call_minutes`, `sms_sent`, `network_type`, `device_type`

### clients.csv
✅ Correct columns:
- `client_id`, `age`, `gender`, `region`, `income_level`, `segment`, `device_type`, `tech_usage`

### subscriptions.csv
✅ Correct columns:
- `sub_id`, `client_id`, `plan_id`, `start_date`, `end_date`, `satisfaction_score`, `churned`, `complaint_count`, `switch_reason`

## Testing
To verify the fix works:

```bash
# Option 1: Local
streamlit run advanced_streamlit_app.py

# Option 2: Docker
docker build -t lightfm-reco .
docker run -p 8501:8501 lightfm-reco streamlit run advanced_streamlit_app.py --server.port=8501 --server.address=0.0.0.0
```

## Status
✅ All column name mismatches fixed
✅ Code now matches actual CSV structure
✅ Ready to run
