# WSMTE — Kaggle Instructions

## Overview
Three notebooks run on Kaggle T4 2x GPU.
All other notebooks run on local PC.

---

## Which Notebooks Run Where

| Notebook | Runs On | GPU Needed |
|----------|---------|------------|
| 01_data_prep.ipynb | Local PC | No |
| 02_feature_engineering.ipynb | Local PC | No |
| 03_finbert_inference.ipynb | Kaggle T4 2x | Yes |
| 04_model_training.ipynb | Kaggle T4 2x | Yes |
| 05_ablation.ipynb | Kaggle T4 2x | Yes |
| 06_evaluation.ipynb | Local PC | No |

---

## One-Time Kaggle Setup

### Enable GPU
- Open notebook in Kaggle
- Settings (right panel) → Accelerator → GPU T4 x2
- Session type → Persistent

### Upload processed data as Kaggle Dataset (after Day 2)
Upload from local data/processed/:
- final_dataset.csv
- X_train.npy, X_val.npy, X_test.npy
- y_clf_train.npy, y_clf_val.npy, y_clf_test.npy
- y_reg_train.npy, y_reg_val.npy, y_reg_test.npy

Name: wsmte-processed-data
Path in Kaggle: /kaggle/input/wsmte-processed-data/

---

## Notebook 03 — FinBERT + mDeBERTa Inference

This notebook processes THREE datasets in order.
Save output CSV after each dataset — do NOT wait for all to finish.

### Setup cell (always run first)
```python
!git clone https://github.com/YOUR_USERNAME/WSMTE.git
import sys
sys.path.append('/kaggle/working/WSMTE')

from config.config import CONFIG
print(f"FinBERT model: {CONFIG['finbert_model']}")
print(f"mDeBERTa model: {CONFIG['mdeberta_model']}")
```

### Upload input data to Kaggle
Add these as Kaggle input datasets:
- kotekar_news.csv → /kaggle/input/wsmte-raw/kotekar_news.csv
- kaggle_news_1.csv → /kaggle/input/wsmte-raw/kaggle_news_1.csv
- kaggle_news_2.csv → /kaggle/input/wsmte-raw/kaggle_news_2.csv

### Part 1 — Kotekar FinBERT (polarity_company)
```python
from src.sentiment.finbert_inference import get_finbert_polarity
from src.sentiment.finbert_inference import prepare_kotekar_text
import pandas as pd

kotekar = pd.read_csv('/kaggle/input/wsmte-raw/kotekar_news.csv')
kotekar['date'] = pd.to_datetime(kotekar['datePublished']).dt.date
kotekar = kotekar[
    (kotekar['date'] >= pd.to_datetime('2020-01-01').date()) &
    (kotekar['date'] <= pd.to_datetime('2024-04-23').date())
]

print(f"Kotekar shape: {kotekar.shape}")
print("Running FinBERT on Kotekar (headline + first 2 sentences)...")
kotekar['text_for_finbert'] = kotekar.apply(prepare_kotekar_text, axis=1)
kotekar['polarity_company'] = get_finbert_polarity(
    kotekar['text_for_finbert'].tolist(), batch_size=32)

kotekar[['date','company','symbol','polarity_company']]\
    .to_csv('/kaggle/working/kotekar_polarity_temp.csv', index=False)
print("Kotekar FinBERT done. Saved kotekar_polarity_temp.csv")
```

### Part 2 — Kotekar mDeBERTa (subjectivity)
```python
from src.sentiment.finbert_inference import get_subjectivity

print("Running mDeBERTa on Kotekar articleBody (full text, 512 token limit)...")
kotekar['subjectivity'] = get_subjectivity(
    kotekar['articleBody'].fillna('').tolist(), batch_size=16)

# Save combined Kotekar sentiment
kotekar[['date','company','symbol','polarity_company','subjectivity']]\
    .to_csv('/kaggle/working/kotekar_sentiment.csv', index=False)
print(f"Kotekar mDeBERTa done. Saved kotekar_sentiment.csv")
print(f"Polarity range: {kotekar['polarity_company'].min():.3f} "
      f"to {kotekar['polarity_company'].max():.3f}")
print(f"Subjectivity range: {kotekar['subjectivity'].min():.3f} "
      f"to {kotekar['subjectivity'].max():.3f}")
```

### Part 3 — Kaggle Dataset 1 FinBERT (Title column)
```python
kaggle1 = pd.read_csv('/kaggle/input/wsmte-raw/kaggle_news_1.csv')
kaggle1['date'] = pd.to_datetime(kaggle1['Date']).dt.date
kaggle1 = kaggle1[
    (kaggle1['date'] >= pd.to_datetime('2020-01-01').date()) &
    (kaggle1['date'] <= pd.to_datetime('2021-04-15').date())
]

print(f"Kaggle1 shape: {kaggle1.shape}")
print("Running FinBERT on Kaggle1 Title column...")
kaggle1['polarity_market'] = get_finbert_polarity(
    kaggle1['Title'].fillna('').tolist(), batch_size=32)

kaggle1[['date','polarity_market']]\
    .to_csv('/kaggle/working/kaggle1_polarity.csv', index=False)
print("Kaggle1 done. Saved kaggle1_polarity.csv")
```

### Part 4 — Kaggle Dataset 2 FinBERT (Headline column)
```python
kaggle2 = pd.read_csv('/kaggle/input/wsmte-raw/kaggle_news_2.csv')
kaggle2['date'] = pd.to_datetime(kaggle2['Date']).dt.date
kaggle2 = kaggle2[
    (kaggle2['date'] >= pd.to_datetime('2022-01-01').date()) &
    (kaggle2['date'] <= pd.to_datetime('2024-04-23').date())
]

print(f"Kaggle2 shape: {kaggle2.shape}")
print("Running FinBERT on Kaggle2 Headline column...")
kaggle2['polarity_market'] = get_finbert_polarity(
    kaggle2['Headline'].fillna('').tolist(), batch_size=32)

kaggle2[['date','polarity_market']]\
    .to_csv('/kaggle/working/kaggle2_polarity.csv', index=False)
print("Kaggle2 done. Saved kaggle2_polarity.csv")
```

### Download outputs (4 files total)
- /kaggle/working/kotekar_sentiment.csv → local data/finbert_outputs/
- /kaggle/working/kaggle1_polarity.csv → local data/finbert_outputs/
- /kaggle/working/kaggle2_polarity.csv → local data/finbert_outputs/

### Runtime estimates (Kaggle T4 2x)
```
Kotekar FinBERT  (~2,719 articles): 15–20 min
Kotekar mDeBERTa (~2,719 articles): 30–45 min
Kaggle1 FinBERT  (~varies):         10–15 min
Kaggle2 FinBERT  (~varies):         15–20 min
─────────────────────────────────────────────
Total:                               70–100 min
```

---

## Notebook 04 — Model Training (Configs A–G)

### Setup cell
```python
!git clone https://github.com/YOUR_USERNAME/WSMTE.git
import sys
sys.path.append('/kaggle/working/WSMTE')

import numpy as np
from config.config import CONFIG
from src.training.trainer import train_multi_run

X_train = np.load('/kaggle/input/wsmte-processed-data/X_train.npy')
X_val   = np.load('/kaggle/input/wsmte-processed-data/X_val.npy')
X_test  = np.load('/kaggle/input/wsmte-processed-data/X_test.npy')
y_clf_train = np.load('/kaggle/input/wsmte-processed-data/y_clf_train.npy')
y_clf_val   = np.load('/kaggle/input/wsmte-processed-data/y_clf_val.npy')
y_clf_test  = np.load('/kaggle/input/wsmte-processed-data/y_clf_test.npy')
y_reg_train = np.load('/kaggle/input/wsmte-processed-data/y_reg_train.npy')
y_reg_val   = np.load('/kaggle/input/wsmte-processed-data/y_reg_val.npy')
y_reg_test  = np.load('/kaggle/input/wsmte-processed-data/y_reg_test.npy')

import json
with open('/kaggle/input/wsmte-processed-data/class_weights.json') as f:
    class_weight_dict = json.load(f)

data = {
    'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
    'y_clf_train': y_clf_train, 'y_clf_val': y_clf_val,
    'y_clf_test': y_clf_test,
    'y_reg_train': y_reg_train, 'y_reg_val': y_reg_val,
    'y_reg_test': y_reg_test,
    'class_weight': class_weight_dict   # NOTE: no 's' — must match trainer.py data.get('class_weight')
}
print(f"Data loaded. X_train: {X_train.shape}")
```

### Training loop
```python
import pandas as pd

results = pd.DataFrame(columns=CONFIG['results_columns'])

for config_name in ['A', 'B', 'C', 'D', 'E', 'F', 'G']:
    print(f"\n{'='*50}\nRunning Config {config_name}\n{'='*50}")
    ablation_cfg = CONFIG['ablation_configs'][config_name]
    config_results = train_multi_run(
        config=CONFIG,
        ablation_cfg=ablation_cfg,
        config_name=config_name,
        data=data,
        n_runs=ablation_cfg['n_runs']
    )
    results = pd.concat([results, config_results], ignore_index=True)
    # Save after every config — protects against session timeout
    results.to_csv('/kaggle/working/ablation_results_partial.csv', index=False)
    print(f"Config {config_name} done.")

results.to_csv('/kaggle/working/ablation_results_AG.csv', index=False)
print("All configs A-G complete.")
```

### Download outputs
- /kaggle/working/ablation_results_AG.csv → local ablation/
- /kaggle/working/config_g_best.h5 → local results/saved_models/

### Expected runtime: 2–4 hours

---

## Notebook 05 — Config H PSO + SHAP

### Setup cell
```python
!git clone https://github.com/YOUR_USERNAME/WSMTE.git
import sys
sys.path.append('/kaggle/working/WSMTE')

import numpy as np
import tensorflow as tf
from config.config import CONFIG
from src.models.wsmte import build_wsmte
from src.models.pso_weighting import run_pso_stage, finetune_with_pso_weights
from src.training.trainer import train_multi_run

# Load data (same as notebook 04)
X_val = np.load('/kaggle/input/wsmte-processed-data/X_val.npy')
y_clf_val = np.load('/kaggle/input/wsmte-processed-data/y_clf_val.npy')
# ... load all arrays

# Load Config G weights as starting point
model = build_wsmte(CONFIG, use_pso=False)
model.load_weights('/kaggle/input/wsmte-models/config_g_best.h5')
print("Config G weights loaded")
```

### PSO + fine-tune
```python
# Stage 2: PSO search
print("Starting PSO weight search (50 iterations, 20 particles)...")
best_weights, best_cost = run_pso_stage(model, X_val, y_clf_val, CONFIG)  # returns (weights, cost)
print(f"PSO weights: LSTM={best_weights[0]:.3f}, "
      f"TCN={best_weights[1]:.3f}, GRU={best_weights[2]:.3f}")

# Stage 3: 30-seed fine-tuning
print("Starting Config H — 30 runs...")
config_h_results = train_multi_run(
    config=CONFIG,
    ablation_cfg=CONFIG['ablation_configs']['H'],
    config_name='H',
    data=data,
    n_runs=30,
    pso_weights=best_weights
)
config_h_results.to_csv('/kaggle/working/ablation_results_H.csv', index=False)
print("Config H complete.")
```

### SHAP analysis
```python
from src.evaluation.shap_analysis import run_shap_analysis
import tensorflow as tf

# Load best Config H model (saved as full .keras — includes PSO architecture)
# run_ablation.py saves config_h_best.keras via pso_model.save()
best_model = tf.keras.models.load_model('/kaggle/working/config_h_best.keras')

run_shap_analysis(
    model=best_model,
    X_test=X_test,
    feature_names=CONFIG['feature_columns'],
    save_path='/kaggle/working/shap_summary.png'
)
print("SHAP done.")
```

### Download outputs
- /kaggle/working/ablation_results_H.csv → local ablation/
- /kaggle/working/config_h_best.h5 → local results/saved_models/
- /kaggle/working/shap_summary.png → local results/figures/

### Expected runtime: 1–2 hours

---

## Important Reminders

### Before every Kaggle session
- Push latest code to GitHub first
- Re-clone in Kaggle every session (never use stale cached version)
- Verify GPU: Settings → Accelerator → GPU T4 x2

### If Kaggle session dies
- trainer.py saves results after EVERY config run to partial CSV
- Check ablation_results_partial.csv to see what completed
- Re-run only incomplete configs

### After every Kaggle session
- Download ALL output files immediately (outputs expire after session)
- Push downloaded files to GitHub
- Update TASKS.md checkboxes
