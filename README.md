# Churn Prediction: A Bricolage Approach to Constructing and Validating Churn Targets in E-Commerce

> **Author:** Tejaswi Kulkarni  
> **Dataset:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) (Kaggle)

---

## Overview

This project addresses a fundamental challenge in transactional e-commerce: **churn cannot be directly observed.** Unlike subscription services, there is no cancellation event — customers simply stop buying. The project constructs a principled, multi-signal churn target using the **Measurement as Bricolage** framework (Guerdan et al., 2025), combining behavioral inactivity with BERT-based sentiment analysis of customer reviews.

Two classifiers — **Logistic Regression** and **Random Forest** — are trained and evaluated using 5-fold stratified cross-validation on 95,095 customers from the Olist dataset (2016–2018).

---

## Key Results

| Model | Accuracy | Recall | Precision | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression | 76.13% | 58.82% | 31.53% | 0.7368 ± 0.0078 |
| Random Forest | 84.22% | 52.79% | 45.06% | **0.7744 ± 0.0045** |

**Top churn driver:** `delivery_delay_delta` (Gini importance: 0.578 — ~6× the second feature)

---

## The Churn Target Construction Problem

Standard churn definitions (e.g., "inactive for N days") risk circularity and fail to capture *why* a customer left. This project operationalizes churn as **both behavioral inactivity AND negative friction signals** from review text and star ratings.

### Churn Target Formula

```
target_churn = is_inactive AND fail_semantic
```

Where:
- **`is_inactive`** = customer has no orders in the 180 days following their last purchase
- **`fail_semantic`** = `consensus_friction > 0.5`

### Friction Signals

```python
# Star-rating friction (normalized to [0, 1])
stars_fric = (5 - review_score) / 4

# BERT-derived negativity probability
ugues_neg_prob  # from multilingual fine-tuned BERT classifier

# Combined friction score
consensus_friction = (ugues_neg_prob + stars_fric) / 2

# Dissonance between text and star signals
friction_dissonance = |ugues_neg_prob - stars_fric|
```

**Churn distribution:** 13,542 churned (14.09%) vs. 82,553 non-churned (85.91%)

---

## Bricolage Framework Evaluation

The target construction was evaluated against the five Guerdan et al. (2025) criteria:

| Criterion | Score |
|---|---|
| Validity | 4 / 5 |
| Simplicity | 5 / 5 |
| Predictability | 4 / 5 |
| Portability | 3 / 5 |
| Resource Efficiency | 5 / 5 |
| **Average** | **4.2 / 5** |

---

## Dataset

**Source:** [Olist Brazilian E-Commerce Public Dataset](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)  
**Period:** 2016–2018 | **Orders:** 99,440 | **Customers:** 96,095

The dataset consists of 9 relational CSV files:

| File | Description |
|---|---|
| `olist_orders_dataset.csv` | Order metadata, timestamps, status |
| `olist_order_items_dataset.csv` | Item-level details (price, freight, product) |
| `olist_order_payments_dataset.csv` | Payment type and value |
| `olist_order_reviews_dataset.csv` | Review scores and text (Portuguese) |
| `olist_customers_dataset.csv` | Customer IDs and location |
| `olist_products_dataset.csv` | Product dimensions and category |
| `olist_sellers_dataset.csv` | Seller location |
| `olist_geolocation_dataset.csv` | ZIP-to-lat/lng mapping |
| `product_category_name_translation.csv` | Portuguese → English category names |

---

## Methodology

### 1. Data Pipeline (5-Step Merge)

Orders → Items → Payments → Reviews → Customers → Products → Sellers

- All datetime fields converted to microsecond precision
- Payments and items aggregated per order using `aggregate_with_explicit_logic()` (explicit business logic, not generic groupby)
- Null imputation via business logic (e.g., missing review score → neutral 3.0, missing freight → median by seller)

### 2. BERT Sentiment Analysis

- **Model:** `ugues` (multilingual fine-tuned sentiment classifier via HuggingFace `transformers`)
- **Input:** Portuguese review text
- **Output:** `ugues_neg_prob` — probability that review expresses negative sentiment
- Applied via `final_bert_prep()` preprocessing function

### 3. Feature Engineering

Key engineered features:

| Feature | Description |
|---|---|
| `delivery_delay_delta` | Actual delivery − estimated delivery (days) |
| `carrier_delivery_delay` | Carrier ship date − order approval date |
| `product_volume_cm3` | length × height × width |
| `is_same_state` | Binary: customer and seller in same state |
| `consensus_friction` | Averaged BERT negativity + star friction |
| `friction_dissonance` | Absolute difference between BERT and star signals |
| `total_items`, `total_payment_value` | Order-level aggregations |

### 4. Preprocessing Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_features),
    ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_features)
])
```

### 5. Model Training

Both models use:
- `class_weight='balanced'` (handles 14/86 class imbalance)
- `StratifiedKFold(n_splits=5)` cross-validation
- ROC-AUC as primary evaluation metric

---

## Feature Importance (Random Forest)

| Rank | Feature | Gini Importance |
|---|---|---|
| 1 | `delivery_delay_delta` | 0.578 |
| 2 | `consensus_friction` | ~0.09 |
| 3 | `freight_value` | ~0.07 |
| 4 | `price` | ~0.06 |
| 5 | `product_volume_cm3` | ~0.05 |

Delivery performance dominates churn prediction — delays are ~6× more predictive than the next feature.

---

## Tech Stack

```
Python 3.x
├── pandas, numpy          # Data manipulation
├── scikit-learn           # ML pipelines, models, evaluation
├── transformers           # HuggingFace BERT (ugues model)
├── matplotlib, seaborn    # Visualization
└── torch                  # BERT inference backend
```

---

## Project Structure

```
├── data/
│   ├── olist_orders_dataset.csv
│   ├── olist_order_items_dataset.csv
│   ├── olist_order_payments_dataset.csv
│   ├── olist_order_reviews_dataset.csv
│   ├── olist_customers_dataset.csv
│   ├── olist_products_dataset.csv
│   ├── olist_sellers_dataset.csv
│   ├── olist_geolocation_dataset.csv
│   └── product_category_name_translation.csv
├── notebooks/
│   └── churn_prediction.ipynb    # Full analysis pipeline
├── README.md
└── requirements.txt
```

---

## Usage

### 1. Install dependencies

```bash
pip install pandas numpy scikit-learn matplotlib seaborn transformers torch
```

### 2. Download the dataset

Download the Olist dataset from [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce) and place all CSV files in the `data/` directory.

### 3. Run the pipeline

Open and run `notebooks/churn_prediction.ipynb` sequentially:

1. **Data loading & merge** — builds the unified order-customer table
2. **Feature engineering** — computes derived features including delivery delays and volume
3. **BERT inference** — scores review text sentiment (requires GPU recommended)
4. **Churn target construction** — applies inactivity + friction thresholds
5. **Model training & evaluation** — trains LR and RF with cross-validation

> **Note:** BERT inference on ~100K reviews is computationally intensive. A GPU or pre-computed `ugues_neg_prob` values are recommended for faster runs.

---

## Limitations

- **Single-cohort dataset:** Olist covers 2016–2018 Brazilian e-commerce; portability to other markets is limited (Bricolage score: 3/5)
- **Threshold sensitivity:** The 180-day inactivity window and `consensus_friction > 0.5` cutoff are theoretically motivated but not tuned empirically
- **BERT language dependency:** The `ugues` model was fine-tuned on Portuguese; review texts in other languages would require a different model
- **Recall–precision tradeoff:** The Random Forest achieves better precision than LR (45% vs. 32%) but at the cost of lower recall (53% vs. 59%)

---

## References

- Guerdan, L., et al. (2025). *Measurement as Bricolage: A Framework for Constructing and Validating Proxy Measurements in Applied ML*. FAccT.
- Olist. (2018). *Brazilian E-Commerce Public Dataset by Olist*. Kaggle. https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
- Devlin, J., et al. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL.
- `ugues` HuggingFace model: Multilingual sentiment classifier fine-tuned for negativity detection.
