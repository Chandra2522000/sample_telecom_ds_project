# Customer Churn Prediction: an end-to-end data science project

**Goal:** predict which telecom customers will cancel, so the retention team contacts the right people instead of everyone.

> The data is **synthetic** (generated in `churn_project.py` with realistic patterns plus deliberate mess: missing values, duplicates, inconsistent text). To use real data, replace `generate_data()` with `pd.read_csv(...)` of e.g. the public IBM/Kaggle "Telco Customer Churn" file. Everything else stays the same.

## Run it

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib scipy
python churn_project.py     # runs the whole pipeline (~1 min), writes data/, figures/, models/
python predict.py           # scores 3 demo customers with the saved model
```

## Project structure

```
churn_project.py   the full pipeline, one function per step
predict.py         load the saved model and score new customers (the "deployment" step)
data/              raw CSV, cleaned CSV, test-set predictions
figures/           EDA, correlation, model evaluation, feature importance
models/            churn_model.joblib + metrics.json
```

## The workflow (what each step teaches)

| # | Step | What happens | Key idea |
|---|------|--------------|----------|
| 1 | Business problem | Missed churner costs $150, a wasted offer costs $30 | Define success in money, not just accuracy |
| 2 | Data | 5,030 rows x 17 columns | |
| 3 | Cleaning | 30 duplicates removed, contract text standardised, missing charges repaired from `total = monthly x tenure` | Use domain logic before generic imputation |
| 4 | EDA | Churn is 27.3% overall; 41% on month-to-month vs 4% on two-year contracts | Look at the data before modelling |
| 5 | Features | `num_addons`, `is_new_customer`, `high_bill_no_support` | Build features from inputs only, never the target |
| 6 | Split | 80/20 stratified; test set untouched until the end | Prevents fooling yourself |
| 7 | Preprocessing | Impute + scale numbers, impute + one-hot categories, inside a `Pipeline` | No data leakage |
| 8 | Model comparison | 5-fold CV on the train set | Always include a dumb baseline |
| 9 | Tuning | `RandomizedSearchCV` on the winner | |
| 10 | Threshold | Choose the cut-off that minimises cost, using out-of-fold predictions | 0.5 is rarely the right threshold |
| 11 | Final test | One evaluation on the hold-out set | |
| 12 | Explain | Permutation importance | Tell the business *why* |
| 13 | Save / use | `joblib` model + `predict.py` | A model nobody can run is not a project |

## Results

Cross-validation on the training set (ROC-AUC, higher is better):

| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| **Logistic Regression** | **0.822** | 0.638 |
| Random Forest | 0.808 | 0.610 |
| Gradient Boosting | 0.794 | 0.598 |
| Baseline (always "stay") | 0.500 | 0.273 |

Hold-out test set (1,000 customers), tuned Logistic Regression at threshold 0.14:

- ROC-AUC **0.840**, PR-AUC **0.645**
- Catches **93%** of churners (recall); 43% of flagged customers really churn (precision)
- Top 10% riskiest customers churn at **69%**, which is 2.5x the average rate

| Strategy | Cost on 1,000 customers |
|---|---|
| Do nothing | $40,950 |
| Contact everyone | $21,810 |
| **Use the model** | **$12,900** |

**What drives churn:** contract type by far, then tenure, monthly charges, support calls, and internet service (see `figures/04_feature_importance.png`).

## Honest notes (good things to say in an interview)

- **Simple won.** Logistic Regression beat the tree models. That is plausible here because the synthetic churn signal is mostly additive, and it is a reminder to try simple models first.
- **The low threshold (0.14)** is a consequence of the cost assumption (a missed churner is 5x worse than a wasted offer). Change `COST_FN` / `COST_FP` and it moves. The model's value is mostly in *excluding* low-risk customers, not in only flagging a few.
- **Synthetic data means the numbers are not real-world benchmarks.** Real data is noisier; expect lower AUC.
- **Simplification:** the cost model assumes every contacted churner is saved. In reality offers succeed only some of the time; add a success rate to be more realistic.

## Ideas to extend it

1. Replace the synthetic data with the real Kaggle Telco dataset.
2. Add SHAP explanations (`pip install shap`) for per-customer reasons.
3. Try class weights / SMOTE and compare against threshold tuning.
4. Calibrate probabilities (`CalibratedClassifierCV`).
5. Wrap `predict.py` in a FastAPI endpoint or a Streamlit app.
6. Track experiments with MLflow.
