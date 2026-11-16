"""
Customer Churn Prediction - an end-to-end data science project
================================================================

Business problem
    A telecom company loses ~27% of its customers. Retention offers cost money,
    so we want to predict WHO is likely to leave and contact only those people.

Pipeline (each step is a function below)
    1. Get data            -> generate_data()   (synthetic, realistic, deliberately messy)
    2. Clean               -> clean_data()
    3. Explore (EDA)       -> make_eda_figures()
    4. Feature engineering -> add_features()
    5. Preprocess          -> build_preprocessor()   (inside a sklearn Pipeline = no leakage)
    6. Compare models      -> cross-validation on the TRAIN set only
    7. Tune best model     -> RandomizedSearchCV
    8. Choose threshold    -> minimise business cost on out-of-fold predictions
    9. Evaluate ONCE       -> hold-out test set
   10. Explain             -> permutation importance
   11. Save / deploy       -> joblib model + predict.py

Run:  python churn_project.py
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # no display needed; we only save PNGs
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import seaborn as sns  # only used for the correlation heatmap
import joblib
from scipy.stats import loguniform, randint

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
BASE = Path(__file__).resolve().parent
DATA_DIR, FIG_DIR, MODEL_DIR = BASE / "data", BASE / "figures", BASE / "models"
for d in (DATA_DIR, FIG_DIR, MODEL_DIR):
    d.mkdir(exist_ok=True)

# Business assumptions (change these and the best threshold changes!)
COST_FN = 150  # a customer we failed to flag leaves  -> we lose ~$150 of margin
COST_FP = 30   # a customer we flagged but would have stayed -> wasted $30 offer

NUMERIC = [
    "tenure", "monthly_charges", "total_charges", "support_calls",
    "senior_citizen", "num_addons", "is_new_customer", "high_bill_no_support",
]
CATEGORICAL = [
    "gender", "partner", "dependents", "contract", "internet_service",
    "online_security", "tech_support", "streaming_tv", "paperless_billing",
    "payment_method",
]
TARGET = "churn"

# Chart colours: validated categorical slots 1 & 2 + neutral chrome
BLUE, ORANGE, RED = "#2a78d6", "#eb6834", "#e34948"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.labelsize": 9,
})


def style_ax(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.title.set_color(INK)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# --------------------------------------------------------------------------- #
# 1. DATA
# --------------------------------------------------------------------------- #
def generate_data(n=5000, seed=RANDOM_STATE):
    """Create a realistic telecom-churn dataset.

    In a real project you would load a CSV / SQL table here instead. We simulate
    so the project runs anywhere, and we bake in realistic patterns (month-to-month
    contracts churn more, long-tenure customers churn less, ...) plus messy
    problems (missing values, duplicates, inconsistent text) to practise cleaning.
    """
    rng = np.random.default_rng(seed)

    contract = rng.choice(["Month-to-month", "One year", "Two year"], size=n, p=[0.55, 0.21, 0.24])
    shape_scale = {"Month-to-month": (1.6, 12), "One year": (3.0, 10), "Two year": (4.0, 12)}
    tenure = np.array([rng.gamma(*shape_scale[c]) for c in contract])
    tenure = np.clip(np.round(tenure), 1, 72).astype(int)

    internet = rng.choice(["DSL", "Fiber optic", "No"], size=n, p=[0.34, 0.44, 0.22])
    has_net = internet != "No"

    def addon(p):
        return np.where(has_net, rng.choice(["Yes", "No"], size=n, p=[p, 1 - p]), "No internet service")

    online_security, tech_support, streaming_tv = addon(0.30), addon(0.30), addon(0.45)

    senior = rng.binomial(1, 0.16, n)
    partner = rng.choice(["Yes", "No"], size=n, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n, p=[0.30, 0.70])
    gender = rng.choice(["Male", "Female"], size=n)
    paperless = rng.choice(["Yes", "No"], size=n, p=[0.59, 0.41])
    payment = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
        size=n, p=[0.34, 0.23, 0.22, 0.21],
    )
    # support calls in the last 3 months (fiber customers call a bit more)
    support_calls = rng.poisson(np.where(internet == "Fiber optic", 1.3, 0.8))

    base_price = pd.Series(internet).map({"DSL": 25, "Fiber optic": 50, "No": 0}).to_numpy()
    monthly = (
        20 + base_price + 6 * (online_security == "Yes") + 6 * (tech_support == "Yes")
        + 9 * (streaming_tv == "Yes") + rng.normal(0, 4, n)
    )
    monthly = np.clip(monthly, 18.5, None).round(2)
    total = (monthly * tenure * (1 + rng.normal(0, 0.02, n))).round(2)

    # --- the hidden "truth": how each factor changes the odds of leaving ----
    z0 = (
        1.3 * (contract == "Month-to-month") - 1.1 * (contract == "Two year")
        + 0.7 * (internet == "Fiber optic") - 0.5 * (internet == "No")
        - 0.035 * tenure + 0.018 * (monthly - 65)
        + 0.35 * support_calls
        - 0.6 * (tech_support == "Yes") - 0.4 * (online_security == "Yes")
        + 0.5 * (payment == "Electronic check") + 0.25 * (paperless == "Yes")
        + 0.3 * senior - 0.2 * (partner == "Yes") - 0.2 * (dependents == "Yes")
    )
    noise = rng.normal(0, 0.6, n)  # things we can't observe -> imperfect predictions
    lo, hi = -8.0, 4.0
    for _ in range(50):  # calibrate the intercept so overall churn is ~27%
        mid = (lo + hi) / 2
        rate = (1 / (1 + np.exp(-(z0 + mid + noise)))).mean()
        lo, hi = (lo, mid) if rate > 0.27 else (mid, hi)
    prob = 1 / (1 + np.exp(-(z0 + (lo + hi) / 2 + noise)))
    churn = rng.binomial(1, prob)

    df = pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "gender": gender, "senior_citizen": senior, "partner": partner, "dependents": dependents,
        "tenure": tenure, "contract": contract, "internet_service": internet,
        "online_security": online_security, "tech_support": tech_support, "streaming_tv": streaming_tv,
        "paperless_billing": paperless, "payment_method": payment,
        "support_calls": support_calls, "monthly_charges": monthly, "total_charges": total,
        "churn": np.where(churn == 1, "Yes", "No"),
    })

    # --- make it messy, like real data ---------------------------------------
    for col, frac in [("total_charges", 0.015), ("monthly_charges", 0.01), ("payment_method", 0.01)]:
        df.loc[rng.choice(n, int(n * frac), replace=False), col] = np.nan
    df.loc[rng.choice(n, 15, replace=False), "contract"] = "month-to-month "  # inconsistent text
    df = pd.concat([df, df.sample(30, random_state=seed)], ignore_index=True)  # duplicate rows
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. CLEANING
# --------------------------------------------------------------------------- #
def clean_data(df):
    df = df.copy()
    report = {"rows_before": len(df)}

    df = df.drop_duplicates()
    report["duplicates_removed"] = report["rows_before"] - len(df)

    df["contract"] = (
        df["contract"].str.strip().str.lower()
        .map({"month-to-month": "Month-to-month", "one year": "One year", "two year": "Two year"})
    )

    # total_charges ~= monthly_charges * tenure, so we can repair one from the other
    report["missing_before"] = df[["monthly_charges", "total_charges", "payment_method"]].isna().sum().to_dict()
    m = df["monthly_charges"].isna() & df["total_charges"].notna()
    df.loc[m, "monthly_charges"] = (df["total_charges"] / df["tenure"])[m].round(2)
    t = df["total_charges"].isna() & df["monthly_charges"].notna()
    df.loc[t, "total_charges"] = (df["monthly_charges"] * df["tenure"])[t].round(2)

    df[TARGET] = (df[TARGET] == "Yes").astype(int)  # target as 0/1
    report["rows_after"] = len(df)
    print(json.dumps(report, indent=2))
    print("Missing values left for the model's imputer:")
    print(df.isna().sum()[lambda s: s > 0].to_string())
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 3. EDA
# --------------------------------------------------------------------------- #
def make_eda_figures(df):
    overall = df[TARGET].mean()
    print(f"Overall churn rate: {overall:.1%}  ({df[TARGET].sum()} of {len(df)} customers)")
    for col in ["contract", "internet_service", "payment_method"]:
        print(f"\nChurn rate by {col}:")
        print((df.groupby(col)[TARGET].mean() * 100).round(1).sort_values(ascending=False).to_string())

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), constrained_layout=True)
    fig.suptitle(f"Who churns? Overall churn rate = {overall:.1%}", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)

    def rate_bar(ax, col, title, order=None, wrap=False):
        s = df.groupby(col)[TARGET].mean()
        s = s.reindex(order) if order else s.sort_values(ascending=False)
        bars = ax.bar(range(len(s)), s.values, width=0.55, color=ORANGE)
        for b, v in zip(bars, s.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.0%}", ha="center", va="bottom",
                    fontsize=9, color=INK2)
        ax.set_xticks(range(len(s)))
        ax.set_xticklabels([str(i).replace(" ", "\n") if wrap else str(i) for i in s.index])
        ax.set_ylim(0, max(s.values) * 1.2)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.set_title(title)
        style_ax(ax)

    rate_bar(axes[0, 0], "contract", "Churn rate by contract", ["Month-to-month", "One year", "Two year"], wrap=True)
    rate_bar(axes[0, 1], "internet_service", "Churn rate by internet service", wrap=True)
    rate_bar(axes[0, 2], "payment_method", "Churn rate by payment method", wrap=True)

    # tenure distribution: two series -> blue (retained) vs orange (churned) + legend
    ax = axes[1, 0]
    bins = np.arange(0, 76, 4)
    ax.hist(df.loc[df[TARGET] == 0, "tenure"], bins=bins, density=True, histtype="stepfilled",
            color=BLUE, alpha=0.65, edgecolor=SURFACE, linewidth=1.2)
    ax.hist(df.loc[df[TARGET] == 1, "tenure"], bins=bins, density=True, histtype="stepfilled",
            color=ORANGE, alpha=0.65, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xlabel("Tenure (months)")
    ax.set_ylabel("Share of customers")
    ax.set_title("Churners leave early")
    ax.legend(handles=[Patch(color=BLUE, alpha=0.65, label="Retained"),
                       Patch(color=ORANGE, alpha=0.65, label="Churned")], frameon=False,
              labelcolor=INK2, fontsize=9)
    style_ax(ax)

    ax = axes[1, 1]
    data = [df.loc[df[TARGET] == 0, "monthly_charges"].dropna(), df.loc[df[TARGET] == 1, "monthly_charges"].dropna()]
    bp = ax.boxplot(data, patch_artist=True, widths=0.45, tick_labels=["Retained", "Churned"],
                    medianprops={"color": INK, "linewidth": 1.8}, whiskerprops={"color": AXIS},
                    capprops={"color": AXIS}, flierprops={"markersize": 3, "markeredgecolor": MUTED})
    for patch, c in zip(bp["boxes"], (BLUE, ORANGE)):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor(c)
    ax.set_ylabel("Monthly charges ($)")
    ax.set_title("Churners pay more per month")
    ax.legend(handles=[Patch(color=BLUE, alpha=0.75, label="Retained"),
                       Patch(color=ORANGE, alpha=0.75, label="Churned")], frameon=False,
              labelcolor=INK2, fontsize=9, loc="upper left")
    style_ax(ax)

    ax = axes[1, 2]
    calls = df["support_calls"].clip(upper=4).rename("calls")
    s = df.groupby(calls)[TARGET].mean()
    bars = ax.bar(range(len(s)), s.values, width=0.55, color=ORANGE)
    for b, v in zip(bars, s.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.0%}", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels(["0", "1", "2", "3", "4+"])
    ax.set_xlabel("Support calls (last 3 months)")
    ax.set_ylim(0, s.max() * 1.2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_title("More support calls, more churn")
    style_ax(ax)

    fig.savefig(FIG_DIR / "01_eda_overview.png", dpi=140)
    plt.close(fig)

    # correlation heatmap (diverging blue <-> red with a neutral midpoint)
    num = df[["tenure", "monthly_charges", "total_charges", "support_calls", "senior_citizen", TARGET]]
    corr = num.corr()
    cmap = LinearSegmentedColormap.from_list("div", [BLUE, "#f0efec", RED])
    fig, ax = plt.subplots(figsize=(6.8, 5.6), constrained_layout=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, vmin=-1, vmax=1, center=0, linewidths=2,
                linecolor=SURFACE, cbar_kws={"shrink": 0.8}, ax=ax,
                annot_kws={"fontsize": 9, "color": INK})
    ax.set_title("Correlation between numeric features", loc="left", color=INK, fontweight="bold")
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    fig.savefig(FIG_DIR / "02_correlation.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def add_features(df):
    """Create new columns from existing ones (never from the target!)."""
    df = df.copy()
    addons = ["online_security", "tech_support", "streaming_tv"]
    df["num_addons"] = (df[addons] == "Yes").sum(axis=1)
    df["is_new_customer"] = (df["tenure"] <= 6).astype(int)
    df["high_bill_no_support"] = ((df["monthly_charges"] > 80) & (df["tech_support"] != "Yes")).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 5. PREPROCESSING + 6/7. MODELS
# --------------------------------------------------------------------------- #
def build_preprocessor():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)])


def make_pipeline(model):
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


MODELS = {
    "Baseline (always 'stay')": DummyClassifier(strategy="prior"),
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE),
    "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
}

SEARCH_SPACES = {
    "Logistic Regression": {"model__C": loguniform(1e-3, 1e2)},
    "Random Forest": {
        "model__n_estimators": randint(200, 500),
        "model__max_depth": [None, 6, 10, 16],
        "model__min_samples_leaf": randint(2, 20),
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "Gradient Boosting": {
        "model__learning_rate": loguniform(0.02, 0.2),
        "model__max_leaf_nodes": randint(8, 40),
        "model__min_samples_leaf": randint(10, 60),
        "model__l2_regularization": loguniform(1e-3, 10),
        "model__max_iter": randint(100, 400),
    },
}


def compare_models(X_train, y_train, cv):
    rows = []
    for name, model in MODELS.items():
        res = cross_validate(make_pipeline(model), X_train, y_train, cv=cv,
                             scoring={"auc": "roc_auc", "ap": "average_precision"}, n_jobs=-1)
        rows.append({"model": name, "cv_roc_auc": res["test_auc"].mean(), "auc_std": res["test_auc"].std(),
                     "cv_pr_auc": res["test_ap"].mean()})
    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    print(table.round(4).to_string(index=False))
    return table


# --------------------------------------------------------------------------- #
# 8. BUSINESS THRESHOLD
# --------------------------------------------------------------------------- #
def cost_curve(y_true, proba, thresholds):
    y_true = np.asarray(y_true)
    costs = []
    for t in thresholds:
        pred = proba >= t
        fn = np.sum((~pred) & (y_true == 1))
        fp = np.sum(pred & (y_true == 0))
        costs.append((fn * COST_FN + fp * COST_FP) / len(y_true))
    return np.array(costs)


# --------------------------------------------------------------------------- #
# 9/10. FIGURES FOR RESULTS
# --------------------------------------------------------------------------- #
def make_eval_figures(y_test, proba, thresholds, oof_cost, threshold, y_oof):
    fpr, tpr, _ = roc_curve(y_test, proba)
    prec, rec, _ = precision_recall_curve(y_test, proba)
    auc, ap = roc_auc_score(y_test, proba), average_precision_score(y_test, proba)
    prevalence = float(np.mean(y_test))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9), constrained_layout=True)
    fig.suptitle("How good is the final model? (hold-out test set)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold", color=INK)

    ax = axes[0, 0]
    ax.plot(fpr, tpr, color=BLUE, linewidth=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1.2, linestyle="--", label="Random guessing")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="lower right")
    style_ax(ax, "y")

    ax = axes[0, 1]
    ax.plot(rec, prec, color=BLUE, linewidth=2, label=f"Model (PR-AUC = {ap:.3f})")
    ax.axhline(prevalence, color=MUTED, linewidth=1.2, linestyle="--", label=f"Random ({prevalence:.0%} churn)")
    ax.set_xlabel("Recall (share of churners we catch)")
    ax.set_ylabel("Precision (share of flags that are right)")
    ax.set_ylim(0, 1.02)
    ax.set_title("Precision vs recall")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper right")
    style_ax(ax, "y")

    ax = axes[1, 0]
    no_action = y_oof.mean() * COST_FN
    contact_all = (1 - y_oof.mean()) * COST_FP
    ax.plot(thresholds, oof_cost, color=BLUE, linewidth=2)
    ax.axhline(no_action, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(0.97, no_action, "Do nothing", ha="right", va="bottom", fontsize=9, color=INK2, transform=ax.get_yaxis_transform())
    ax.axhline(contact_all, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(0.97, contact_all, "Contact everyone", ha="right", va="bottom", fontsize=9, color=INK2, transform=ax.get_yaxis_transform())
    best_cost = oof_cost[np.argmin(np.abs(thresholds - threshold))]
    ax.plot([threshold], [best_cost], "o", color=ORANGE, markersize=9, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.annotate(f"best threshold = {threshold:.2f}", (threshold, best_cost), xytext=(14, 46),
                textcoords="offset points", fontsize=9, color=INK,
                arrowprops={"arrowstyle": "-", "color": MUTED, "linewidth": 1})
    ax.set_xlabel("Probability threshold for contacting a customer")
    ax.set_ylabel("Expected cost per customer ($)")
    ax.set_title(f"Choosing the threshold (missed churner \\${COST_FN}, wasted offer \\${COST_FP})")
    style_ax(ax, "y")

    ax = axes[1, 1]
    cm = confusion_matrix(y_test, proba >= threshold)
    seq = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#2a78d6", "#0d366b"])
    ax.imshow(cm, cmap=seq, vmin=0, vmax=cm.max() * 1.1)
    labels = [["True negative\n(left alone, stayed)", "False positive\n(wasted offer)"],
              ["False negative\n(missed churner)", "True positive\n(caught churner)"]]
    for i in range(2):
        for j in range(2):
            color = "#ffffff" if cm[i, j] > cm.max() * 0.5 else INK
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center", fontsize=10, color=color)
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actually stayed", "Actually churned"])
    ax.set_title(f"Confusion matrix at threshold {threshold:.2f}")
    ax.tick_params(colors=INK2, labelsize=9, length=0)
    for s in ax.spines.values():
        s.set_visible(False)

    fig.savefig(FIG_DIR / "03_model_evaluation.png", dpi=140)
    plt.close(fig)


def make_importance_figure(imp):
    top = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 5.6), constrained_layout=True)
    ax.barh(top["feature"], top["importance"], xerr=top["std"], height=0.6, color=BLUE,
            error_kw={"ecolor": MUTED, "elinewidth": 1, "capsize": 2})
    ax.set_xlabel("Drop in ROC-AUC when the feature is shuffled")
    ax.set_title("What drives the predictions?")
    style_ax(ax, "x")
    fig.savefig(FIG_DIR / "04_feature_importance.png", dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    header("1. DATA")
    raw = generate_data()
    raw.to_csv(DATA_DIR / "telco_churn_raw.csv", index=False)
    print(f"Raw data: {raw.shape[0]} rows x {raw.shape[1]} columns  -> data/telco_churn_raw.csv")
    print(raw.head(5).to_string(index=False))

    header("2. CLEANING")
    df = clean_data(raw)
    df.to_csv(DATA_DIR / "telco_churn_clean.csv", index=False)

    header("3. EXPLORATORY DATA ANALYSIS")
    make_eda_figures(df)
    print("\nSaved figures/01_eda_overview.png and figures/02_correlation.png")

    header("4. FEATURE ENGINEERING")
    df = add_features(df)
    print("Added: num_addons, is_new_customer, high_bill_no_support")

    header("5. TRAIN / TEST SPLIT")
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    print(f"Train: {len(X_train)} rows (churn {y_train.mean():.1%}) | Test: {len(X_test)} rows (churn {y_test.mean():.1%})")
    print("The test set is locked away until the very end.")

    header("6. COMPARE MODELS (5-fold cross-validation, train set only)")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    table = compare_models(X_train, y_train, cv)
    winner = table[table["model"] != "Baseline (always 'stay')"].iloc[0]["model"]
    print(f"\nBest by CV ROC-AUC: {winner}")

    header(f"7. TUNE THE WINNER: {winner}")
    search = RandomizedSearchCV(
        make_pipeline(MODELS[winner]), SEARCH_SPACES[winner], n_iter=15, cv=StratifiedKFold(3, shuffle=True, random_state=RANDOM_STATE),
        scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1, refit=True)
    search.fit(X_train, y_train)
    best = search.best_estimator_
    print(f"Best CV ROC-AUC after tuning: {search.best_score_:.4f}")
    print("Best params:", {k.replace('model__', ''): (round(float(v), 4) if isinstance(v, (float, np.floating)) else v)
                           for k, v in search.best_params_.items()})

    header("8. CHOOSE THE DECISION THRESHOLD (business cost)")
    oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
    thresholds = np.linspace(0.05, 0.95, 91)
    oof_cost = cost_curve(y_train, oof, thresholds)
    threshold = float(thresholds[np.argmin(oof_cost)])
    print(f"Cost of a missed churner = ${COST_FN}, cost of a wasted offer = ${COST_FP}")
    print(f"Best threshold (from out-of-fold train predictions): {threshold:.2f}")

    header("9. FINAL EVALUATION ON THE HOLD-OUT TEST SET")
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    auc, ap = roc_auc_score(y_test, proba), average_precision_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    print(f"ROC-AUC {auc:.3f} | PR-AUC {ap:.3f} | Brier {brier:.3f}")
    print(f"\nClassification report at threshold {threshold:.2f}:")
    print(classification_report(y_test, pred, target_names=["Stayed", "Churned"], digits=3))

    order = np.argsort(-proba)
    top = order[: int(0.10 * len(order))]
    lift = y_test.to_numpy()[top].mean() / y_test.mean()
    print(f"Top-10% riskiest customers: {y_test.to_numpy()[top].mean():.0%} actually churn "
          f"({lift:.1f}x the average rate)")

    y_arr = y_test.to_numpy()
    model_cost = (np.sum((pred == 0) & (y_arr == 1)) * COST_FN + np.sum((pred == 1) & (y_arr == 0)) * COST_FP)
    nothing_cost = y_arr.sum() * COST_FN
    everyone_cost = (len(y_arr) - y_arr.sum()) * COST_FP
    print(f"\nBusiness cost on {len(y_arr)} test customers:")
    print(f"  Do nothing        : ${nothing_cost:,.0f}")
    print(f"  Contact everyone  : ${everyone_cost:,.0f}")
    print(f"  Use the model     : ${model_cost:,.0f}  (saves ${nothing_cost - model_cost:,.0f} vs doing nothing)")

    make_eval_figures(y_test, proba, thresholds, oof_cost, threshold, y_train.to_numpy())

    header("10. EXPLAIN THE MODEL (permutation importance on test set)")
    r = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=10,
                               random_state=RANDOM_STATE, n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_test.columns, "importance": r.importances_mean, "std": r.importances_std})
           .sort_values("importance", ascending=False).reset_index(drop=True))
    print(imp.head(10).round(4).to_string(index=False))
    make_importance_figure(imp)

    header("11. SAVE ARTIFACTS")
    joblib.dump(best, MODEL_DIR / "churn_model.joblib")
    metrics = {
        "model": winner, "best_params": {k: (float(v) if isinstance(v, (float, np.floating)) else
                                              int(v) if isinstance(v, (int, np.integer)) else v)
                                         for k, v in search.best_params_.items()},
        "cv_table": table.round(4).to_dict(orient="records"),
        "test_roc_auc": round(auc, 4), "test_pr_auc": round(ap, 4), "test_brier": round(brier, 4),
        "threshold": threshold, "cost_fn": COST_FN, "cost_fp": COST_FP,
        "top_decile_lift": round(float(lift), 2),
        "business_cost": {"do_nothing": float(nothing_cost), "contact_everyone": float(everyone_cost),
                          "model": float(model_cost)},
        "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    scored = X_test.copy()
    scored.insert(0, "customer_id", df.loc[X_test.index, "customer_id"])
    scored["churn_probability"] = proba.round(4)
    scored["actually_churned"] = y_test
    scored.sort_values("churn_probability", ascending=False).to_csv(DATA_DIR / "test_predictions.csv", index=False)
    print("Saved: models/churn_model.joblib, models/metrics.json, data/test_predictions.csv")
    print("\nDone. Try:  python predict.py")


if __name__ == "__main__":
    main()
