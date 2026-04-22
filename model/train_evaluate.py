"""
RTW Readiness ML Pipeline
==========================
Models: Logistic Regression (baseline) + Random Forest (primary)
Evaluation: Accuracy, Precision, Recall, F1, ROC-AUC
Interpretation: Feature importance + permutation importance
"""

import sys
import os
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data"))
from generate_dataset import generate_dataset

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("RTW READINESS PREDICTION — ML PIPELINE")
print("=" * 65)

df = generate_dataset(400)
df.to_csv("../data/synthetic_rtw_dataset.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
TARGET = "rtw_ready"
NUMERIC_FEATURES = [
    "fatigue_score", "pain_score", "cognitive_limitation_score",
    "physical_functioning_score", "anxiety_score", "rtw_confidence_score",
    "months_since_treatment", "comorbidities", "age",
    "depression_indicator", "employer_support",
]
CATEGORICAL_FEATURES = [
    "cancer_type", "treatment_type", "disease_stage",
    "work_behavior_type", "socioeconomic_status",
    "job_type", "employer_flexibility",
]

X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = df[TARGET]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
])

# ─────────────────────────────────────────────────────────────────────────────
# 3. MODELS
# ─────────────────────────────────────────────────────────────────────────────
lr_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(max_iter=1000, random_state=42, C=1.0)),
])

rf_pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=5,
        random_state=42, n_jobs=-1,
    )),
])

models = {"Logistic Regression": lr_pipeline, "Random Forest": rf_pipeline}

# ─────────────────────────────────────────────────────────────────────────────
# 4. CROSS-VALIDATION
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Cross-Validation (5-fold Stratified) ──")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, model in models.items():
    auc_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
    cv_results[name] = auc_scores
    print(f"  {name:22s}: AUC = {auc_scores.mean():.3f} ± {auc_scores.std():.3f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FIT & EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Test-Set Evaluation ──")
test_results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    test_results[name] = {"model": model, "y_pred": y_pred, "y_prob": y_prob, "auc": auc}
    print(f"\n  {name}")
    print(f"  ROC-AUC: {auc:.3f}")
    report = classification_report(y_test, y_pred, target_names=["Not Ready", "Ready"])
    print("\n".join("    " + l for l in report.splitlines()))

# Primary model
primary_name = "Random Forest"
primary = test_results[primary_name]

# ─────────────────────────────────────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (RF native + permutation)
# ─────────────────────────────────────────────────────────────────────────────
rf_clf = rf_pipeline.named_steps["clf"]
prep = rf_pipeline.named_steps["prep"]

# Get feature names after encoding
cat_feature_names = list(
    prep.named_transformers_["cat"].get_feature_names_out(CATEGORICAL_FEATURES)
)
all_feature_names = NUMERIC_FEATURES + cat_feature_names

importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance": rf_clf.feature_importances_,
}).sort_values("importance", ascending=False)

# Group back to original feature names for readability
def map_to_original(feat_name):
    for orig in CATEGORICAL_FEATURES:
        if feat_name.startswith(orig + "_"):
            return orig
    return feat_name

importance_df["original_feature"] = importance_df["feature"].apply(map_to_original)
grouped_importance = (
    importance_df.groupby("original_feature")["importance"]
    .sum()
    .reset_index()
    .sort_values("importance", ascending=False)
    .rename(columns={"original_feature": "feature"})
)

print("\n── Top Feature Importances (grouped) ──")
for _, row in grouped_importance.head(10).iterrows():
    bar = "█" * int(row["importance"] * 200)
    print(f"  {row['feature']:30s} {row['importance']:.4f}  {bar}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
VISUALS_DIR = "../visuals"
os.makedirs(VISUALS_DIR, exist_ok=True)

PALETTE = {"ready": "#2196F3", "not_ready": "#FF7043", "neutral": "#78909C"}
plt.rcParams.update({
    "font.family": "sans-serif", "axes.spines.top": False,
    "axes.spines.right": False, "axes.labelsize": 11,
    "axes.titlesize": 13, "figure.dpi": 150,
})

# ── 7a: Master dashboard ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 13))
fig.patch.set_facecolor("#FAFAFA")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.38)

# Panel 1: ROC curves
ax1 = fig.add_subplot(gs[0, 0])
for name, res in test_results.items():
    fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
    color = PALETTE["ready"] if name == "Random Forest" else PALETTE["not_ready"]
    ax1.plot(fpr, tpr, lw=2.5, color=color,
             label=f"{name}\n(AUC={res['auc']:.3f})")
ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)
ax1.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["ready"])
ax1.set_xlabel("False Positive Rate")
ax1.set_ylabel("True Positive Rate")
ax1.set_title("ROC Curves")
ax1.legend(fontsize=9, loc="lower right")

# Panel 2: Confusion matrix (RF)
ax2 = fig.add_subplot(gs[0, 1])
cm = confusion_matrix(y_test, primary["y_pred"])
disp = ConfusionMatrixDisplay(cm, display_labels=["Not Ready", "Ready"])
disp.plot(ax=ax2, colorbar=False, cmap="Blues")
ax2.set_title("Confusion Matrix\n(Random Forest)")

# Panel 3: Cross-val AUC box
ax3 = fig.add_subplot(gs[0, 2])
cv_data = [cv_results["Logistic Regression"], cv_results["Random Forest"]]
bp = ax3.boxplot(cv_data, patch_artist=True, widths=0.5,
                 medianprops=dict(color="white", linewidth=2))
colors = [PALETTE["not_ready"], PALETTE["ready"]]
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.85)
ax3.set_xticklabels(["Logistic\nRegression", "Random\nForest"])
ax3.set_ylabel("ROC-AUC")
ax3.set_title("5-Fold CV AUC Distribution")
ax3.set_ylim(0.5, 1.0)
ax3.axhline(0.8, ls="--", color="gray", lw=1, alpha=0.5, label="AUC=0.80")
ax3.legend(fontsize=8)

# Panel 4: Feature importance
ax4 = fig.add_subplot(gs[1, :2])
top_n = grouped_importance.head(12)
colors_imp = [PALETTE["ready"] if i < 3 else PALETTE["neutral"]
              for i in range(len(top_n))]
bars = ax4.barh(top_n["feature"][::-1], top_n["importance"][::-1],
                color=colors_imp[::-1], edgecolor="white", linewidth=0.5)
for bar, val in zip(bars, top_n["importance"][::-1]):
    ax4.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
             f"{val:.3f}", va="center", fontsize=8.5, color="#333")
ax4.set_xlabel("Feature Importance (Mean Decrease Impurity)")
ax4.set_title("Feature Importance — Random Forest\n(blue = top 3 predictors)")
ax4.set_xlim(0, top_n["importance"].max() * 1.18)

# Panel 5: Precision-Recall curve
ax5 = fig.add_subplot(gs[1, 2])
for name, res in test_results.items():
    prec, rec, _ = precision_recall_curve(y_test, res["y_prob"])
    ap = average_precision_score(y_test, res["y_prob"])
    color = PALETTE["ready"] if name == "Random Forest" else PALETTE["not_ready"]
    ax5.plot(rec, prec, lw=2.5, color=color, label=f"{name} (AP={ap:.3f})")
ax5.set_xlabel("Recall")
ax5.set_ylabel("Precision")
ax5.set_title("Precision-Recall Curve")
ax5.legend(fontsize=9)
ax5.set_xlim(0, 1)
ax5.set_ylim(0, 1.05)

fig.suptitle(
    "RTW Readiness Prediction — ML Dashboard\n"
    "Synthetic dataset grounded in cancer survivorship literature",
    fontsize=15, fontweight="bold", y=1.01
)
plt.savefig(f"{VISUALS_DIR}/rtw_dashboard.png", bbox_inches="tight", dpi=150)
print(f"\n✓ Saved: rtw_dashboard.png")

# ── 7b: Clinical insights plot ───────────────────────────────────────────────
fig2, axes = plt.subplots(1, 3, figsize=(16, 5))
fig2.patch.set_facecolor("#FAFAFA")

# Fatigue vs RTW (strongest predictor from literature)
ax = axes[0]
ready_fatigue = df[df["rtw_ready"] == 1]["fatigue_score"]
not_ready_fatigue = df[df["rtw_ready"] == 0]["fatigue_score"]
ax.violinplot([not_ready_fatigue, ready_fatigue], positions=[0, 1],
              showmedians=True)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Not Ready", "Ready"])
ax.set_ylabel("Fatigue Score (1–10)")
ax.set_title("Fatigue Score by RTW Readiness\n(literature: strongest predictor)")
ax.axhline(7, ls="--", color="red", alpha=0.4, lw=1.5, label="High fatigue threshold")
ax.legend(fontsize=8)

# Confidence vs RTW
ax = axes[1]
ready_conf = df[df["rtw_ready"] == 1]["rtw_confidence_score"]
not_ready_conf = df[df["rtw_ready"] == 0]["rtw_confidence_score"]
ax.violinplot([not_ready_conf, ready_conf], positions=[0, 1], showmedians=True)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Not Ready", "Ready"])
ax.set_ylabel("RTW Confidence (1–10)")
ax.set_title("RTW Confidence by Readiness\n(Hou 2021: WAI significant predictor)")

# Work behavior type vs RTW rate
ax = axes[2]
wb_rtw = df.groupby("work_behavior_type")["rtw_ready"].mean().sort_values()
colors_wb = [PALETTE["not_ready"] if v < 0.5 else PALETTE["ready"]
             for v in wb_rtw.values]
bars2 = ax.barh(wb_rtw.index, wb_rtw.values, color=colors_wb, edgecolor="white")
ax.axvline(0.5, ls="--", color="gray", lw=1.5, alpha=0.6)
ax.set_xlabel("Proportion RTW Ready")
ax.set_title("Work Behavior Type vs RTW Rate\n(Ullrich 2022: unambitious OR=4.48)")
for bar, val in zip(bars2, wb_rtw.values):
    ax.text(val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}", va="center", fontsize=9)
ax.set_xlim(0, 1.15)

fig2.suptitle("Clinical Insights from Feature Analysis", fontsize=14,
              fontweight="bold")
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/clinical_insights.png", bbox_inches="tight", dpi=150)
print(f"✓ Saved: clinical_insights.png")

# ── 7c: Permutation importance (model-agnostic validation) ───────────────────
X_test_transformed = preprocessor.transform(X_test)
rf_only = rf_pipeline.named_steps["clf"]
perm_result = permutation_importance(
    rf_only, X_test_transformed, y_test, n_repeats=20, random_state=42, n_jobs=-1
)
perm_df = pd.DataFrame({
    "feature": all_feature_names,
    "mean": perm_result.importances_mean,
    "std": perm_result.importances_std,
}).sort_values("mean", ascending=False).head(15)

fig3, ax = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor("#FAFAFA")
colors_perm = [PALETTE["ready"] if i < 3 else PALETTE["neutral"]
               for i in range(len(perm_df))]
ax.barh(perm_df["feature"][::-1], perm_df["mean"][::-1],
        xerr=perm_df["std"][::-1], color=colors_perm[::-1],
        capsize=3, edgecolor="white")
ax.set_xlabel("Mean Decrease in AUC (Permutation Importance)")
ax.set_title("Permutation Feature Importance — Random Forest\n"
             "(more reliable than MDI; blue = top 3)")
ax.axvline(0, color="black", lw=0.8)
plt.tight_layout()
plt.savefig(f"{VISUALS_DIR}/permutation_importance.png", bbox_inches="tight", dpi=150)
print(f"✓ Saved: permutation_importance.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CLINICAL INTERPRETATION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("CLINICAL INTERPRETATION SUMMARY")
print("=" * 65)

high_fatigue = df[df["fatigue_score"] > 7]["rtw_ready"].mean()
low_fatigue = df[df["fatigue_score"] <= 4]["rtw_ready"].mean()
print(f"\n• Patients with fatigue > 7: {high_fatigue:.1%} RTW ready")
print(f"  Patients with fatigue ≤ 4: {low_fatigue:.1%} RTW ready")

high_anx = df[(df["anxiety_score"] > 6) & (df["fatigue_score"] > 7)]["rtw_ready"].mean()
print(f"\n• Fatigue >7 AND anxiety >6: {high_anx:.1%} RTW ready")

for wb in ["resigned", "unambitious", "healthy_ambitious"]:
    pct = df[df["work_behavior_type"] == wb]["rtw_ready"].mean()
    print(f"\n• Work behavior = {wb:22s}: {pct:.1%} RTW ready")

chemo_rate = df[df["treatment_type"].isin(["surgery_chemo", "surgery_chemo_radiation"])]["rtw_ready"].mean()
no_chemo_rate = df[df["treatment_type"].isin(["surgery_only", "surgery_radiation"])]["rtw_ready"].mean()
print(f"\n• Chemotherapy patients: {chemo_rate:.1%} RTW ready")
print(f"  No chemotherapy:        {no_chemo_rate:.1%} RTW ready")

print("\n" + "=" * 65)
print("Pipeline complete. Visuals saved to ../visuals/")
print("=" * 65)
