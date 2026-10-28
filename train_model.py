# ============================================================
# STROKE PREDICTION SYSTEM
# Machine Learning Based Stroke Prediction
# Using Balanced Random Forest
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from imblearn.ensemble import BalancedRandomForestClassifier

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

DATA_PATH = "data/healthcare-dataset-stroke-data.csv"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("STROKE PREDICTION MODEL TRAINING")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset shape: {df.shape}")
print(f"Number of records: {len(df)}")


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\nDataset columns:")
print(list(df.columns))

print("\nStroke class distribution:")
print(df["stroke"].value_counts())

print("\nStroke class percentage:")
print(df["stroke"].value_counts(normalize=True) * 100)


# ============================================================
# 4. REMOVE ID COLUMN
# ============================================================

if "id" in df.columns:
    df = df.drop(columns=["id"])

print("\nID column removed.")


# ============================================================
# 5. DEFINE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["stroke"])
y = df["stroke"]


# ============================================================
# 6. DEFINE FEATURE TYPES
# ============================================================

categorical_features = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status"
]

numerical_features = [
    "age",
    "hypertension",
    "heart_disease",
    "avg_glucose_level",
    "bmi"
]


# ============================================================
# 7. STRATIFIED 70/15/15 SPLIT
# ============================================================

print("\n" + "=" * 70)
print("DATA SPLITTING")
print("=" * 70)

# First split:
# 70% training
# 30% temporary

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=RANDOM_STATE,
    stratify=y
)

# Second split:
# 15% validation
# 15% test

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp
)

print(f"\nTraining set:   {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
print(f"Validation set: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
print(f"Test set:       {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

print("\nTraining stroke distribution:")
print(y_train.value_counts())

print("\nValidation stroke distribution:")
print(y_val.value_counts())

print("\nTest stroke distribution:")
print(y_test.value_counts())


# ============================================================
# 8. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# ============================================================
# 9. FIT PREPROCESSOR ONLY ON TRAINING DATA
# ============================================================

print("\nFitting preprocessing pipeline...")

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)

print("Preprocessing completed.")

print(f"Processed training shape: {X_train_processed.shape}")
print(f"Processed validation shape: {X_val_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")


# ============================================================
# 10. HELPER FUNCTION FOR EVALUATION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    probabilities,
    threshold=0.50
):
    """
    Evaluate binary classification model at a selected threshold.
    """

    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_true, predictions)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities
    )

    cm = confusion_matrix(y_true, predictions)

    tn, fp, fn, tp = cm.ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print(f"Threshold    : {threshold:.4f}")
    print(f"Accuracy     : {accuracy:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")
    print(f"Specificity  : {specificity:.4f}")
    print(f"F1 Score     : {f1:.4f}")
    print(f"ROC-AUC      : {roc_auc:.4f}")
    print(f"PR-AUC       : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    return {
        "Model": model_name,
        "Threshold": threshold,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "Sensitivity": recall,
        "Specificity": specificity,
        "F1": f1,
        "ROC_AUC": roc_auc,
        "PR_AUC": pr_auc,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }


# ============================================================
# 11. TRAIN MULTIPLE MODELS
# ============================================================

results = {}

probabilities = {}

models = {}


# ------------------------------------------------------------
# 11.1 Logistic Regression
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

logistic_model = LogisticRegression(
    max_iter=3000,
    class_weight="balanced",
    C=0.5,
    solver="liblinear",
    random_state=RANDOM_STATE
)

logistic_model.fit(
    X_train_processed,
    y_train
)

logistic_prob = logistic_model.predict_proba(
    X_test_processed
)[:, 1]

probabilities["Logistic Regression"] = logistic_prob

results["Logistic Regression"] = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_prob,
    threshold=0.50
)


# ------------------------------------------------------------
# 11.2 Decision Tree
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING DECISION TREE")
print("=" * 70)

decision_tree = DecisionTreeClassifier(
    max_depth=8,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=RANDOM_STATE
)

decision_tree.fit(
    X_train_processed,
    y_train
)

dt_prob = decision_tree.predict_proba(
    X_test_processed
)[:, 1]

probabilities["Decision Tree"] = dt_prob

results["Decision Tree"] = evaluate_model(
    "Decision Tree",
    y_test,
    dt_prob,
    threshold=0.50
)


# ------------------------------------------------------------
# 11.3 Random Forest
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

random_forest = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

random_forest.fit(
    X_train_processed,
    y_train
)

rf_prob = random_forest.predict_proba(
    X_test_processed
)[:, 1]

probabilities["Random Forest"] = rf_prob

results["Random Forest"] = evaluate_model(
    "Random Forest",
    y_test,
    rf_prob,
    threshold=0.50
)


# ------------------------------------------------------------
# 11.4 Extra Trees
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("TRAINING EXTRA TREES")
print("=" * 70)

extra_trees = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

extra_trees.fit(
    X_train_processed,
    y_train
)

extra_prob = extra_trees.predict_proba(
    X_test_processed
)[:, 1]

probabilities["Extra Trees"] = extra_prob

results["Extra Trees"] = evaluate_model(
    "Extra Trees",
    y_test,
    extra_prob,
    threshold=0.50
)


# ============================================================
# 12. BALANCED RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BALANCED RANDOM FOREST")
print("=" * 70)

balanced_rf = BalancedRandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

balanced_rf.fit(
    X_train_processed,
    y_train
)

# Test probabilities at default threshold
brf_test_prob = balanced_rf.predict_proba(
    X_test_processed
)[:, 1]

probabilities["Balanced Random Forest"] = brf_test_prob

results["Balanced Random Forest"] = evaluate_model(
    "Balanced Random Forest",
    y_test,
    brf_test_prob,
    threshold=0.50
)


# ============================================================
# 13. THRESHOLD TUNING USING VALIDATION SET ONLY
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD TUNING")
print("=" * 70)

brf_val_prob = balanced_rf.predict_proba(
    X_val_processed
)[:, 1]

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

threshold_records = []

best_threshold = 0.50
best_f1 = -1

for threshold in thresholds:

    val_predictions = (
        brf_val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_predictions,
        zero_division=0
    )

    accuracy = accuracy_score(
        y_val,
        val_predictions
    )

    threshold_records.append(
        {
            "Threshold": threshold,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1": f1
        }
    )

    print(
        f"Threshold={threshold:.2f} | "
        f"Accuracy={accuracy:.4f} | "
        f"Precision={precision:.4f} | "
        f"Recall={recall:.4f} | "
        f"F1={f1:.4f}"
    )

    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold


threshold_df = pd.DataFrame(threshold_records)

print("\nSelected threshold:")
print(f"{best_threshold:.2f}")

print(f"Validation F1 at selected threshold: {best_f1:.4f}")


# ============================================================
# 14. FINAL BALANCED RANDOM FOREST TEST EVALUATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL TEST EVALUATION")
print("=" * 70)

final_result = evaluate_model(
    "Balanced Random Forest - Final",
    y_test,
    brf_test_prob,
    threshold=best_threshold
)

results["Balanced Random Forest - Final"] = final_result


# ============================================================
# 15. SAVE MODEL COMPARISON
# ============================================================

comparison_df = pd.DataFrame(
    results.values()
)

comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

comparison_df.to_csv(
    comparison_path,
    index=False
)

print(
    f"\nModel comparison saved to: {comparison_path}"
)


# ============================================================
# 16. SAVE FINAL METRICS
# ============================================================

final_metrics_df = pd.DataFrame(
    [final_result]
)

final_metrics_path = os.path.join(
    MODEL_DIR,
    "final_metrics.csv"
)

final_metrics_df.to_csv(
    final_metrics_path,
    index=False
)

print(
    f"Final metrics saved to: {final_metrics_path}"
)


# ============================================================
# 17. SAVE THRESHOLD RESULTS
# ============================================================

threshold_path = os.path.join(
    MODEL_DIR,
    "threshold_results.csv"
)

threshold_df.to_csv(
    threshold_path,
    index=False
)


# ============================================================
# 18. SAVE MODEL PACKAGE
# ============================================================

model_package = {
    "model": balanced_rf,
    "preprocessor": preprocessor,
    "threshold": float(best_threshold),
    "features": list(X.columns),
    "categorical_features": categorical_features,
    "numerical_features": numerical_features
}

model_path = os.path.join(
    MODEL_DIR,
    "stroke_prediction_model.pkl"
)

joblib.dump(
    model_package,
    model_path
)

print(
    f"Final model saved to: {model_path}"
)


# ============================================================
# 19. SAVE PREPROCESSOR SEPARATELY
# ============================================================

preprocessor_path = os.path.join(
    MODEL_DIR,
    "preprocessor.pkl"
)

joblib.dump(
    preprocessor,
    preprocessor_path
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    (brf_test_prob >= best_threshold).astype(int)
)

plt.figure(figsize=(7, 6))

plt.imshow(cm)

plt.title(
    "Confusion Matrix - Balanced Random Forest"
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["No Stroke", "Stroke"]
)

plt.yticks(
    [0, 1],
    ["No Stroke", "Stroke"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()

cm_path = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 21. ROC CURVE COMPARISON
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, prob in probabilities.items():

    fpr, tpr, _ = roc_curve(
        y_test,
        prob
    )

    auc_value = roc_auc_score(
        y_test,
        prob
    )

    plt.plot(
        fpr,
        tpr,
        label=f"{model_name} (AUC={auc_value:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison - Stroke Prediction"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

roc_path = os.path.join(
    MODEL_DIR,
    "roc_curve_comparison.png"
)

plt.savefig(
    roc_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 22. PRECISION-RECALL CURVE
# ============================================================

plt.figure(figsize=(8, 6))

for model_name, prob in probabilities.items():

    precision, recall, _ = precision_recall_curve(
        y_test,
        prob
    )

    pr_auc = average_precision_score(
        y_test,
        prob
    )

    plt.plot(
        recall,
        precision,
        label=f"{model_name} (PR-AUC={pr_auc:.3f})"
    )

plt.xlabel("Recall")
plt.ylabel("Precision")

plt.title(
    "Precision-Recall Curve - Stroke Prediction"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

pr_path = os.path.join(
    MODEL_DIR,
    "precision_recall_curve.png"
)

plt.savefig(
    pr_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 23. THRESHOLD TUNING GRAPH
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Selected Threshold = {best_threshold:.2f}"
)

plt.xlabel("Threshold")
plt.ylabel("Score")

plt.title(
    "Threshold Tuning - Balanced Random Forest"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

threshold_graph_path = os.path.join(
    MODEL_DIR,
    "threshold_tuning.png"
)

plt.savefig(
    threshold_graph_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 24. FEATURE IMPORTANCE
# ============================================================

try:

    feature_names = preprocessor.get_feature_names_out()

    importances = balanced_rf.feature_importances_

    feature_importance_df = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importances
        }
    ).sort_values(
        by="Importance",
        ascending=False
    )

    feature_importance_csv = os.path.join(
        MODEL_DIR,
        "feature_importance.csv"
    )

    feature_importance_df.to_csv(
        feature_importance_csv,
        index=False
    )

    # Top 15 features
    top_features = feature_importance_df.head(15)

    plt.figure(figsize=(10, 7))

    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance"][::-1]
    )

    plt.xlabel("Importance")

    plt.ylabel("Feature")

    plt.title(
        "Top 15 Feature Importances - Balanced Random Forest"
    )

    plt.tight_layout()

    feature_importance_png = os.path.join(
        MODEL_DIR,
        "feature_importance.png"
    )

    plt.savefig(
        feature_importance_png,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nFeature importance files saved."
    )

except Exception as e:

    print(
        "\nFeature importance could not be generated:"
    )

    print(e)


# ============================================================
# 25. CLASSIFICATION REPORT
# ============================================================

final_predictions = (
    brf_test_prob >= best_threshold
).astype(int)

report = classification_report(
    y_test,
    final_predictions,
    target_names=[
        "No Stroke",
        "Stroke"
    ],
    zero_division=0
)

print("\n" + "=" * 70)
print("FINAL CLASSIFICATION REPORT")
print("=" * 70)

print(report)


# ============================================================
# 26. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nDataset:")
print(f"Total records : {len(df)}")

print("\nData split:")
print(f"Training      : {len(X_train)}")
print(f"Validation    : {len(X_val)}")
print(f"Testing       : {len(X_test)}")

print("\nFinal model:")
print("Balanced Random Forest")

print(f"\nSelected threshold: {best_threshold:.2f}")

print("\nFinal test performance:")

print(
    f"Accuracy     : {final_result['Accuracy']:.4f}"
)

print(
    f"Precision    : {final_result['Precision']:.4f}"
)

print(
    f"Sensitivity  : {final_result['Sensitivity']:.4f}"
)

print(
    f"Specificity  : {final_result['Specificity']:.4f}"
)

print(
    f"F1 Score     : {final_result['F1']:.4f}"
)

print(
    f"ROC-AUC      : {final_result['ROC_AUC']:.4f}"
)

print(
    f"PR-AUC       : {final_result['PR_AUC']:.4f}"
)

print("\nGenerated files:")

print("  models/stroke_prediction_model.pkl")
print("  models/preprocessor.pkl")
print("  models/model_comparison.csv")
print("  models/final_metrics.csv")
print("  models/threshold_results.csv")
print("  models/confusion_matrix.png")
print("  models/roc_curve_comparison.png")
print("  models/precision_recall_curve.png")
print("  models/threshold_tuning.png")
print("  models/feature_importance.csv")
print("  models/feature_importance.png")

print("\n" + "=" * 70)
print("READY FOR STREAMLIT")
print("=" * 70)