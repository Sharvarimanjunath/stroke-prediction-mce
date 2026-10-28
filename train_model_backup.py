# ============================================================
# STROKE PREDICTION PROJECT
# UPDATED COMPLETE TRAINING CODE
# ============================================================

import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier
)

from sklearn.linear_model import LogisticRegression

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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")


# ============================================================
# 1. SETTINGS
# ============================================================

RANDOM_STATE = 42

DATA_PATH = os.path.join(
    "data",
    "healthcare-dataset-stroke-data.csv"
)

MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("STROKE PREDICTION PROJECT")
print("=" * 70)

print("\nLoading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\nDataset not found at:\n{DATA_PATH}\n\n"
        "Make sure your CSV file is inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)


# ============================================================
# 3. BASIC DATA INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("DATASET INFORMATION")
print("=" * 70)

print("\nColumns:")
print(list(df.columns))

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())


# ============================================================
# 4. REMOVE UNNECESSARY COLUMN
# ============================================================

# ID does not contain useful medical information.
if "id" in df.columns:
    df = df.drop(columns=["id"])

print("\nID column removed.")


# ============================================================
# 5. TARGET VARIABLE
# ============================================================

TARGET = "stroke"

if TARGET not in df.columns:
    raise ValueError(
        "Target column 'stroke' was not found in the dataset."
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]


# ============================================================
# 6. ORIGINAL STROKE DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("ORIGINAL STROKE DISTRIBUTION")
print("=" * 70)

print(y.value_counts())

print("\nOriginal stroke percentage:")

percentage = (
    y.value_counts(normalize=True) * 100
).round(2)

print(percentage)


# ============================================================
# 7. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

print("Training data shape:", X_train.shape)
print("Testing data shape :", X_test.shape)

print("\nTraining stroke distribution:")
print(y_train.value_counts())

print("\nTesting stroke distribution:")
print(y_test.value_counts())


# ============================================================
# 8. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y_train
)

print("\n" + "=" * 70)
print("TRAIN / VALIDATION / TEST SPLIT")
print("=" * 70)

print("Training data shape   :", X_train.shape)
print("Validation data shape :", X_val.shape)
print("Testing data shape    :", X_test.shape)

print("\nTraining stroke distribution:")
print(y_train.value_counts())

print("\nValidation stroke distribution:")
print(y_val.value_counts())

print("\nTesting stroke distribution:")
print(y_test.value_counts())


# ============================================================
# 9. FEATURE INFORMATION
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

# Keep only columns that actually exist.
categorical_features = [
    col for col in categorical_features
    if col in X.columns
]

numerical_features = [
    col for col in numerical_features
    if col in X.columns
]

print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# 10. PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("PREPROCESSING")
print("=" * 70)


# Numerical preprocessing
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


# Categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)


# ============================================================
# 11. PREPROCESS DATA
# ============================================================

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)


print("\nPreprocessing completed successfully!")

print(
    "Processed training shape   :",
    X_train_processed.shape
)

print(
    "Processed validation shape :",
    X_val_processed.shape
)

print(
    "Processed testing shape    :",
    X_test_processed.shape
)


# ============================================================
# 12. SMOTE BALANCING
# ============================================================

print("\n" + "=" * 70)
print("SMOTE BALANCING")
print("=" * 70)

print("\nBefore SMOTE:")
print(y_train.value_counts())


smote = SMOTE(
    sampling_strategy=1.0,
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ============================================================
# 13. EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    model_name,
    y_true,
    probabilities,
    threshold=0.50
):
    """
    Evaluate a binary classification model.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        y_true,
        predictions
    )

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

    cm = confusion_matrix(
        y_true,
        predictions
    )

    print("\n" + "-" * 70)
    print(model_name)
    print("-" * 70)

    print(f"Threshold : {threshold:.2f}")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"PR-AUC    : {pr_auc:.4f}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            predictions,
            target_names=[
                "No Stroke",
                "Stroke"
            ],
            zero_division=0
        )
    )

    return {
        "model": model_name,
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm
    }


# ============================================================
# 14. RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("RANDOM FOREST")
print("=" * 70)

rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

rf_model.fit(
    X_train_smote,
    y_train_smote
)

print("\nRandom Forest training completed!")


rf_val_prob = rf_model.predict_proba(
    X_val_processed
)[:, 1]

rf_test_prob = rf_model.predict_proba(
    X_test_processed
)[:, 1]


rf_result = evaluate_model(
    "RANDOM FOREST RESULTS",
    y_test,
    rf_test_prob,
    threshold=0.50
)


# ============================================================
# 15. LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("LOGISTIC REGRESSION")
print("=" * 70)

lr_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    C=0.5,
    solver="liblinear",
    random_state=RANDOM_STATE
)

lr_model.fit(
    X_train_smote,
    y_train_smote
)

print("\nLogistic Regression training completed!")


lr_val_prob = lr_model.predict_proba(
    X_val_processed
)[:, 1]

lr_test_prob = lr_model.predict_proba(
    X_test_processed
)[:, 1]


lr_result = evaluate_model(
    "LOGISTIC REGRESSION RESULTS",
    y_test,
    lr_test_prob,
    threshold=0.50
)


# ============================================================
# 16. EXTRA TREES
# ============================================================

print("\n" + "=" * 70)
print("EXTRA TREES CLASSIFIER")
print("=" * 70)

extra_model = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

extra_model.fit(
    X_train_smote,
    y_train_smote
)

print("\nExtra Trees training completed!")


extra_val_prob = extra_model.predict_proba(
    X_val_processed
)[:, 1]

extra_test_prob = extra_model.predict_proba(
    X_test_processed
)[:, 1]


extra_result = evaluate_model(
    "EXTRA TREES RESULTS",
    y_test,
    extra_test_prob,
    threshold=0.50
)


# ============================================================
# 17. IMPROVED BALANCED RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("IMPROVED BALANCED RANDOM FOREST")
print("=" * 70)

balanced_rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    class_weight={
        0: 1,
        1: 4
    },
    random_state=RANDOM_STATE,
    n_jobs=-1
)

balanced_rf.fit(
    X_train_smote,
    y_train_smote
)

print("\nBalanced Random Forest training completed!")


balanced_val_prob = balanced_rf.predict_proba(
    X_val_processed
)[:, 1]

balanced_test_prob = balanced_rf.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 18. THRESHOLD TUNING
# ============================================================

print("\n" + "=" * 70)
print("THRESHOLD TUNING USING VALIDATION DATA")
print("=" * 70)

thresholds = np.arange(
    0.10,
    0.91,
    0.05
)

threshold_results = []

for threshold in thresholds:

    validation_predictions = (
        balanced_val_prob >= threshold
    ).astype(int)

    precision = precision_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        validation_predictions,
        zero_division=0
    )

    threshold_results.append(
        {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    )

    print(
        f"Threshold: {threshold:.2f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"F1: {f1:.4f}"
    )


threshold_df = pd.DataFrame(
    threshold_results
)


# ============================================================
# 19. BEST THRESHOLD
# ============================================================

best_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]

best_threshold = float(
    best_row["threshold"]
)

best_validation_f1 = float(
    best_row["f1"]
)

print("\n" + "-" * 70)

print(
    f"Best Threshold: {best_threshold:.2f}"
)

print(
    f"Best Validation F1 Score: "
    f"{best_validation_f1:.4f}"
)


# ============================================================
# 20. FINAL BALANCED RANDOM FOREST RESULTS
# ============================================================

balanced_result = evaluate_model(
    "FINAL BALANCED RANDOM FOREST RESULTS",
    y_test,
    balanced_test_prob,
    threshold=best_threshold
)


# ============================================================
# 21. SAVE BEST MODEL
# ============================================================

model_package = {
    "model": balanced_rf,
    "preprocessor": preprocessor,
    "threshold": best_threshold,
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

print("\n" + "=" * 70)
print("MODEL SAVING")
print("=" * 70)

print(
    f"\nModel saved successfully at:\n"
    f"{model_path}"
)


# ============================================================
# 22. MODEL COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame(
    [
        rf_result,
        lr_result,
        extra_result,
        balanced_result
    ]
)

comparison_display = comparison[
    [
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "pr_auc"
    ]
].copy()

print(
    comparison_display.to_string(
        index=False
    )
)


# Save comparison
comparison_path = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

comparison_display.to_csv(
    comparison_path,
    index=False
)


# ============================================================
# 23. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

cm = confusion_matrix(
    y_test,
    (
        balanced_test_prob >= best_threshold
    ).astype(int)
)

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "Confusion Matrix - Improved Balanced Random Forest"
)

plt.colorbar()

classes = [
    "No Stroke",
    "Stroke"
]

plt.xticks(
    [0, 1],
    classes
)

plt.yticks(
    [0, 1],
    classes
)

plt.xlabel("Predicted")
plt.ylabel("Actual")


for i in range(cm.shape[0]):

    for j in range(cm.shape[1]):

        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
            fontsize=14
        )


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

plt.show()


# ============================================================
# 24. ROC CURVE COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("ROC CURVE COMPARISON")
print("=" * 70)


rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_test_prob
)

lr_fpr, lr_tpr, _ = roc_curve(
    y_test,
    lr_test_prob
)

extra_fpr, extra_tpr, _ = roc_curve(
    y_test,
    extra_test_prob
)

balanced_fpr, balanced_tpr, _ = roc_curve(
    y_test,
    balanced_test_prob
)


rf_auc = roc_auc_score(
    y_test,
    rf_test_prob
)

lr_auc = roc_auc_score(
    y_test,
    lr_test_prob
)

extra_auc = roc_auc_score(
    y_test,
    extra_test_prob
)

balanced_auc = roc_auc_score(
    y_test,
    balanced_test_prob
)


plt.figure(figsize=(9, 7))

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest (AUC = {rf_auc:.4f})"
)

plt.plot(
    lr_fpr,
    lr_tpr,
    label=f"Logistic Regression (AUC = {lr_auc:.4f})"
)

plt.plot(
    extra_fpr,
    extra_tpr,
    label=f"Extra Trees (AUC = {extra_auc:.4f})"
)

plt.plot(
    balanced_fpr,
    balanced_tpr,
    label=f"Balanced Random Forest (AUC = {balanced_auc:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    "r--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve Comparison"
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

plt.show()


# ============================================================
# 25. PRECISION-RECALL CURVE
# ============================================================

print("\n" + "=" * 70)
print("PRECISION-RECALL CURVE")
print("=" * 70)


precision_curve, recall_curve, _ = (
    precision_recall_curve(
        y_test,
        balanced_test_prob
    )
)

pr_auc = average_precision_score(
    y_test,
    balanced_test_prob
)


plt.figure(figsize=(9, 7))

plt.plot(
    recall_curve,
    precision_curve,
    label=f"Balanced Random Forest (PR-AUC = {pr_auc:.4f})"
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

plt.show()


# ============================================================
# 26. THRESHOLD VS F1
# ============================================================

plt.figure(figsize=(9, 6))

plt.plot(
    threshold_df["threshold"],
    threshold_df["f1"],
    marker="o",
    label="F1 Score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best Threshold = {best_threshold:.2f}"
)

plt.xlabel("Decision Threshold")

plt.ylabel("F1 Score")

plt.title(
    "Threshold Tuning - Validation Data"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()

threshold_path = os.path.join(
    MODEL_DIR,
    "threshold_tuning.png"
)

plt.savefig(
    threshold_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 27. FEATURE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


try:

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        balanced_rf
        .feature_importances_
    )

    feature_importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances
        }
    )

    feature_importance_df = (
        feature_importance_df
        .sort_values(
            "importance",
            ascending=False
        )
    )

    print(
        "\nTop 15 important features:"
    )

    print(
        feature_importance_df
        .head(15)
        .to_string(index=False)
    )

    # Save feature importance
    feature_importance_path = os.path.join(
        MODEL_DIR,
        "feature_importance.csv"
    )

    feature_importance_df.to_csv(
        feature_importance_path,
        index=False
    )

    # Plot top 15
    top_features = (
        feature_importance_df
        .head(15)
        .sort_values(
            "importance"
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        top_features["feature"],
        top_features["importance"]
    )

    plt.xlabel(
        "Importance"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Top 15 Feature Importances"
    )

    plt.tight_layout()

    feature_plot_path = os.path.join(
        MODEL_DIR,
        "feature_importance.png"
    )

    plt.savefig(
        feature_plot_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

except Exception as e:

    print(
        "\nFeature importance plot could not be generated:"
    )

    print(e)


# ============================================================
# 28. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL PROJECT SUMMARY")
print("=" * 70)

print(
    f"\nBest Model:"
    f" Improved Balanced Random Forest"
)

print(
    f"Best Threshold:"
    f" {best_threshold:.2f}"
)

print(
    f"Test Accuracy:"
    f" {balanced_result['accuracy']:.4f}"
)

print(
    f"Test Precision:"
    f" {balanced_result['precision']:.4f}"
)

print(
    f"Test Recall:"
    f" {balanced_result['recall']:.4f}"
)

print(
    f"Test F1 Score:"
    f" {balanced_result['f1']:.4f}"
)

print(
    f"Test ROC-AUC:"
    f" {balanced_result['roc_auc']:.4f}"
)

print(
    f"Test PR-AUC:"
    f" {balanced_result['pr_auc']:.4f}"
)

print("\nFiles saved in models folder:")

print("1. stroke_prediction_model.pkl")
print("2. model_comparison.csv")
print("3. confusion_matrix.png")
print("4. roc_curve_comparison.png")
print("5. precision_recall_curve.png")
print("6. threshold_tuning.png")
print("7. feature_importance.csv")
print("8. feature_importance.png")

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)