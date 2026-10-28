import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Stroke Prediction System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "stroke_prediction_model.pkl"
)

COMPARISON_PATH = os.path.join(
    MODEL_DIR,
    "model_comparison.csv"
)

FINAL_METRICS_PATH = os.path.join(
    MODEL_DIR,
    "final_metrics.csv"
)

THRESHOLD_RESULTS_PATH = os.path.join(
    MODEL_DIR,
    "threshold_results.csv"
)

CONFUSION_MATRIX_PATH = os.path.join(
    MODEL_DIR,
    "confusion_matrix.png"
)

ROC_PATH = os.path.join(
    MODEL_DIR,
    "roc_curve_comparison.png"
)

PR_PATH = os.path.join(
    MODEL_DIR,
    "precision_recall_curve.png"
)

THRESHOLD_PATH = os.path.join(
    MODEL_DIR,
    "threshold_tuning.png"
)

FEATURE_IMPORTANCE_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.png"
)

FEATURE_CSV_PATH = os.path.join(
    MODEL_DIR,
    "feature_importance.csv"
)


# ============================================================
# LOAD MODEL
# ============================================================

try:

    saved_model = joblib.load(
        MODEL_PATH
    )

except FileNotFoundError:

    st.error(
        "❌ Model file not found.\n\n"
        "Please make sure that "
        "'stroke_prediction_model.pkl' "
        "is inside the models folder."
    )

    st.stop()

except Exception as e:

    st.error(
        f"❌ Error loading model: {e}"
    )

    st.stop()


# ============================================================
# EXTRACT MODEL COMPONENTS
# ============================================================

if isinstance(saved_model, dict):

    model = saved_model.get(
        "model"
    )

    preprocessor = saved_model.get(
        "preprocessor"
    )

    threshold = saved_model.get(
        "threshold",
        0.50
    )

    feature_columns = saved_model.get(
        "features"
    )

else:

    model = saved_model

    preprocessor = None

    threshold = 0.50

    feature_columns = None


if model is None:

    st.error(
        "❌ Valid machine learning model "
        "was not found inside the pickle file."
    )

    st.stop()


# ============================================================
# LOAD FINAL MODEL METRICS
# ============================================================

final_metrics = None

if os.path.exists(FINAL_METRICS_PATH):

    try:

        metrics_df = pd.read_csv(
            FINAL_METRICS_PATH
        )

        if not metrics_df.empty:

            final_metrics = metrics_df.iloc[0]

    except Exception as e:

        st.warning(
            f"⚠️ Could not load final metrics: {e}"
        )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        # 🧠 Stroke Prediction

        **Purpose:** Predict potential stroke
        risk based on patient health and
        demographic information.
        """
    )

    st.markdown("---")

    st.subheader(
        "🎯 Decision Rule"
    )

    st.write(
        "The optimized decision threshold is:"
    )

    st.metric(
        "Threshold",
        f"{threshold:.2f}"
    )

    st.write(
        f"""
        If the predicted probability is greater
        than or equal to **{threshold:.2f}**, the
        system classifies the patient as having
        predicted stroke risk.
        """
    )

    st.markdown("---")

    st.subheader(
        "🤖 Final Model"
    )

    st.write(
        "Balanced Random Forest"
    )

    st.markdown("---")

    st.caption(
        "For academic and research purposes only."
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🧠 Stroke Prediction System"
)

st.subheader(
    "Machine Learning Based Stroke Risk Prediction"
)

st.write(
    """
    Enter the patient's information below to
    estimate the predicted stroke risk using
    the trained Balanced Random Forest model.
    """
)

st.info(
    "⚠️ This application is a machine-learning "
    "research prototype and is not a medical diagnosis."
)


# ============================================================
# PATIENT INPUT
# ============================================================

st.header(
    "👤 Patient Information"
)

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=0.0,
        max_value=120.0,
        value=50.0,
        step=1.0
    )

    gender = st.selectbox(
        "Gender",
        [
            "Male",
            "Female",
            "Other"
        ]
    )

    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    heart_disease = st.selectbox(
        "Heart Disease",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

    ever_married = st.selectbox(
        "Ever Married",
        [
            "Yes",
            "No"
        ]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    avg_glucose_level = st.number_input(
        "Average Glucose Level",
        min_value=0.0,
        max_value=500.0,
        value=100.0,
        step=0.1
    )

    bmi = st.number_input(
        "BMI",
        min_value=5.0,
        max_value=100.0,
        value=25.0,
        step=0.1
    )

    work_type = st.selectbox(
        "Work Type",
        [
            "Private",
            "Self-employed",
            "Govt_job",
            "children",
            "Never_worked"
        ]
    )

    residence_type = st.selectbox(
        "Residence Type",
        [
            "Urban",
            "Rural"
        ]
    )

    smoking_status = st.selectbox(
        "Smoking Status",
        [
            "formerly smoked",
            "never smoked",
            "smokes",
            "Unknown"
        ]
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.markdown("---")

predict_button = st.button(
    "🔍 Predict Stroke Risk",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame({

        "gender": [gender],

        "age": [age],

        "hypertension": [
            hypertension
        ],

        "heart_disease": [
            heart_disease
        ],

        "ever_married": [
            ever_married
        ],

        "work_type": [
            work_type
        ],

        "Residence_type": [
            residence_type
        ],

        "avg_glucose_level": [
            avg_glucose_level
        ],

        "bmi": [
            bmi
        ],

        "smoking_status": [
            smoking_status
        ]
    })


    try:

        # ----------------------------------------------------
        # PREPROCESSING
        # ----------------------------------------------------

        if preprocessor is not None:

            processed_data = (
                preprocessor.transform(
                    input_data
                )
            )

        else:

            processed_data = input_data


        # ----------------------------------------------------
        # PREDICT PROBABILITY
        # ----------------------------------------------------

        if hasattr(
            model,
            "predict_proba"
        ):

            probability = (
                model.predict_proba(
                    processed_data
                )[0][1]
            )

        else:

            prediction_value = (
                model.predict(
                    processed_data
                )[0]
            )

            probability = float(
                prediction_value
            )


        # ----------------------------------------------------
        # APPLY OPTIMIZED THRESHOLD
        # ----------------------------------------------------

        prediction = int(
            probability >= threshold
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.markdown("---")

        st.header(
            "📊 Prediction Result"
        )

        result_col1, result_col2 = (
            st.columns(2)
        )

        with result_col1:

            st.metric(
                "Stroke Probability",
                f"{probability * 100:.2f}%"
            )

        with result_col2:

            st.metric(
                "Decision Threshold",
                f"{threshold:.2f}"
            )


        # ----------------------------------------------------
        # PROBABILITY INTERPRETATION
        # ----------------------------------------------------

        if prediction == 1:

            st.error(
                "⚠️ Predicted Stroke Risk"
            )

            st.warning(
                f"""
                The model estimates a stroke
                probability of
                **{probability * 100:.2f}%**.

                This probability is above the
                selected decision threshold of
                **{threshold:.2f}**.
                """
            )

        else:

            st.success(
                "✅ No Stroke Predicted"
            )

            st.info(
                f"""
                The model estimates a stroke
                probability of
                **{probability * 100:.2f}%**.

                This probability is below the
                selected decision threshold of
                **{threshold:.2f}**.
                """
            )


        # ====================================================
        # PATIENT SUMMARY
        # ====================================================

        st.markdown("---")

        st.subheader(
            "👤 Patient Information"
        )

        display_data = pd.DataFrame({

            "Parameter": [

                "Age",
                "Gender",
                "Hypertension",
                "Heart Disease",
                "Ever Married",
                "Work Type",
                "Residence Type",
                "Average Glucose Level",
                "BMI",
                "Smoking Status"

            ],

            "Value": [

                age,

                gender,

                "Yes"
                if hypertension == 1
                else "No",

                "Yes"
                if heart_disease == 1
                else "No",

                ever_married,

                work_type,

                residence_type,

                avg_glucose_level,

                bmi,

                smoking_status
            ]
        })

        st.table(
            display_data
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed."
        )

        st.exception(e)


# ============================================================
# MODEL PERFORMANCE SECTION
# ============================================================

st.markdown("---")

st.header(
    "📈 Model Performance"
)

st.write(
    """
    The following metrics represent the performance
    of the final Balanced Random Forest model on
    the independent test set.
    """
)


# ============================================================
# LOAD MODEL COMPARISON
# ============================================================

comparison_df = None

if os.path.exists(
    COMPARISON_PATH
):

    try:

        comparison_df = pd.read_csv(
            COMPARISON_PATH
        )

    except Exception as e:

        st.warning(
            f"⚠️ Could not load model comparison: {e}"
        )


# ============================================================
# MODEL COMPARISON TABLE
# ============================================================

if comparison_df is not None:

    st.subheader(
        "🤖 Machine Learning Model Comparison"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# FINAL MODEL METRICS
# ============================================================

st.subheader(
    "🎯 Final Balanced Random Forest Performance"
)


if final_metrics is not None:

    # --------------------------------------------------------
    # FIRST ROW
    # --------------------------------------------------------

    metric1, metric2, metric3 = (
        st.columns(3)
    )

    with metric1:

        st.metric(
            "Accuracy",
            f"{final_metrics['Accuracy'] * 100:.2f}%"
        )

    with metric2:

        st.metric(
            "Precision",
            f"{final_metrics['Precision'] * 100:.2f}%"
        )

    with metric3:

        st.metric(
            "Sensitivity",
            f"{final_metrics['Sensitivity'] * 100:.2f}%"
        )


    # --------------------------------------------------------
    # SECOND ROW
    # --------------------------------------------------------

    metric4, metric5, metric6 = (
        st.columns(3)
    )

    with metric4:

        st.metric(
            "Specificity",
            f"{final_metrics['Specificity'] * 100:.2f}%"
        )

    with metric5:

        st.metric(
            "F1 Score",
            f"{final_metrics['F1'] * 100:.2f}%"
        )

    with metric6:

        st.metric(
            "ROC-AUC",
            f"{final_metrics['ROC_AUC']:.4f}"
        )


    # --------------------------------------------------------
    # THIRD ROW
    # --------------------------------------------------------

    metric7, metric8 = (
        st.columns(2)
    )

    with metric7:

        st.metric(
            "PR-AUC",
            f"{final_metrics['PR_AUC']:.4f}"
        )

    with metric8:

        st.metric(
            "Decision Threshold",
            f"{threshold:.2f}"
        )


else:

    st.warning(
        "⚠️ Final metrics file was not found. "
        "Please run train_model.py first."
    )


# ============================================================
# FINAL PERFORMANCE SUMMARY
# ============================================================

if final_metrics is not None:

    st.markdown("---")

    st.subheader(
        "📋 Final Test Set Summary"
    )

    summary_data = pd.DataFrame({

        "Metric": [
            "Accuracy",
            "Precision",
            "Sensitivity",
            "Specificity",
            "F1 Score",
            "ROC-AUC",
            "PR-AUC",
            "Decision Threshold"
        ],

        "Value": [

            f"{final_metrics['Accuracy'] * 100:.2f}%",

            f"{final_metrics['Precision'] * 100:.2f}%",

            f"{final_metrics['Sensitivity'] * 100:.2f}%",

            f"{final_metrics['Specificity'] * 100:.2f}%",

            f"{final_metrics['F1'] * 100:.2f}%",

            f"{final_metrics['ROC_AUC']:.4f}",

            f"{final_metrics['PR_AUC']:.4f}",

            f"{threshold:.2f}"
        ]
    })

    st.dataframe(
        summary_data,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# DECISION THRESHOLD EXPLANATION
# ============================================================

st.markdown("---")

st.header(
    "🎯 Decision Threshold Explanation"
)

st.info(
    f"""
    The final decision threshold used by the
    application is **{threshold:.2f}**.

    The threshold was selected using the
    validation dataset based on the F1-score.

    If the predicted stroke probability is
    **greater than or equal to {threshold:.2f}**,
    the system displays **Predicted Stroke Risk**.

    If the predicted probability is **less than
    {threshold:.2f}**, the system displays
    **No Stroke Predicted**.
    """
)


# ============================================================
# LOAD THRESHOLD RESULTS
# ============================================================

threshold_df = None

if os.path.exists(
    THRESHOLD_RESULTS_PATH
):

    try:

        threshold_df = pd.read_csv(
            THRESHOLD_RESULTS_PATH
        )

    except Exception:

        threshold_df = None


# ============================================================
# THRESHOLD TABLE
# ============================================================

if threshold_df is not None:

    st.subheader(
        "📊 Threshold Evaluation"
    )

    display_threshold_df = threshold_df.copy()

    display_threshold_df["Selected"] = (
        np.isclose(
            display_threshold_df["Threshold"],
            threshold
        )
    )

    st.dataframe(
        display_threshold_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# THRESHOLD EXAMPLES
# ============================================================

st.subheader(
    "💡 Decision Examples"
)

example_probabilities = [
    0.30,
    0.50,
    threshold - 0.01,
    threshold,
    0.80
]

example_predictions = []

for probability_value in example_probabilities:

    if probability_value >= threshold:

        example_predictions.append(
            "Predicted Stroke Risk"
        )

    else:

        example_predictions.append(
            "No Stroke Predicted"
        )


example_data = pd.DataFrame({

    "Stroke Probability": [
        f"{value * 100:.0f}%"
        for value in example_probabilities
    ],

    "Prediction": example_predictions
})

st.table(
    example_data
)


# ============================================================
# GRAPH SECTION
# ============================================================

st.markdown("---")

st.header(
    "📊 Model Evaluation Graphs"
)

st.write(
    """
    The following graphs were generated during
    model training and final evaluation.
    """
)


# ============================================================
# GRAPH DISPLAY FUNCTION
# ============================================================

def display_graph(
    title,
    image_path,
    description
):

    st.subheader(
        title
    )

    st.caption(
        description
    )

    if os.path.exists(
        image_path
    ):

        left, center, right = st.columns(
            [1, 2, 1]
        )

        with center:

            st.image(
                image_path,
                use_container_width=True
            )

    else:

        st.warning(
            f"Graph file not found: "
            f"{os.path.basename(image_path)}"
        )


# ============================================================
# 1. CONFUSION MATRIX
# ============================================================

display_graph(

    "1️⃣ Confusion Matrix",

    CONFUSION_MATRIX_PATH,

    """
    The confusion matrix shows the number of
    correct and incorrect predictions for
    stroke and no-stroke classes using the
    final decision threshold.
    """
)


# ============================================================
# 2. ROC CURVE
# ============================================================

display_graph(

    "2️⃣ ROC Curve Comparison",

    ROC_PATH,

    """
    ROC curves compare the ability of the
    evaluated machine-learning models to
    distinguish between stroke and no-stroke
    cases.
    """
)


# ============================================================
# 3. PRECISION-RECALL CURVE
# ============================================================

display_graph(

    "3️⃣ Precision-Recall Curve",

    PR_PATH,

    """
    The Precision-Recall curve is particularly
    useful for evaluating classification
    performance on an imbalanced dataset.
    """
)


# ============================================================
# 4. THRESHOLD TUNING
# ============================================================

display_graph(

    "4️⃣ Threshold Tuning",

    THRESHOLD_PATH,

    f"""
    This graph shows how Precision, Recall and
    F1-score change across different decision
    thresholds. The selected threshold is
    **{threshold:.2f}**.
    """
)


# ============================================================
# 5. FEATURE IMPORTANCE
# ============================================================

display_graph(

    "5️⃣ Feature Importance",

    FEATURE_IMPORTANCE_PATH,

    """
    Feature importance shows the relative
    contribution of the processed input features
    to the Balanced Random Forest model.
    """
)


# ============================================================
# FEATURE IMPORTANCE TABLE
# ============================================================

if os.path.exists(
    FEATURE_CSV_PATH
):

    st.subheader(
        "🔎 Top Important Features"
    )

    try:

        feature_df = pd.read_csv(
            FEATURE_CSV_PATH
        )

        st.dataframe(
            feature_df.head(15),
            use_container_width=True,
            hide_index=True
        )

    except Exception as e:

        st.warning(
            f"Could not load feature importance data: {e}"
        )



# ============================================================
# DISCLAIMER
# ============================================================

st.markdown("---")

st.warning(
    """
    ⚠️ Important Disclaimer

    This application is developed for academic
    and research purposes. The prediction generated
    by this system should not be considered a
    medical diagnosis or a substitute for professional
    medical advice.
    """
)


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    **Stroke Prediction Using Machine Learning**

    Developed as an academic machine-learning project.
    """
)