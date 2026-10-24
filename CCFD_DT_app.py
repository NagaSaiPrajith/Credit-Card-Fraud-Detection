import os
import io
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)

warnings.filterwarnings("ignore", category=UserWarning)

st.set_page_config(
    page_title="Credit Card Fraud | Decision Tree Playground",
    page_icon="🌳",
    layout="wide",
)

st.title("🌳 Credit Card Fraud Detection")
st.caption("Interactive Decision Tree Playground — tune hyperparameters, train, and inspect model performance.")

with st.sidebar:
    st.header("Dataset")
    uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])
    st.caption("Expected target column: `Class` (0 = legitimate, 1 = fraud).")
    st.divider()
    st.header("Model settings")
    criterion = st.selectbox("Criterion", ["gini", "entropy", "log_loss"], index=0)
    unlimited_depth = st.checkbox("Unlimited max depth (None)", value=False)
    max_depth = st.slider(
        "Max depth", min_value=1, max_value=100, value=6,
        disabled=unlimited_depth,
        help="Ignored when unlimited depth is enabled.",
    )
    min_samples_split = st.slider(
        "Min samples split", min_value=2, max_value=500, value=10,
        help="Scikit-learn requires at least 2.",
    )
    min_samples_leaf = st.slider(
        "Min samples leaf", min_value=1, max_value=500, value=5,
    )
    st.divider()
    st.header("Imbalance handling")
    balancing = st.selectbox(
        "Training strategy",
        ["None", "Class weights (balanced)", "SMOTE"],
        index=0,
        help="Balancing is applied to training data only. Test data remains untouched.",
    )
    threshold = st.slider(
        "Fraud probability threshold",
        min_value=0.01, max_value=0.99, value=0.50, step=0.01,
        help="Predict fraud when predicted fraud probability is at least this threshold.",
    )
    st.divider()
    st.header("Evaluation")
    test_size = st.slider(
        "Test set proportion", min_value=0.10, max_value=0.40, value=0.20, step=0.05
    )
    tree_preview_depth = st.slider(
        "Tree preview depth", min_value=1, max_value=5, value=5,
        help="Only this many levels are displayed in the visualization.",
    )
    run_model = st.button("🚀 Run model", type="primary", use_container_width=True)

@st.cache_data(show_spinner=False)
def load_data(file_bytes):
    return pd.read_csv(io.BytesIO(file_bytes))

if uploaded_file is not None:
    try:
        df = load_data(uploaded_file.getvalue())
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        st.stop()
else:
    default_path = "creditcard.csv"
    if os.path.exists(default_path):
        try:
            df = pd.read_csv(default_path)
        except Exception as exc:
            st.error(f"Could not read {default_path}: {exc}")
            st.stop()
    else:
        st.info(
            "Upload your `creditcard.csv` in the sidebar to get started. "
            "The app also automatically loads `creditcard.csv` from the project folder if present."
        )
        st.markdown(
            """
            **Expected dataset format**
            - One row per transaction
            - A column named `Class`
            - `Class = 0`: legitimate transaction
            - `Class = 1`: fraudulent transaction
            """
        )
        st.stop()

if "Class" not in df.columns:
    st.error("The dataset must contain a `Class` target column (0 = legitimate, 1 = fraud).")
    st.stop()

if df.empty:
    st.error("The uploaded dataset has no rows.")
    st.stop()

if df["Class"].isna().any():
    st.error("The `Class` column contains missing values. Please clean them before training.")
    st.stop()

classes = set(pd.Series(df["Class"]).unique().tolist())
if not classes.issubset({0, 1, 0.0, 1.0}) or len(classes) != 2:
    st.error("The `Class` column must contain both classes encoded as 0 and 1.")
    st.stop()

# Convert target to integer and clean feature columns safely.
y = df["Class"].astype(int)
X = df.drop(columns=["Class"]).copy()
# Keep numeric columns; this standard dataset is fully numeric.
non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
if non_numeric:
    st.warning(f"Dropping non-numeric feature columns: {', '.join(map(str, non_numeric))}")
    X = X.select_dtypes(include=[np.number])
if X.shape[1] == 0:
    st.error("No numeric feature columns were found.")
    st.stop()

# Handle missing/infinite feature values with training-derived medians after split.
X = X.replace([np.inf, -np.inf], np.nan)
if y.value_counts().min() < 2:
    st.error("Each class needs at least two rows for a stratified train/test split.")
    st.stop()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Transactions", f"{len(df):,}")
col_b.metric("Fraud cases", f"{int((y == 1).sum()):,}")
col_c.metric("Fraud prevalence", f"{100 * (y == 1).mean():.3f}%")
st.caption(f"Features used: {X.shape[1]} | Target: `Class`")

# Fixed split for fair comparisons; changing test proportion intentionally changes split.
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
except ValueError as exc:
    st.error(f"Could not split the dataset: {exc}")
    st.stop()

# Impute missing values using training-set medians only.
medians = X_train.median()
X_train = X_train.fillna(medians).fillna(0)
X_test = X_test.fillna(medians).fillna(0)

if run_model:
    try:
        with st.spinner("Training and evaluating the decision tree..."):
            X_fit, y_fit = X_train, y_train
            class_weight = None

            if balancing == "Class weights (balanced)":
                class_weight = "balanced"
            elif balancing == "SMOTE":
                try:
                    from imblearn.over_sampling import SMOTE
                except ImportError:
                    st.error("SMOTE requires imbalanced-learn. Install it with: pip install imbalanced-learn")
                    st.stop()
                # Avoid requesting more neighbors than available minority examples.
                minority_count = int(y_train.value_counts().min())
                if minority_count < 2:
                    st.error("SMOTE needs at least two minority-class training examples.")
                    st.stop()
                smote = SMOTE(random_state=42, k_neighbors=min(5, minority_count - 1))
                X_fit, y_fit = smote.fit_resample(X_train, y_train)

            model = DecisionTreeClassifier(
                criterion=criterion,
                max_depth=None if unlimited_depth else max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=42,
            )
            model.fit(X_fit, y_fit)

            y_prob = model.predict_proba(X_test)[:, 1]
            # Find the probability column corresponding to class 1 explicitly.
            fraud_col = list(model.classes_).index(1)
            y_prob = model.predict_proba(X_test)[:, fraud_col]
            y_pred = (y_prob >= threshold).astype(int)

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            avg_precision = average_precision_score(y_test, y_prob)
            cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        st.session_state["results"] = {
            "model": model,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_precision": avg_precision,
            "cm": cm,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "X_test": X_test,
            "features": list(X.columns),
            "params": {
                "criterion": criterion,
                "max_depth": None if unlimited_depth else max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "balancing": balancing,
                "threshold": threshold,
            },
            "tree_preview_depth": tree_preview_depth,
        }
        st.success("Model trained and evaluated successfully.")
    except Exception as exc:
        st.error(f"Training failed: {exc}")

results = st.session_state.get("results")

if results is not None:
    st.divider()
    st.header("📊 Evaluation results")
    params = results["params"]
    st.caption(
        f"Last run: criterion={params['criterion']}, max_depth={params['max_depth']}, "
        f"min_samples_split={params['min_samples_split']}, "
        f"min_samples_leaf={params['min_samples_leaf']}, "
        f"strategy={params['balancing']}, threshold={params['threshold']:.2f}"
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{results['accuracy']:.2%}")
    m2.metric("Precision (fraud)", f"{results['precision']:.2%}")
    m3.metric("Recall (fraud)", f"{results['recall']:.2%}")
    m4.metric("F1-score", f"{results['f1']:.2%}")
    m5.metric("Average precision", f"{results['avg_precision']:.2%}")

    st.info(
        "For fraud detection, recall measures the share of actual fraud caught; "
        "precision measures the share of fraud alerts that are truly fraudulent. "
        "Accuracy can be misleading when fraud is rare."
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader("Confusion matrix")
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=results["cm"],
            display_labels=["Legitimate (0)", "Fraud (1)"],
        )
        disp.plot(ax=ax_cm, cmap="Blues", values_format="d", colorbar=False)
        ax_cm.set_title("Actual vs predicted")
        fig_cm.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    with right:
        st.subheader("Classification report")
        report = classification_report(
            results["y_test"], results["y_pred"],
            labels=[0, 1], target_names=["Legitimate", "Fraud"],
            output_dict=True, zero_division=0
        )
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.style.format(precision=3), use_container_width=True)

    st.divider()
    st.header("🌳 Decision tree preview")
    st.caption(
        f"Showing at most {results['tree_preview_depth']} levels. "
        "The ellipsis marks branches hidden beyond the selected preview depth."
    )
    fig_tree, ax_tree = plt.subplots(figsize=(22, 10))
    plot_tree(
        results["model"],
        feature_names=results["features"],
        class_names=["Legitimate", "Fraud"],
        filled=True,
        rounded=True,
        max_depth=results["tree_preview_depth"],
        impurity=True,
        proportion=False,
        fontsize=8,
        ax=ax_tree,
    )
    fig_tree.tight_layout()
    st.pyplot(fig_tree)
    plt.close(fig_tree)

    with st.expander("Show current model parameters and tree size"):
        st.json({
            **results["params"],
            "tree_node_count": int(results["model"].tree_.node_count),
            "tree_depth": int(results["model"].get_depth()),
            "leaves": int(results["model"].get_n_leaves()),
            "training_rows_after_balancing": int(len(results["y_test"]) if params["balancing"] == "None" else 0),
        })

    csv_report = pd.DataFrame(
        classification_report(
            results["y_test"], results["y_pred"],
            labels=[0, 1], target_names=["Legitimate", "Fraud"],
            output_dict=True, zero_division=0
        )
    ).transpose().to_csv().encode("utf-8")
    st.download_button(
        "Download classification report (CSV)",
        data=csv_report,
        file_name="fraud_classification_report.csv",
        mime="text/csv",
    )
else:
    st.markdown("### Ready when you are")
    st.write("Choose your settings in the sidebar, then click **Run model**.")
    st.write("The app keeps the test set untouched when applying class weighting or SMOTE.")
