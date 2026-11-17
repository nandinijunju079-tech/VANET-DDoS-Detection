
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="VANET DDoS Detection Dashboard",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "real"

st.title("🚗 VANET DDoS Attack Detection and Mitigation")
st.caption(
    "Real-data dashboard | VeReMi NextGen DoS scenario | "
    "Random Forest, XGBoost, SHAP, LIME and EWMA"
)

# ---------------------------------------
# LOAD DATA
# ---------------------------------------

FILES = {
    "models": "receiver_model_evaluation.csv",
    "shap": "receiver_shap_importance.csv",
    "confidence": "receiver_confidence_results.csv",
    "ewma": "receiver_ewma_test.csv",
    "severity": "receiver_severity_mitigation.csv",
    "comparison": "receiver_detection_comparison.csv"
}

missing = [
    name for name, filename in FILES.items()
    if not (OUTPUT_DIR / filename).exists()
]

if missing:
    st.error("Missing real-data output files: " + ", ".join(missing))
    st.stop()


@st.cache_data
def load_csv(filename):
    return pd.read_csv(OUTPUT_DIR / filename)


models = load_csv(FILES["models"])
shap_data = load_csv(FILES["shap"])
confidence = load_csv(FILES["confidence"])
ewma = load_csv(FILES["ewma"])
severity = load_csv(FILES["severity"])
comparison = load_csv(FILES["comparison"])

# Load only Test metrics for the main model comparison
test_models = models[
    models["Split"].str.lower() == "test"
].copy()

# ---------------------------------------
# SIDEBAR
# ---------------------------------------

st.sidebar.title("Dashboard Navigation")

section = st.sidebar.radio(
    "Choose a section",
    [
        "Overview",
        "Model Performance",
        "Confidence Detection",
        "Explainable AI",
        "EWMA Monitoring",
        "Severity and Mitigation",
        "Detection Comparison",
        "Explore Test Records"
    ]
)

st.sidebar.divider()
st.sidebar.info(
    "Dataset: VeReMi NextGen DoS scenario\n\n"
    "Split: Official Test set\n\n"
    "Total observations: 60,718"
)

# ---------------------------------------
# OVERVIEW
# ---------------------------------------

if section == "Overview":

    st.header("Network Traffic Overview")

    total = len(severity)

    normal = (
        confidence["Confidence_Classification"] == "Normal"
    ).sum()

    suspicious = (
        confidence["Confidence_Classification"] == "Suspicious"
    ).sum()

    dos = (
        confidence["Confidence_Classification"] == "DoS"
    ).sum()

    allow = (
        severity["Mitigation_Action"] == "ALLOW"
    ).sum()

    monitor = (
        severity["Mitigation_Action"] == "MONITOR"
    ).sum()

    filter_count = (
        severity["Mitigation_Action"] == "FILTER"
    ).sum()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Observations", f"{total:,}")
    c2.metric("Normal", f"{normal:,}")
    c3.metric("Suspicious", f"{suspicious:,}")
    c4.metric("DoS", f"{dos:,}")

    st.divider()

    st.subheader("Mitigation Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric("Allow", f"{allow:,}")
    c2.metric("Monitor", f"{monitor:,}")
    c3.metric("Filter", f"{filter_count:,}")

    st.divider()

    st.subheader("Confidence Classification")

    confidence_counts = (
        confidence["Confidence_Classification"]
        .value_counts()
        .reindex(["Normal", "Suspicious", "DoS"], fill_value=0)
    )

    st.bar_chart(confidence_counts)

    st.subheader("Mitigation Distribution")

    mitigation_counts = (
        severity["Mitigation_Action"].value_counts()
        .reindex(["ALLOW", "MONITOR", "FILTER"], fill_value=0)
    )

    st.bar_chart(mitigation_counts)

    st.info(
        "These results represent receiver-message observations "
        "from the dataset's DoS scenario. They are not live network "
        "packet captures."
    )

# ---------------------------------------
# MODEL PERFORMANCE
# ---------------------------------------

elif section == "Model Performance":

    st.header("Random Forest vs XGBoost")

    if test_models.empty:
        st.warning("Test model results were not found.")
        st.stop()

    for model_name in ["Random Forest", "XGBoost"]:

        selected = test_models[
            test_models["Model"] == model_name
        ]

        if selected.empty:
            continue

        result = selected.iloc[0]

        st.subheader(model_name)

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Accuracy",
            f"{result['Accuracy'] * 100:.2f}%"
        )

        c2.metric(
            "Precision",
            f"{result['Precision'] * 100:.2f}%"
        )

        c3.metric(
            "Recall",
            f"{result['Recall'] * 100:.2f}%"
        )

        c4.metric(
            "F1 Score",
            f"{result['F1'] * 100:.2f}%"
        )

        st.caption(
            f"FPR: {result['FPR'] * 100:.3f}% | "
            f"Prediction time: {result['Prediction_Time_Seconds']:.4f} sec | "
            f"Average latency: {result['Average_Latency_ms']:.6f} ms/record"
        )

        st.divider()

    st.subheader("Test Model Comparison")

    metric_columns = [
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FPR",
        "Prediction_Time_Seconds",
        "Average_Latency_ms"
    ]

    display_models = test_models[metric_columns].copy()

    for col in ["Accuracy", "Precision", "Recall", "F1", "FPR"]:
        display_models[col] = display_models[col] * 100

    st.dataframe(
        display_models,
        use_container_width=True,
        hide_index=True
    )

# ---------------------------------------
# CONFIDENCE DETECTION
# ---------------------------------------

elif section == "Confidence Detection":

    st.header("Confidence-Aware Detection")

    st.write(
        "The Random Forest model assigns a DoS probability "
        "to each receiver-message observation."
    )

    c1, c2, c3 = st.columns(3)

    counts = confidence[
        "Confidence_Classification"
    ].value_counts()

    for col, label in zip(
        [c1, c2, c3],
        ["Normal", "Suspicious", "DoS"]
    ):
        col.metric(label, f"{counts.get(label, 0):,}")

    st.subheader("Classification Distribution")

    st.bar_chart(
        counts.reindex(
            ["Normal", "Suspicious", "DoS"],
            fill_value=0
        )
    )

    st.subheader("Actual Label vs Confidence Classification")

    confidence_table = pd.crosstab(
        confidence["Actual_Label"],
        confidence["Confidence_Classification"]
    )

    st.dataframe(confidence_table, use_container_width=True)

    st.subheader("DoS Probability Distribution")

    fig, ax = plt.subplots()
    ax.hist(
        confidence["DoS_Probability_Percent"],
        bins=30
    )
    ax.set_xlabel("DoS Probability (%)")
    ax.set_ylabel("Number of Observations")
    ax.set_title("Random Forest DoS Probability")
    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "Prototype thresholds: below 30% = Normal; "
        "30% to below 70% = Suspicious; 70% or above = DoS."
    )

# ---------------------------------------
# EXPLAINABLE AI
# ---------------------------------------

elif section == "Explainable AI":

    st.header("Explainable Artificial Intelligence")

    st.subheader("SHAP Global Feature Importance")

    shap_sorted = shap_data.sort_values(
        "Mean_Absolute_SHAP",
        ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        shap_sorted["Feature"],
        shap_sorted["Mean_Absolute_SHAP"]
    )

    ax.set_xlabel("Mean Absolute SHAP Value")
    ax.set_title("Global Feature Importance")

    st.pyplot(fig)
    plt.close(fig)

    st.caption(
        "The message-frequency feature has the highest SHAP "
        "importance in this experiment. SHAP importance indicates "
        "contribution magnitude, not causation."
    )

    st.divider()

    st.subheader("SHAP Explanation Images")

    shap_images = [
        OUTPUT_DIR / "receiver_shap_global.png",
        OUTPUT_DIR / "receiver_shap_beeswarm.png"
    ]

    for image_path in shap_images:
        if image_path.exists():
            st.image(
                str(image_path),
                caption=image_path.name,
                use_container_width=True
            )
        else:
            st.warning(f"Image not found: {image_path.name}")

    st.divider()

    st.subheader("LIME Local Explanations")

    lime_images = [
        (
            "Benign Message Explanation",
            OUTPUT_DIR / "receiver_lime_benign.png"
        ),
        (
            "Attacker Message Explanation",
            OUTPUT_DIR / "receiver_lime_attacker.png"
        )
    ]

    for caption, image_path in lime_images:
        if image_path.exists():
            st.image(
                str(image_path),
                caption=caption,
                use_container_width=True
            )
        else:
            st.warning(f"Image not found: {image_path.name}")

    st.info(
        "SHAP explains global feature contributions. "
        "LIME explains individual predictions using local "
        "approximations."
    )

# ---------------------------------------
# EWMA MONITORING
# ---------------------------------------

elif section == "EWMA Monitoring":

    st.header("EWMA Network Anomaly Monitoring")

    st.write(
        "EWMA monitors message frequency over time "
        "for individual receiver-sender streams."
    )

    c1, c2, c3 = st.columns(3)

    anomaly_count = (
        ewma["EWMA_Anomaly"] == "Anomaly"
    ).sum()

    normal_count = (
        ewma["EWMA_Anomaly"] == "Normal"
    ).sum()

    c1.metric("Total Records", f"{len(ewma):,}")
    c2.metric("Anomalies", f"{anomaly_count:,}")
    c3.metric("Normal", f"{normal_count:,}")

    st.subheader("EWMA Frequency Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        ewma["EWMA_Frequency"],
        bins=30
    )

    ax.axvline(
        2.0226,
        linestyle="--",
        label="Training-derived threshold"
    )

    ax.set_xlabel("EWMA Frequency")
    ax.set_ylabel("Number of Observations")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("EWMA Classification")

    st.bar_chart(
        ewma["EWMA_Anomaly"].value_counts()
    )

    st.caption(
        "EWMA alpha = 0.3. Threshold = 2.0226, "
        "estimated from benign training observations."
    )

# ---------------------------------------
# SEVERITY AND MITIGATION
# ---------------------------------------

elif section == "Severity and Mitigation":

    st.header("Severity-Based Alerting and Mitigation")

    st.subheader("Severity Distribution")

    severity_counts = (
        severity["Severity"]
        .value_counts()
        .reindex(["Low", "Medium", "High"], fill_value=0)
    )

    st.bar_chart(severity_counts)

    st.subheader("Mitigation Decision Distribution")

    mitigation_counts = (
        severity["Mitigation_Action"]
        .value_counts()
        .reindex(["ALLOW", "MONITOR", "FILTER"], fill_value=0)
    )

    st.bar_chart(mitigation_counts)

    st.divider()

    st.subheader("Actual Label vs Severity")

    st.dataframe(
        pd.crosstab(
            severity["Actual_Label"],
            severity["Severity"]
        ),
        use_container_width=True
    )

    st.subheader("Actual Label vs Mitigation")

    st.dataframe(
        pd.crosstab(
            severity["Actual_Label"],
            severity["Mitigation_Action"]
        ),
        use_container_width=True
    )

    st.info(
        "Mitigation actions are simulated prototype decisions. "
        "The FILTER action does not block real network packets."
    )

# ---------------------------------------
# DETECTION COMPARISON
# ---------------------------------------

elif section == "Detection Comparison":

    st.header("Detection Method Comparison")

    st.write(
        "Comparison of Random Forest, EWMA, combined alert "
        "and filter-only strategies on the test observations."
    )

    display_comparison = comparison.copy()

    percentage_columns = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "FPR"
    ]

    for col in percentage_columns:
        display_comparison[col] *= 100

    st.dataframe(
        display_comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("F1 Score Comparison")

    f1_data = comparison.set_index("Method")[["F1"]] * 100

    st.bar_chart(f1_data)

    st.subheader("Recall Comparison")

    recall_data = (
        comparison.set_index("Method")[["Recall"]] * 100
    )

    st.bar_chart(recall_data)

    st.subheader("Accuracy Comparison")

    accuracy_data = (
        comparison.set_index("Method")[["Accuracy"]] * 100
    )

    st.bar_chart(accuracy_data)

    st.caption(
        "Combined Alert treats both MONITOR and FILTER as positive "
        "alerts. Filter Only treats only FILTER as positive."
    )

# ---------------------------------------
# EXPLORE TEST RECORDS
# ---------------------------------------

elif section == "Explore Test Records":

    st.header("Explore Receiver Test Records")

    st.write(
        "Filter and inspect individual receiver-message "
        "observations from the test dataset."
    )

    severity_options = ["All"] + sorted(
        severity["Severity"].dropna().unique().tolist()
    )

    selected_severity = st.selectbox(
        "Select severity",
        severity_options
    )

    action_options = ["All"] + sorted(
        severity["Mitigation_Action"].dropna().unique().tolist()
    )

    selected_action = st.selectbox(
        "Select mitigation action",
        action_options
    )

    filtered = severity.copy()

    if selected_severity != "All":
        filtered = filtered[
            filtered["Severity"] == selected_severity
        ]

    if selected_action != "All":
        filtered = filtered[
            filtered["Mitigation_Action"] == selected_action
        ]

    st.metric("Matching Observations", f"{len(filtered):,}")

    display_columns = [
        "vehicle_speed",
        "messages_received_last_1s",
        "DoS_Probability_Percent",
        "Confidence_Classification",
        "EWMA_Anomaly",
        "Severity_Score",
        "Severity",
        "Mitigation_Action",
        "Actual_Label"
    ]

    st.dataframe(
        filtered[display_columns].head(1000),
        use_container_width=True,
        hide_index=True
    )

    st.download_button(
        "Download Filtered Test Records",
        data=filtered.to_csv(index=False),
        file_name="filtered_vanet_test_records.csv",
        mime="text/csv"
    )

# ---------------------------------------
# FOOTER
# ---------------------------------------

st.divider()

st.caption(
    "Final-Year Cyber Security Project | "
    "Explainable AI-Based VANET DoS Detection and Mitigation"
)
