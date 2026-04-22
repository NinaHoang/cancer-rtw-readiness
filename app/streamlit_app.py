from pathlib import Path
import sys
import math

import streamlit as st
import pandas as pd

# Allow import from project root if running from /app
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(
    page_title="RTW Readiness PoV",
    page_icon="💼",
    layout="wide",
)

st.title("AI-enabled Return-to-Work Readiness PoV")
st.caption(
    "Proof-of-value demo for assessing return-to-work readiness in cancer survivorship. "
    "This tool is literature-informed and for demonstration only."
)

st.markdown("---")

# -----------------------------
# Helper functions
# -----------------------------
def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def compute_readiness(inputs: dict):
    chemo_flag = 1 if inputs["treatment_type"] in ["surgery_chemo", "surgery_chemo_radiation"] else 0

    stage_num = {"I": 1, "II": 2, "III": 3}
    stage_score = stage_num[inputs["disease_stage"]]

    flex_num = {"low": 0, "medium": 1, "high": 2}
    flex_score = flex_num[inputs["employer_flexibility"]]

    ses_num = {"low": 0, "middle": 1, "high": 2}
    ses_score = ses_num[inputs["socioeconomic_status"]]

    wb_penalty_map = {
        "healthy_ambitious": 0.0,
        "unambitious": 1.5,
        "resigned": 2.5,
        "excessively_ambitious": 0.5,
    }
    wb_penalty = wb_penalty_map[inputs["work_behavior_type"]]

    job_pen = {"sedentary": 0.0, "mixed": 0.8, "physical": 1.8}
    job_score = job_pen[inputs["job_type"]]

    contributions = {
        "Base calibration": 4.5,
        "Fatigue": -0.35 * inputs["fatigue_score"],
        "Anxiety": -0.25 * inputs["anxiety_score"],
        "Cognitive limitation": -0.20 * inputs["cognitive_limitation_score"],
        "Pain": -0.20 * inputs["pain_score"],
        "Depression": -0.40 * inputs["depression_indicator"],
        "Chemotherapy": -0.50 * chemo_flag,
        "Disease stage": -0.30 * stage_score,
        "Comorbidities": -0.20 * inputs["comorbidities"],
        "Age": -0.015 * inputs["age"],
        "Work behavior": -0.50 * wb_penalty,
        "Job type": -0.30 * job_score,
        "RTW confidence": 0.35 * inputs["rtw_confidence_score"],
        "Employer support": 0.40 * inputs["employer_support"],
        "Employer flexibility": 0.25 * flex_score,
        "Socioeconomic status": 0.20 * ses_score,
        "Months since treatment": 0.015 * inputs["months_since_treatment"],
        "Physical functioning": 0.20 * inputs["physical_functioning_score"],
    }

    readiness_score = sum(contributions.values())
    probability = sigmoid(readiness_score)
    ready = int(probability > 0.50)

    return readiness_score, probability, ready, contributions


def summarize_factors(contributions: dict):
    items = [(k, v) for k, v in contributions.items() if k != "Base calibration"]
    positives = sorted([x for x in items if x[1] > 0], key=lambda x: x[1], reverse=True)[:3]
    negatives = sorted([x for x in items if x[1] < 0], key=lambda x: x[1])[:3]
    return positives, negatives


# -----------------------------
# Inputs
# -----------------------------
left, right = st.columns([1, 1.2])

with left:
    st.subheader("Patient and workplace inputs")

    age = st.slider("Age", 25, 65, 50)

    cancer_type = st.selectbox(
        "Cancer type",
        ["breast", "colorectal", "prostate", "other"],
    )

    treatment_type = st.selectbox(
        "Treatment type",
        ["surgery_only", "surgery_chemo", "surgery_chemo_radiation", "surgery_radiation"],
    )

    disease_stage = st.selectbox("Disease stage", ["I", "II", "III"])
    months_since_treatment = st.slider("Months since treatment completion", 1, 48, 12)
    comorbidities = st.slider("Number of comorbidities", 0, 3, 1)

    fatigue_score = st.slider("Fatigue score", 1, 10, 6)
    pain_score = st.slider("Pain score", 1, 10, 5)
    cognitive_limitation_score = st.slider("Cognitive limitation score", 1, 10, 5)
    physical_functioning_score = st.slider("Physical functioning score", 1, 10, 6)

    anxiety_score = st.slider("Anxiety score", 1, 10, 5)
    depression_indicator = st.selectbox("Depression indicator", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    rtw_confidence_score = st.slider("RTW confidence score", 1, 10, 6)

    work_behavior_type = st.selectbox(
        "Work behavior type",
        ["healthy_ambitious", "unambitious", "resigned", "excessively_ambitious"],
    )

    socioeconomic_status = st.selectbox(
        "Socioeconomic status",
        ["low", "middle", "high"],
    )

    job_type = st.selectbox("Job type", ["sedentary", "mixed", "physical"])
    employer_flexibility = st.selectbox("Employer flexibility", ["low", "medium", "high"])
    employer_support = st.selectbox("Employer support", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    run = st.button("Assess RTW readiness", type="primary")

inputs = {
    "age": age,
    "cancer_type": cancer_type,
    "treatment_type": treatment_type,
    "disease_stage": disease_stage,
    "months_since_treatment": months_since_treatment,
    "comorbidities": comorbidities,
    "fatigue_score": fatigue_score,
    "pain_score": pain_score,
    "cognitive_limitation_score": cognitive_limitation_score,
    "physical_functioning_score": physical_functioning_score,
    "anxiety_score": anxiety_score,
    "depression_indicator": depression_indicator,
    "rtw_confidence_score": rtw_confidence_score,
    "work_behavior_type": work_behavior_type,
    "socioeconomic_status": socioeconomic_status,
    "job_type": job_type,
    "employer_flexibility": employer_flexibility,
    "employer_support": employer_support,
}

# -----------------------------
# Output
# -----------------------------
with right:
    st.subheader("Assessment output")

    if run:
        readiness_score, probability, ready, contributions = compute_readiness(inputs)
        positives, negatives = summarize_factors(contributions)

        if ready:
            st.success(f"Predicted status: READY")
        else:
            st.error(f"Predicted status: NOT READY")

        st.metric("Probability of RTW readiness", f"{probability:.1%}")
        st.metric("Readiness score", f"{readiness_score:.2f}")

        st.markdown("### Key positive contributors")
        if positives:
            for name, value in positives:
                st.write(f"- **{name}**: {value:+.2f}")
        else:
            st.write("No strong positive contributors identified.")

        st.markdown("### Key negative contributors")
        if negatives:
            for name, value in negatives:
                st.write(f"- **{name}**: {value:+.2f}")
        else:
            st.write("No strong negative contributors identified.")

        st.markdown("### Input summary")
        st.dataframe(pd.DataFrame([inputs]), use_container_width=True)

        st.info(
            "This is a literature-informed proof-of-value tool using a synthetic scoring logic. "
            "It is not clinically validated and should not be used for real patient decision-making."
        )
    else:
        st.write("Enter values on the left and click **Assess RTW readiness**.")