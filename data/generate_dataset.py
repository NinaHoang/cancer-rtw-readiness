"""
Synthetic RTW Readiness Dataset Generator
==========================================
Grounded in evidence from 8 peer-reviewed studies on return to work in cancer survivors.

Key literature sources informing feature selection and label logic:
- Duijts et al. (2014): Fatigue, cognitive limitations, depression → work ability
- Islam et al. (2014): Fatigue, chemotherapy, physical job demand → RTW barriers
- Hou et al. (2021): Depression, fatigue, cognitive limitations, work ability → employment readiness
- Ullrich et al. (2022): Fatigue, SES, unambitious work behavior → long-term RTW
- van Maarschalkerweerd et al. (2020): Barriers shift over time (disease → personal/work)
- den Bakker et al. (2018): Neo-adjuvant therapy, comorbidities → RTW delay
- Forbes et al. (2024): Age, comorbidities, health insurance → RTW in Australia
- Hoving et al. (2009): Counseling + exercise intervention → 75-85% RTW rates
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 400  # sample size — sufficient for robust ML with this feature count


def generate_dataset(n=N):
    # ── Clinical features ────────────────────────────────────────────────────
    cancer_type = np.random.choice(
        ["breast", "colorectal", "prostate", "other"],
        size=n,
        p=[0.45, 0.25, 0.15, 0.15],  # reflects epidemiological prevalence
    )

    # Treatment type: chemo hardest on RTW (Islam 2014, den Bakker 2018)
    treatment = np.random.choice(
        ["surgery_only", "surgery_chemo", "surgery_chemo_radiation", "surgery_radiation"],
        size=n,
        p=[0.20, 0.40, 0.25, 0.15],
    )

    # Months since treatment completion: longer = higher readiness (Hou 2021)
    months_since_treatment = np.random.exponential(scale=14, size=n).clip(1, 48).astype(int)

    # Disease stage (early = better RTW, den Bakker 2018)
    disease_stage = np.random.choice(["I", "II", "III"], size=n, p=[0.40, 0.35, 0.25])

    # Comorbidities (Forbes 2024: 3+ comorbidities → lower RTW)
    comorbidities = np.random.choice([0, 1, 2, 3], size=n, p=[0.40, 0.30, 0.20, 0.10])

    # ── Functional / symptom features ────────────────────────────────────────
    # Fatigue: strongest single predictor across all studies (1=none, 10=severe)
    fatigue = np.random.normal(loc=5.5, scale=2.2, size=n).clip(1, 10)

    # Pain: secondary predictor (Islam 2014, Duijts 2014)
    pain = np.random.normal(loc=4.5, scale=2.0, size=n).clip(1, 10)

    # Cognitive limitations: "most problematic post-treatment symptom" (Duijts 2014)
    cognitive_limitation = np.random.normal(loc=4.8, scale=2.1, size=n).clip(1, 10)

    # Physical functioning (1=very poor, 10=excellent)
    physical_functioning = (10 - fatigue + np.random.normal(0, 1, n)).clip(1, 10)

    # ── Psychological features ────────────────────────────────────────────────
    # Anxiety (Hou 2021, Duijts 2014)
    anxiety = np.random.normal(loc=5.0, scale=2.3, size=n).clip(1, 10)

    # Depression indicator (binary; PHQ-9 ≥10 threshold from Hou 2021)
    depression = (np.random.normal(loc=5.0, scale=2.5, size=n) > 6.5).astype(int)

    # Confidence to return to work (self-efficacy; Islam 2014, Hou 2021)
    rtw_confidence = np.random.normal(loc=5.5, scale=2.2, size=n).clip(1, 10)

    # Work behavior: unambitious/resigned → worse RTW (Ullrich 2022, OR 4.48)
    work_behavior = np.random.choice(
        ["healthy_ambitious", "unambitious", "resigned", "excessively_ambitious"],
        size=n,
        p=[0.28, 0.35, 0.17, 0.20],
    )

    # ── Socioeconomic / work-related features ─────────────────────────────────
    # SES: low SES OR 4.81 for not working (Ullrich 2022)
    ses = np.random.choice(["low", "middle", "high"], size=n, p=[0.25, 0.50, 0.25])

    # Job type: physical work → harder RTW (Islam 2014)
    job_type = np.random.choice(
        ["sedentary", "mixed", "physical"], size=n, p=[0.45, 0.35, 0.20]
    )

    # Employer flexibility: key facilitator (van Maarschalkerweerd 2020, Islam 2014)
    employer_flexibility = np.random.choice(
        ["low", "medium", "high"], size=n, p=[0.25, 0.40, 0.35]
    )

    # Employer support (Hoving 2009: counseling → 76% vs 54% RTW)
    employer_support = np.random.choice([0, 1], size=n, p=[0.35, 0.65])

    # Age: older → less likely to RTW (Islam 2014, Forbes 2024)
    age = np.random.normal(loc=50, scale=8, size=n).clip(25, 65).astype(int)

    # ── Encode categoricals for label generation ──────────────────────────────
    chemo_flag = np.isin(treatment, ["surgery_chemo", "surgery_chemo_radiation"]).astype(int)
    stage_num = {"I": 1, "II": 2, "III": 3}
    stage_score = np.array([stage_num[s] for s in disease_stage])
    flex_num = {"low": 0, "medium": 1, "high": 2}
    flex_score = np.array([flex_num[f] for f in employer_flexibility])
    ses_num = {"low": 0, "middle": 1, "high": 2}
    ses_score = np.array([ses_num[s] for s in ses])
    wb_penalty = np.where(work_behavior == "resigned", 2.5,
                 np.where(work_behavior == "unambitious", 1.5,
                 np.where(work_behavior == "excessively_ambitious", 0.5, 0)))
    job_pen = {"sedentary": 0, "mixed": 0.8, "physical": 1.8}
    job_score = np.array([job_pen[j] for j in job_type])

    # ── Label generation: evidence-based scoring ──────────────────────────────
    # Each term weighted by effect size / OR from the literature.
    # Intercept (+4.5) calibrated so that the average patient has ~55% probability
    # of being RTW-ready, consistent with literature rates of 43–93% (Islam 2014).
    readiness_score = (
        + 4.5                               # intercept (calibration)
        - 0.35 * fatigue                    # strongest predictor (Duijts, Islam, Ullrich)
        - 0.25 * anxiety                    # significant predictor (Hou, Duijts)
        - 0.20 * cognitive_limitation       # Duijts 2014; Hou 2021
        - 0.20 * pain                       # Islam 2014
        - 0.40 * depression                 # Hou 2021 (PHQ-9 β = −0.27)
        - 0.50 * chemo_flag                 # Islam 2014; den Bakker 2018
        - 0.30 * stage_score                # den Bakker 2018
        - 0.20 * comorbidities              # Forbes 2024; den Bakker 2018
        - 0.015 * age                       # Forbes 2024; Ullrich 2022
        - 0.50 * wb_penalty                 # Ullrich 2022 (OR 4.48)
        - 0.30 * job_score                  # Islam 2014
        + 0.35 * rtw_confidence             # Hou 2021 (WAI)
        + 0.40 * employer_support           # Hoving 2009
        + 0.25 * flex_score                 # van Maarschalkerweerd 2020
        + 0.20 * ses_score                  # Ullrich 2022 (OR 4.81 for low SES)
        + 0.015 * months_since_treatment    # Hou 2021 (r = 0.09)
        + 0.20 * physical_functioning
        + np.random.normal(0, 0.5, n)       # residual noise
    )

    # Sigmoid → probability → binary label
    prob_ready = 1 / (1 + np.exp(-readiness_score))
    rtw_ready = (prob_ready > 0.50).astype(int)

    df = pd.DataFrame({
        # Clinical
        "cancer_type": cancer_type,
        "treatment_type": treatment,
        "months_since_treatment": months_since_treatment,
        "disease_stage": disease_stage,
        "comorbidities": comorbidities,
        # Functional
        "fatigue_score": fatigue.round(1),
        "pain_score": pain.round(1),
        "cognitive_limitation_score": cognitive_limitation.round(1),
        "physical_functioning_score": physical_functioning.round(1),
        # Psychological
        "anxiety_score": anxiety.round(1),
        "depression_indicator": depression,
        "rtw_confidence_score": rtw_confidence.round(1),
        "work_behavior_type": work_behavior,
        # Socioeconomic / Work
        "socioeconomic_status": ses,
        "job_type": job_type,
        "employer_flexibility": employer_flexibility,
        "employer_support": employer_support,
        "age": age,
        # Label
        "rtw_ready": rtw_ready,
    })

    print(f"Dataset generated: {n} rows")
    print(f"Class balance — Ready: {rtw_ready.sum()} ({rtw_ready.mean():.1%}) | "
          f"Not Ready: {(1-rtw_ready).sum()} ({(1-rtw_ready).mean():.1%})")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    df.to_csv("synthetic_rtw_dataset.csv", index=False)
    print("\nSaved → synthetic_rtw_dataset.csv")
    print(df.head(3).to_string())
