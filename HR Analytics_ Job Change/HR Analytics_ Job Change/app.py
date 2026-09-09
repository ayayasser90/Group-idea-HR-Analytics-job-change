import pickle
from pathlib import Path
 
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
 
BASE_DIR = Path(__file__).resolve().parent
 
st.set_page_config(page_title="Smart Recruitment Assistant", page_icon="🧭", layout="wide")
 

EDUCATION_MAPPING = {
    "Primary School": 0,
    "High School": 1,
    "Graduate": 2,
    "Masters": 3,
    "Phd": 4,
}
 
COMPANY_SIZE_MAPPING = {
    "<10": 0,
    "10/49": 1,
    "50-99": 2,
    "100-500": 3,
    "500-999": 4,
    "1000-4999": 5,
    "5000-9999": 6,
    "10000+": 7,
}
 
CATEGORICAL_COLS = ["gender", "enrolled_university", "major_discipline", "company_type"]
 
 
def require_file(filename: str):
   
    if not (BASE_DIR / filename).exists():
        st.error(
            f"error "
           
        )
        st.stop()
 

COMPANY_SIZE_LABELS = {v: k for k, v in COMPANY_SIZE_MAPPING.items()}
 
 

@st.cache_resource
def load_model_package():
    with open(BASE_DIR / "smart_recruitment_model.pkl", "rb") as f:
        package = pickle.load(f)
    model = package["model"]
    selector = package["selector"]
    feature_columns = package["feature_columns"]
    selected_features = [
        f for f, keep in zip(feature_columns, selector.get_support()) if keep
    ]
    return model, selector, feature_columns, selected_features
 
 
def _clean_experience(series: pd.Series) -> pd.Series:
    series = series.replace({"<1": 0, ">20": 21})
    return pd.to_numeric(series, errors="coerce")
 
 
def _clean_last_new_job(series: pd.Series) -> pd.Series:
    series = series.replace({"never": 0, ">4": 5})
    return pd.to_numeric(series, errors="coerce")
 
 

@st.cache_data
def load_eda_data():
    df = pd.read_csv(BASE_DIR / "train_clean.csv")
 
    if pd.api.types.is_numeric_dtype(df["company_size"]):
        df["company_size"] = df["company_size"].map(COMPANY_SIZE_LABELS)
 
    bins = [-1, 2, 5, 10, 15, 100]
    labels = ["0-2", "3-5", "6-10", "11-15", "16+"]
    df["experience_bucket"] = pd.cut(df["experience"], bins=bins, labels=labels)
    return df
 
 

@st.cache_data
def fit_training_impute_stats():
    require_file("aug_train.csv")
    df = pd.read_csv(BASE_DIR / "aug_train.csv")
    dfc = df.copy()
 
    dfc["gender"] = dfc["gender"].fillna("Unknown")
 
    education_level_mode = dfc["education_level"].mode()[0]
    dfc["education_level"] = dfc["education_level"].fillna(education_level_mode)
 
    temp = dfc.dropna(subset=["education_level", "enrolled_university"])
    university_modes = temp.groupby("education_level")["enrolled_university"].agg(lambda x: x.mode()[0])
    enrolled_university_mode = dfc["enrolled_university"].mode()[0]
    dfc["enrolled_university"] = dfc["enrolled_university"].fillna(dfc["education_level"].map(university_modes))
    dfc["enrolled_university"] = dfc["enrolled_university"].fillna(enrolled_university_mode)
 
    temp = dfc.dropna(subset=["education_level", "major_discipline"])
    major_modes = temp.groupby("education_level")["major_discipline"].agg(lambda x: x.mode()[0])
    major_discipline_mode = dfc["major_discipline"].mode()[0]
    dfc["major_discipline"] = dfc["major_discipline"].fillna(dfc["education_level"].map(major_modes))
    dfc["major_discipline"] = dfc["major_discipline"].fillna(major_discipline_mode)
 
    temp = dfc.dropna(subset=["relevent_experience", "experience"])
    experience_modes = temp.groupby("relevent_experience")["experience"].agg(lambda x: x.mode()[0])
    experience_mode = dfc["experience"].mode()[0]
    dfc["experience"] = dfc["experience"].fillna(dfc["relevent_experience"].map(experience_modes))
    dfc["experience"] = dfc["experience"].fillna(experience_mode)
 
    temp = dfc.dropna(subset=["experience", "last_new_job"])
    last_job_modes = temp.groupby("experience")["last_new_job"].agg(lambda x: x.mode()[0])
    last_new_job_mode = dfc["last_new_job"].mode()[0]
 
    dfc["company_type"] = dfc["company_type"].fillna("Unknown")
    temp = dfc.dropna(subset=["company_size"])
    company_size_modes = temp.groupby("company_type")["company_size"].agg(lambda x: x.mode()[0])
    company_size_mode = dfc["company_size"].mode()[0]
 
    return {
        "education_level_mode": education_level_mode,
        "university_modes": university_modes,
        "enrolled_university_mode": enrolled_university_mode,
        "major_modes": major_modes,
        "major_discipline_mode": major_discipline_mode,
        "experience_modes": experience_modes,
        "experience_mode": experience_mode,
        "last_job_modes": last_job_modes,
        "last_new_job_mode": last_new_job_mode,
        "company_size_modes": company_size_modes,
        "company_size_mode": company_size_mode,
    }
 
 
def clean_test_like_notebook(test_df: pd.DataFrame, stats: dict) -> pd.DataFrame:
    tc = test_df.copy()
 
    tc["gender"] = tc["gender"].fillna("Unknown")
    tc["education_level"] = tc["education_level"].fillna(stats["education_level_mode"])
    tc["enrolled_university"] = tc["enrolled_university"].fillna(tc["education_level"].map(stats["university_modes"]))
    tc["enrolled_university"] = tc["enrolled_university"].fillna(stats["enrolled_university_mode"])
    tc["major_discipline"] = tc["major_discipline"].fillna(tc["education_level"].map(stats["major_modes"]))
    tc["major_discipline"] = tc["major_discipline"].fillna(stats["major_discipline_mode"])
    tc["experience"] = tc["experience"].fillna(tc["relevent_experience"].map(stats["experience_modes"]))
    tc["experience"] = tc["experience"].fillna(stats["experience_mode"])
    tc["last_new_job"] = tc["last_new_job"].fillna(tc["experience"].map(stats["last_job_modes"]))
    tc["last_new_job"] = tc["last_new_job"].fillna(stats["last_new_job_mode"])
    tc["company_type"] = tc["company_type"].fillna("Unknown")
    tc["company_size"] = tc["company_size"].fillna(tc["company_type"].map(stats["company_size_modes"]))
    tc["company_size"] = tc["company_size"].fillna(stats["company_size_mode"])
 
    tc["experience"] = _clean_experience(tc["experience"]).astype(int)
    tc["last_new_job"] = _clean_last_new_job(tc["last_new_job"]).astype(int)
 
    tc["job_stability"] = tc["last_new_job"] / (tc["experience"] + 1)
    tc["low_experience"] = (tc["experience"] <= 2).astype(int)
    tc["recent_job_change"] = (tc["last_new_job"] <= 1).astype(int)
    tc["high_experience"] = (tc["experience"] >= 10).astype(int)
    return tc
 
 
def encode_for_model(df_readable: pd.DataFrame, feature_columns) -> pd.DataFrame:
   
   
    enc = df_readable.copy()
 
    enc["relevent_experience"] = enc["relevent_experience"].map({
        "Has relevent experience": 1,
        "No relevent experience": 0,
    })
 
    enc["education_level"] = enc["education_level"].map(EDUCATION_MAPPING)
    enc["company_size"] = enc["company_size"].map(COMPANY_SIZE_MAPPING)
    enc["company_size"] = enc["company_size"].fillna(-1)
 
    enc = pd.get_dummies(enc, columns=CATEGORICAL_COLS, drop_first=False)
    enc = enc.reindex(columns=feature_columns, fill_value=0)
    return enc
 
 

@st.cache_data
def load_predictions():
    require_file("aug_test.csv")
    require_file("aug_train.csv")
    test_raw = pd.read_csv(BASE_DIR / "aug_test.csv")
 
   
    stats = fit_training_impute_stats()
    test_clean = clean_test_like_notebook(test_raw, stats)
 
    preds_path = BASE_DIR / "final_job_change_predictions_2.csv"
    if preds_path.exists():
        preds = pd.read_csv(preds_path)
        result = preds.merge(test_clean, on="enrollee_id", how="left")
    else:
        # fallback: احسبي التوقعات لايف
        model, selector, feature_columns, _ = load_model_package()
        X_test = encode_for_model(test_clean, feature_columns)
        X_test_selected = selector.transform(X_test)
        probabilities = model.predict_proba(X_test_selected)[:, 1] * 100
 
        result = test_clean.copy()
        result["job_change_probability"] = probabilities.round(2)
 
    return result
 
 
def risk_label(probability_pct: float):
    if probability_pct >= 70:
        return "🔴 High", "#e74c3c"
    elif probability_pct >= 40:
        return "🟠 Medium", "#f39c12"
    else:
        return "🟢 Low", "#27ae60"
 
 

def build_candidate_row(form_inputs: dict, feature_columns) -> pd.DataFrame:
    row = {
        "city_development_index": form_inputs["city_development_index"],
        "gender": form_inputs["gender"],
        "relevent_experience": form_inputs["relevent_experience"],
        "enrolled_university": form_inputs["enrolled_university"],
        "education_level": form_inputs["education_level"],
        "major_discipline": form_inputs["major_discipline"],
        "experience": form_inputs["experience"],
        "company_size": form_inputs["company_size"],
        "company_type": form_inputs["company_type"],
        "last_new_job": form_inputs["last_new_job"],
        "training_hours": form_inputs["training_hours"],
    }
    df = pd.DataFrame([row])
 
    df["job_stability"] = df["last_new_job"] / (df["experience"] + 1)
    df["low_experience"] = (df["experience"] <= 2).astype(int)
    df["recent_job_change"] = (df["last_new_job"] <= 1).astype(int)
    df["high_experience"] = (df["experience"] >= 10).astype(int)
 
    return encode_for_model(df, feature_columns)
 
 
def predict_candidate(form_inputs: dict):
    model, selector, feature_columns, selected_features = load_model_package()
    row = build_candidate_row(form_inputs, feature_columns)
    row_selected = selector.transform(row)
    probability = model.predict_proba(row_selected)[0, 1]
    return probability, selected_features, model
 
 

def page_home():
    df = load_eda_data()
    total_candidates = len(df)
    job_changers_pct = df["target"].mean() * 100
    stay_pct = 100 - job_changers_pct
 
    st.title("Smart Recruitment Assistant")
    st.subheader("Understand your candidates. Spot potential job changers. Make better hiring decisions.")
    st.write("")
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Candidates analyzed", f"{total_candidates:,}")
    col2.metric("Potential job changers", f"{job_changers_pct:.0f}%")
    col3.metric("Likely to stay", f"{stay_pct:.0f}%")
 
    st.write("")
    col_chart, col_text = st.columns([1, 1])
 
    with col_chart:
        pie_df = df["target"].map({0: "Likely to Stay", 1: "Likely to Change"}).value_counts().reset_index()
        pie_df.columns = ["Status", "Count"]
        fig = px.pie(
            pie_df, names="Status", values="Count", hole=0.55, color="Status",
            color_discrete_map={"Likely to Stay": "#27ae60", "Likely to Change": "#e74c3c"},
            title="Are candidates really at risk of leaving?"
        )
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
 
    with col_text:
        st.write("")
        if job_changers_pct > 0:
            st.markdown(f"### 1 in {round(100 / job_changers_pct)} candidates showed a tendency to change jobs.")
       
        
 
 
def page_drivers():
    st.title("What Drives Job Change?")
    df = load_eda_data()
 
    st.markdown("### 🎓 Does education matter?")
    edu_order = ["Primary School", "High School", "Graduate", "Masters", "Phd"]
    edu_rate = (
        df.dropna(subset=["education_level"])
        .groupby("education_level")["target"].mean().mul(100)
        .reindex(edu_order).dropna().reset_index()
    )
    edu_rate.columns = ["Education Level", "Job Change Rate (%)"]
    st.plotly_chart(
        px.bar(edu_rate, x="Education Level", y="Job Change Rate (%)",
               text_auto=".1f", color_discrete_sequence=["#3498db"],
               title="Education Level vs Job Change Rate"),
        use_container_width=True
    )
 
    st.markdown("### Does experience make candidates more stable?")
    exp_rate = (
        df.dropna(subset=["experience_bucket"])
        .groupby("experience_bucket", observed=True)["target"].mean().mul(100).reset_index()
    )
    exp_rate.columns = ["Experience Group (years)", "Job Change Rate (%)"]
    st.plotly_chart(
        px.bar(exp_rate, x="Experience Group (years)", y="Job Change Rate (%)",
               text_auto=".1f", color_discrete_sequence=["#9b59b6"],
               title="Experience Group vs Job Change Rate"),
        use_container_width=True
    )
 
    st.markdown("### Recent job movement")
    job_move_rate = (
        df.dropna(subset=["last_new_job"])
        .groupby("last_new_job")["target"].mean().mul(100).reset_index()
    )
    job_move_rate.columns = ["Years Since Last Job Change", "Current Job Change Rate (%)"]
    st.plotly_chart(
        px.bar(job_move_rate, x="Years Since Last Job Change", y="Current Job Change Rate (%)",
               text_auto=".1f", color_discrete_sequence=["#e67e22"],
               title="Previous Job Change vs Current Job Change"),
        use_container_width=True
    )
 
 
def page_explorer():
    st.title("Candidate Risk Explorer")
    df = load_eda_data()
 
    st.sidebar.header("Filters")
 
    def multiselect_all(label, series):
        options = sorted(series.dropna().unique())
        return st.sidebar.multiselect(label, options, default=options)
 
    education = multiselect_all("Education", df["education_level"])
    experience_bucket = st.sidebar.multiselect(
        "Experience", ["0-2", "3-5", "6-10", "11-15", "16+"],
        default=["0-2", "3-5", "6-10", "11-15", "16+"]
    )
    company_size = multiselect_all("Company Size", df["company_size"])
    company_type = multiselect_all("Company Type", df["company_type"])
    relevent_experience = multiselect_all("Relevant Experience", df["relevent_experience"])
    enrolled_university = multiselect_all("University Enrollment", df["enrolled_university"])
 
    filtered = df[
        df["education_level"].isin(education) &
        df["experience_bucket"].astype(str).isin(experience_bucket) &
        df["company_size"].isin(company_size) &
        df["company_type"].isin(company_type) &
        df["relevent_experience"].isin(relevent_experience) &
        df["enrolled_university"].isin(enrolled_university)
    ]
 
    st.markdown("### Which candidates are most likely to change jobs?")
 
    if len(filtered) == 0:
        st.warning("No candidates match the current filters.")
        return
 
    col1, col2, col3 = st.columns(3)
    col1.metric("Job change rate", f"{filtered['target'].mean() * 100:.0f}%")
    col2.metric("Candidate count", f"{len(filtered):,}")
    col3.metric("Average training hours", f"{filtered['training_hours'].mean():.0f}")
 
    exp_rate = (
        filtered.dropna(subset=["experience_bucket"])
        .groupby("experience_bucket", observed=True)["target"].mean().mul(100).reset_index()
    )
    exp_rate.columns = ["Experience Group (years)", "Job Change Rate (%)"]
    st.plotly_chart(
        px.bar(exp_rate, x="Experience Group (years)", y="Job Change Rate (%)",
               text_auto=".1f", color_discrete_sequence=["#3498db"],
               title="Job Change Rate by Experience (filtered group)"),
        use_container_width=True
    )
 
    with st.expander("View filtered candidates table"):
        display_df = filtered.copy()
        display_df["target"] = display_df["target"].map({0: "Likely to Stay", 1: "Likely to Change"})
        display_cols = [
            "enrollee_id", "education_level", "major_discipline", "experience_bucket",
            "relevent_experience", "enrolled_university", "company_size", "company_type",
            "training_hours", "target",
        ]
        display_cols = [c for c in display_cols if c in display_df.columns]
        st.dataframe(
            display_df[display_cols].rename(columns={
                "enrollee_id": "Candidate", "education_level": "Education",
                "major_discipline": "Major", "experience_bucket": "Experience (yrs)",
                "relevent_experience": "Relevant Experience", "enrolled_university": "University",
                "company_size": "Company Size", "company_type": "Company Type",
                "training_hours": "Training Hours", "target": "Status",
            }),
            use_container_width=True, hide_index=True,
        )
 
 
def page_predict():
    st.title("Candidate Risk Check")
    st.caption("Tell us about the candidate.")
 
    with st.form("candidate_form"):
        col1, col2 = st.columns(2)
 
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female", "Other", "Unknown"])
            education_level = st.selectbox(
                "Education", ["Primary School", "High School", "Graduate", "Masters", "Phd"], index=2
            )
            experience = st.slider("Experience (years)", 0, 21, 5)
            relevent_experience = st.selectbox(
                "Relevant experience", ["Has relevent experience", "No relevent experience"]
            )
            enrolled_university = st.selectbox(
                "University status", ["no_enrollment", "Full time course", "Part time course"]
            )
            major_discipline = st.selectbox(
                "Major discipline",
                ["STEM", "Humanities", "Other", "Business Degree", "Arts", "No Major"]
            )
 
        with col2:
            company_size = st.selectbox(
                "Company size",
                ["<10", "10/49", "50-99", "100-500", "500-999", "1000-4999", "5000-9999", "10000+"],
                index=2
            )
            company_type = st.selectbox(
                "Company type",
                ["Pvt Ltd", "Funded Startup", "Public Sector", "Early Stage Startup", "NGO", "Other", "Unknown"]
            )
            last_new_job = st.slider("Years since previous job change (5 = more than 4)", 0, 5, 1)
            training_hours = st.number_input("Training hours", min_value=1, max_value=400, value=40)
            city_development_index = st.slider("City development index", 0.4, 1.0, 0.9, step=0.01)
 
        submitted = st.form_submit_button("Assess Candidate")
 
    if submitted:
        form_inputs = {
            "gender": gender, "education_level": education_level, "experience": experience,
            "relevent_experience": relevent_experience, "enrolled_university": enrolled_university,
            "major_discipline": major_discipline,
            "company_size": company_size, "company_type": company_type,
            "last_new_job": last_new_job, "training_hours": training_hours,
            "city_development_index": city_development_index,
        }
 
        probability, selected_features, model = predict_candidate(form_inputs)
        probability_pct = probability * 100
        label, color = risk_label(probability_pct)
 
        st.write("---")
        col_gauge, col_info = st.columns([1, 1])
 
        with col_gauge:
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=probability_pct, number={"suffix": "%"},
                title={"text": "Job Change Risk"},
                gauge={
                    "axis": {"range": [0, 100]}, "bar": {"color": color},
                    "steps": [
                        {"range": [0, 40], "color": "#eafaf1"},
                        {"range": [40, 70], "color": "#fef5e7"},
                        {"range": [70, 100], "color": "#fdedec"},
                    ],
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
            st.plotly_chart(fig, use_container_width=True)
 
        with col_info:
            st.markdown(f"## {label}")
            st.markdown("#### Key profile signals")
            st.caption("Factors the model relies on most across all candidates.")
 
            importances = model.feature_importances_
            top_idx = importances.argsort()[::-1][:3]
            max_importance = importances.max()
            impact_labels = ["Higher impact", "Moderate impact", "Lower impact"]
 
            for rank, idx in enumerate(top_idx):
                feature_name = selected_features[idx].replace("_", " ").title()
                bar_pct = int((importances[idx] / max_importance) * 100)
                st.write(f"**{feature_name}** — {impact_labels[rank]}")
                st.progress(bar_pct)
 
 
def page_prioritize():
    st.title(" Who Should HR Look At First?")
    df = load_predictions()
    df["Risk"] = df["job_change_probability"].apply(lambda p: risk_label(p)[0])
 
    st.sidebar.header("Filters")
    risk_options = ["🔴 High", "🟠 Medium", "🟢 Low"]
    selected_risks = st.sidebar.multiselect("Risk level", risk_options, default=risk_options)
    top_n = st.sidebar.radio("Show", ["Top 10", "Top 20", "All"], index=0)
 
    filtered = df[df["Risk"].isin(selected_risks)].sort_values("job_change_probability", ascending=False)
    if top_n == "Top 10":
        filtered = filtered.head(10)
    elif top_n == "Top 20":
        filtered = filtered.head(20)
 
    col1, col2, col3 = st.columns(3)
    col1.metric("🔴 High risk", int((df["Risk"] == "🔴 High").sum()))
    col2.metric("🟠 Medium risk", int((df["Risk"] == "🟠 Medium").sum()))
    col3.metric("🟢 Low risk", int((df["Risk"] == "🟢 Low").sum()))
 
    display_cols = [
        "enrollee_id", "job_change_probability", "Risk", "education_level", "major_discipline",
        "experience", "relevent_experience", "enrolled_university",
        "company_size", "company_type", "training_hours"
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
 
    st.dataframe(
        filtered[display_cols].rename(columns={
            "enrollee_id": "Candidate", "job_change_probability": "Risk %",
            "education_level": "Education", "major_discipline": "Major",
            "experience": "Experience (yrs)", "relevent_experience": "Relevant Experience",
            "enrolled_university": "University",
            "company_size": "Company Size", "company_type": "Company Type",
            "training_hours": "Training Hours",
        }),
        use_container_width=True, hide_index=True
    )
 
    st.download_button(
        "Download filtered list as CSV",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="prioritized_candidates.csv", mime="text/csv"
    )
 
 

PAGES = {
    " The Talent Story": page_home,
    " What Drives Job Change?": page_drivers,
    " Candidate Risk Explorer": page_explorer,
    " Check a Candidate": page_predict,
    " Prioritize Candidates": page_prioritize,
}
 
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()
 