import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ============================================
# CSV 로드
# ============================================
def load_data():
    candidates = [
        "StudentsPerformance.csv",
        "StudentsPerformance_clean.csv",
        "StudentsPerformance_1000rows_synthetic.csv",
        "students.csv",
    ]

    df = None
    for f in candidates:
        try:
            df = pd.read_csv(f)
            st.sidebar.success(f"📂 Loaded dataset: {f}")
            break
        except:
            pass

    if df is None:
        st.error("❌ CSV 파일이 없습니다. app.py와 같은 곳에 넣으세요.")
        st.stop()

    rename_map = {
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parental_level_of_education",
        "test preparation course": "test_preparation_course",
        "math score": "math_score",
        "reading score": "reading_score",
        "writing score": "writing_score",
    }

    return df.rename(columns=rename_map)


# ============================================
# 모델 학습
# ============================================
def train_model(df, target):
    features = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
    ]

    X = df[features]
    y = df[target]

    transformer = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), features)]
    )

    model = RandomForestRegressor(
        n_estimators=500,
        random_state=42,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", transformer),
        ("model", model)
    ])

    pipe.fit(X, y)

    preds = pipe.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)

    return pipe, rmse, mae, r2


# ============================================
# Feature Importance 계산
# ============================================
def get_feature_importance(model, df):
    ohe = model.named_steps["prep"].named_transformers_["cat"]
    feature_names = ohe.get_feature_names_out()

    importances = model.named_steps["model"].feature_importances_

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)


# ============================================
# PDF 리포트 생성
# ============================================
def generate_pdf(pred, target, inputs_dict):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)

    pdf.cell(200, 10, txt="Student Score Prediction Report", ln=True, align="C")
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt=f"Predicted {target}: {pred:.2f}", ln=True)
    pdf.ln(5)

    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, txt="Input Variables:", ln=True)
    pdf.ln(3)

    for k, v in inputs_dict.items():
        pdf.cell(200, 8, txt=f"- {k}: {v}", ln=True)

    output_path = "prediction_report.pdf"
    pdf.output(output_path)
    return output_path


# ============================================
# Streamlit UI
# ============================================
def main():

    st.set_page_config(page_title="학생 성적 예측 시스템", page_icon="📈")

    st.markdown("""
        <h1 style="text-align:center; color:#222;">📈 학생 성적 예측 시스템</h1>
        <p style="text-align:center; color:#555;">
            전문가 수준의 머신러닝 분석 · 변수 영향 분석 · 예측 보고서 생성 기능 제공
        </p>
        <hr>
    """, unsafe_allow_html=True)

    df = load_data()

    # Tabs UI
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 데이터 분석",
        "⚙️ 모델 학습 & 성능",
        "📝 점수 예측",
        "📑 PDF 리포트"
    ])

    # ============================================================
    # 1) 데이터 분석 탭
    # ============================================================
    with tab1:
        st.header("📊 학생 데이터 분석")

        st.subheader("1) 기본 통계")
        st.write(df.describe())

        st.subheader("2) 점수 분포 시각화")
        col1, col2, col3 = st.columns(3)
        for ax, score in zip([col1, col2, col3],
                             ["math_score", "reading_score", "writing_score"]):
            with ax:
                fig, bins, patches = plt.hist(df[score], bins=20, color="#5A8DEE")
                plt.title(f"{score} Distribution")
                st.pyplot(plt.gcf())
                plt.clf()

        st.subheader("3) 부모 학력에 따른 평균 점수")
        st.bar_chart(df.groupby("parental_level_of_education")[
                     ["math_score", "reading_score", "writing_score"]].mean())

    # ============================================================
    # 2) 모델 학습 & 성능 평가 탭
    # ============================================================
    with tab2:
        st.header("⚙️ 모델 학습 및 성능")

        target = st.selectbox("예측 대상 점수", ["math_score", "reading_score", "writing_score"])

        model, rmse, mae, r2 = train_model(df, target)

        st.subheader("📈 성능 지표")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("MAE", f"{mae:.2f}")
        st.metric("R² Score", f"{r2:.3f}")

        st.subheader("🔍 Feature Importance")
        fi = get_feature_importance(model, df)
        st.dataframe(fi)

        fig, ax = plt.subplots()
        ax.barh(fi["feature"], fi["importance"], color="#1A73E8")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    # ============================================================
    # 3) 점수 예측 탭
    # ============================================================
    with tab3:
        st.header("📝 학생 점수 예측")

        with st.form("predict_form"):
            gender = st.selectbox("Gender", df["gender"].unique())
            race = st.selectbox("Race/Ethnicity", df["race_ethnicity"].unique())
            pedu = st.selectbox("Parental Education", df["parental_level_of_education"].unique())
            lunch = st.selectbox("Lunch", df["lunch"].unique())
            prep = st.selectbox("Test Preparation", df["test_preparation_course"].unique())
            target2 = st.selectbox("예측할 점수", ["math_score", "reading_score", "writing_score"])

            submitted = st.form_submit_button("예측 실행")

        if submitted:
            model, *_ = train_model(df, target2)

            input_data = pd.DataFrame([{
                "gender": gender,
                "race_ethnicity": race,
                "parental_level_of_education": pedu,
                "lunch": lunch,
                "test_preparation_course": prep,
            }])

            pred = model.predict(input_data)[0]
            st.success(f"📘 예측된 {target2}: {pred:.2f}점")

            st.session_state["last_pred"] = pred
            st.session_state["last_target"] = target2
            st.session_state["inputs"] = input_data.iloc[0].to_dict()

    # ============================================================
    # 4) PDF 리포트 탭
    # ============================================================
    with tab4:
        st.header("📑 예측 PDF 리포트 생성")

        if "last_pred" not in st.session_state:
            st.info("먼저 점수 예측을 진행해주세요.")
        else:
            pred = st.session_state["last_pred"]
            target = st.session_state["last_target"]
            inputs = st.session_state["inputs"]

            file_path = generate_pdf(pred, target, inputs)

            with open(file_path, "rb") as pdf:
                st.download_button(
                    "📥 PDF 리포트 다운로드",
                    data=pdf,
                    file_name="prediction_report.pdf",
                    mime="application/pdf"
                )


if __name__ == "__main__":
    main()
