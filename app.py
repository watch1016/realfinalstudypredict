import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


# ============================================
# 1) CSV 파일 로드 함수
# ============================================
def load_data():
    candidates = [
        "StudentsPerformance.csv",
        "StudentsPerformance_clean.csv",
        "StudentsPerformance_1000rows_synthetic.csv",
        "students.csv"
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
        st.error("❌ CSV 파일을 찾을 수 없습니다. app.py와 동일 경로에 CSV 파일을 두세요.")
        st.stop()

    rename_map = {
        "race/ethnicity": "race_ethnicity",
        "parental level of education": "parental_level_of_education",
        "test preparation course": "test_preparation_course",
        "math score": "math_score",
        "reading score": "reading_score",
        "writing score": "writing_score",
    }

    df = df.rename(columns=rename_map)
    return df


# ============================================
# 2) 모델 학습
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
        n_estimators=400,
        random_state=42,
        max_depth=None,
        n_jobs=-1
    )

    pipe = Pipeline([
        ("prep", transformer),
        ("model", model)
    ])

    pipe.fit(X, y)

    preds = pipe.predict(X)
    rmse = np.sqrt(mean_squared_error(y, preds))
    r2 = r2_score(y, preds)

    return pipe, rmse, r2


# ============================================
# 3) Streamlit UI
# ============================================
def main():

    st.set_page_config(
        page_title="학생 성적 예측 시스템",
        page_icon="📈",
        layout="centered"
    )

    # Header
    st.markdown("""
        <h1 style="text-align:center; color:#222;">
            📈 학생 성적 예측 시스템
        </h1>
        <p style="text-align:center; color:#555; font-size:17px;">
            머신러닝 기반 예측 알고리즘을 활용하여<br>
            입력하신 학생 정보에 대한 **신뢰성 있는 성적 예측 결과를 제공합니다.**
        </p>
        <hr style="margin-top:20px; margin-bottom:30px;">
    """, unsafe_allow_html=True)

    df = load_data()

    # Data preview (professional style)
    st.markdown("### 📄 Dataset Overview")
    st.caption("모델이 학습하는 데이터셋의 첫 5행입니다.")
    st.dataframe(df.head(), use_container_width=True)
    st.markdown("---")

    # Model training
    st.markdown("### ⚙️ Model Configuration & Training")

    target = st.selectbox(
        "예측할 점수를 선택하세요:",
        ["math_score", "reading_score", "writing_score"]
    )

    st.info("모델을 학습하고 성능을 평가합니다. 잠시만 기다려주세요...")

    model, rmse, r2 = train_model(df, target)

    st.success("모델 학습 완료!")

    colA, colB = st.columns(2)
    with colA:
        st.metric("RMSE (Training)", f"{rmse:.2f}")
    with colB:
        st.metric("R² Score (Training)", f"{r2:.3f}")

    st.markdown("""
        <p style="color:#777; font-size:14px;">
        ※ RMSE는 낮을수록 좋고, R²는 1에 가까울수록 예측 성능이 좋습니다.
        </p>
        <hr>
    """, unsafe_allow_html=True)

    # Feature input form
    st.markdown("### 📝 Student Profile Input")

    with st.form("predict_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", sorted(df["gender"].unique()))
            lunch = st.selectbox("Lunch Type", sorted(df["lunch"].unique()))
            race = st.selectbox("Race/Ethnicity", sorted(df["race_ethnicity"].unique()))

        with col2:
            pedu = st.selectbox("Parental Education", sorted(df["parental_level_of_education"].unique()))
            prep = st.selectbox("Test Preparation Course", sorted(df["test_preparation_course"].unique()))

        submitted = st.form_submit_button("🔍 Predict Score")

    # Prediction output
    if submitted:

        input_df = pd.DataFrame([{
            "gender": gender,
            "race_ethnicity": race,
            "parental_level_of_education": pedu,
            "lunch": lunch,
            "test_preparation_course": prep,
        }])

        pred = model.predict(input_df)[0]

        st.markdown("""
            <div style="
                padding: 25px; 
                border-radius: 10px; 
                background: #f7f9fc;
                border: 1px solid #d9e1ec;
                margin-top: 20px;">
                <h3 style="color:#1a3c6e;">📘 예측 결과 보고서</h3>
                <p style="font-size:16px; color:#333;">
                    아래는 입력하신 학생 정보를 기반으로 생성된 성적 예측 결과입니다.
                </p>
            </div>
        """, unsafe_allow_html=True)

        st.success(f"🎯 예측된 {target} 점수: **{pred:.2f}점**")

        st.caption("본 예측 결과는 통계적 모델을 기반으로 하며 절대적인 판단 기준이 아닙니다.")


if __name__ == "__main__":
    main()
