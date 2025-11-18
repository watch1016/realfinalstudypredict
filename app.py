import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ============================================
# 1) CSV 파일 로드
# ============================================
def load_data():
    files = [
        "StudentsPerformance.csv",
        "StudentsPerformance_clean.csv",
        "StudentsPerformance_1000rows_synthetic.csv",
        "students.csv",
    ]

    df = None
    for f in files:
        try:
            df = pd.read_csv(f)
            st.sidebar.success(f"📁 Loaded: {f}")
            break
        except:
            pass

    if df is None:
        st.error("❌ CSV 파일을 찾을 수 없습니다. app.py와 같은 폴더에 넣어주세요.")
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
# 2) 모델 학습 함수
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

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), features)]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42
    )

    pipe = Pipeline([
        ("prep", preprocessor),
        ("model", model)
    ])

    pipe.fit(X, y)
    return pipe


# ============================================
# 3) Streamlit UI
# ============================================
def main():

    st.set_page_config(page_title="학생 점수 예측기", page_icon="📘", layout="centered")

    st.markdown("""
        <h1 style='text-align:center; color:#333;'>📘 학생 성적 예측기</h1>
        <p style='text-align:center; font-size:17px; color:#555;'>
            머신러닝(Random Forest)을 사용하여<br>
            <b>학생들의 성적을 예측하는 앱</b>입니다.
        </p>
    """, unsafe_allow_html=True)

    df = load_data()

    # -----------------------------
    # 데이터 미리보기 카드
    # -----------------------------
    st.markdown("### 📊 데이터 미리보기")
    with st.container():
        st.dataframe(df.head(), use_container_width=True)

    st.markdown("---")

    # -----------------------------
    # 예측 섹션
    # -----------------------------
    st.markdown("### 🎯 예측할 과목 선택")

    target = st.selectbox(
        "어떤 점수를 예측할까요?",
        ["math_score", "reading_score", "writing_score"]
    )

    st.info("📚 선택한 CSV 데이터를 기반으로 모델을 학습합니다.")
    model = train_model(df, target)
    st.success("✅ 모델 학습 완료!")

    st.markdown("---")
    st.markdown("### 📝 학생 정보 입력")

    # -----------------------------
    # 입력 폼(폼은 자동 rerun 안 됨 → 안정적)
    # -----------------------------
    with st.form("predict_form"):

        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("👤 Gender", sorted(df["gender"].unique()))
            race = st.selectbox("🌎 Race/Ethnicity", sorted(df["race_ethnicity"].unique()))
            lunch = st.selectbox("🥪 Lunch Type", sorted(df["lunch"].unique()))

        with col2:
            pedu = st.selectbox("🎓 Parental Education", sorted(df["parental_level_of_education"].unique()))
            prep = st.selectbox("📘 Test Preparation", sorted(df["test_preparation_course"].unique()))

        submitted = st.form_submit_button("📌 점수 예측하기")

    # -----------------------------
    # 예측 실행
    # -----------------------------
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
        <div style='padding:20px; background:#f0f7ff; border-radius:10px; border:1px solid #cce0ff;'>
            <h3 style='color:#0066cc;'>📘 예측 결과</h3>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"🎉 예측된 **{target}** 점수는 **{pred:.2f}점** 입니다!")


if __name__ == "__main__":
    main()
