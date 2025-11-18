import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline


# ============================================
# 1) CSV 파일 불러오기
# ============================================
def load_data():
    possible_files = [
        "StudentsPerformance.csv",
        "StudentsPerformance_clean.csv",
        "StudentsPerformance_1000rows_synthetic.csv",
        "students.csv",
    ]

    df = None
    for f in possible_files:
        try:
            df = pd.read_csv(f)
            st.success(f"CSV 파일 로드 성공: {f}")
            break
        except:
            pass

    if df is None:
        st.error("⛔ CSV 파일을 찾을 수 없습니다. app.py와 같은 폴더에 넣어주세요.")
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
        "test_preparation_course"
    ]

    X = df[features]
    y = df[target]

    preprocessor = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), features)]
    )

    model = RandomForestRegressor(n_estimators=300, random_state=42)

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
    st.title("📊 학생 점수 예측기 (CSV 기반 머신러닝)")

    st.write("""
    이 앱은 **학생들 데이터(CSV)** 를 기반으로  
    **모델을 학습한 뒤**,  
    사용자가 입력한 조건에 따라  
    **해당 학생의 예상 점수**를 예측합니다.
    """)

    df = load_data()

    st.subheader("🔎 데이터 미리보기")
    st.dataframe(df.head())

    st.subheader("🎯 예측할 과목 선택")
    target = st.selectbox(
        "어떤 점수를 예측할까요?",
        ["math_score", "reading_score", "writing_score"]
    )

    # 모델 학습
    st.info("모델을 학습 중입니다...")
    model = train_model(df, target)
    st.success("모델 학습 완료!")

    st.subheader("📝 학생 정보를 입력하세요")

    gender = st.selectbox("Gender", sorted(df["gender"].unique()))
    race = st.selectbox("Race/Ethnicity", sorted(df["race_ethnicity"].unique()))
    pedu = st.selectbox("Parental Education", sorted(df["parental_level_of_education"].unique()))
    lunch = st.selectbox("Lunch", sorted(df["lunch"].unique()))
    prep = st.selectbox("Test Preparation", sorted(df["test_preparation_course"].unique()))

    if st.button("예측하기"):
        input_df = pd.DataFrame([{
            "gender": gender,
            "race_ethnicity": race,
            "parental_level_of_education": pedu,
            "lunch": lunch,
            "test_preparation_course": prep,
        }])

        pred = model.predict(input_df)[0]

        st.success(f"📘 예측된 {target} 점수는 **{pred:.2f}점** 입니다!")


if __name__ == "__main__":
    main()
