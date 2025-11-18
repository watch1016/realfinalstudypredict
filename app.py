import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


# ===============================
# 1) CSV 파일 로드 + 컬럼명 통일
# ===============================
def load_dataset():
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
            st.success(f"Detected CSV file: {f}")
            break
        except:
            pass

    if df is None:
        st.error(
            "CSV 파일을 찾을 수 없습니다.\n"
            "app.py와 같은 폴더에 CSV 파일을 넣어주세요."
        )
        st.stop()

    rename_map = {
        "race/ethnicity": "race_ethnicity",
        "race ethnicity": "race_ethnicity",
        "race_ethnicity": "race_ethnicity",

        "parental level of education": "parental_level_of_education",
        "parental_level_of_education": "parental_level_of_education",

        "test preparation course": "test_preparation_course",
        "test_preparation_course": "test_preparation_course",

        "math score": "math_score",
        "math_score": "math_score",

        "reading score": "reading_score",
        "reading_score": "reading_score",

        "writing score": "writing_score",
        "writing_score": "writing_score",
    }

    df = df.rename(columns=rename_map)
    return df


# ===============================
# 2) 모델 학습 함수
# ===============================
def train_single_target(df, target):
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
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), features)],
        remainder="drop",
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return pipe, rmse, r2


# ===============================
# 3) Streamlit UI
# ===============================
def main():
    st.title("📊 학생 성적 예측기 — 오류 없는 안정판")
    df = load_dataset()

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    st.subheader("모델 학습")

    target_col = st.selectbox(
        "예측할 과목 선택",
        ["math_score", "reading_score", "writing_score"]
    )

    if st.button("모델 학습하기"):
        model, rmse, r2 = train_single_target(df, target_col)

        st.success("모델 학습 완료!")
        st.write(f"RMSE: **{rmse:.3f}**")
        st.write(f"R²: **{r2:.3f}**")

        st.subheader("예측하기")

        gender = st.selectbox("Gender", sorted(df["gender"].unique()))
        race = st.selectbox("Race/Ethnicity", sorted(df["race_ethnicity"].unique()))
        pedu = st.selectbox("Parent Education", sorted(df["parental_level_of_education"].unique()))
        lunch = st.selectbox("Lunch", sorted(df["lunch"].unique()))
        prep = st.selectbox("Test Preparation", sorted(df["test_preparation_course"].unique()))

        if st.button("점수 예측 실행"):
            input_df = pd.DataFrame([{
                "gender": gender,
                "race_ethnicity": race,
                "parental_level_of_education": pedu,
                "lunch": lunch,
                "test_preparation_course": prep,
            }])

            pred = model.predict(input_df)[0]
            st.success(f"{target_col} 예측 점수: **{pred:.2f}**")


if __name__ == "__main__":
    main()
