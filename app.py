import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error


# -------------------------------------------
# 1) CSV 파일 자동 로딩 + 컬럼명 자동 정규화
# -------------------------------------------
def load_dataset():
    possible_files = [
        "StudentsPerformance.csv",
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
            "app.py와 같은 위치에 'StudentsPerformance.csv' 또는 synthetic CSV를 올려주세요."
        )
        st.stop()

    # 컬럼명 매핑 — 어떤 형태든 표준 이름으로 통일
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

    required_cols = [
        "gender",
        "race_ethnicity",
        "parental_level_of_education",
        "lunch",
        "test_preparation_course",
        "math_score",
        "reading_score",
        "writing_score",
    ]

    for col in required_cols:
        if col not in df.columns:
            st.error(f"필수 컬럼 누락: {col}")
            st.write("현재 CSV 컬럼:", list(df.columns))
            st.stop()

    return df


# -------------------------------------------
# 2) 모델 학습 함수 — RMSE 수동 계산으로 모든 버전 호환
# -------------------------------------------
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

    # 범주형 변수 인코딩
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

    # RMSE 계산 (squared=False 사용 안 함 — 모든 sklearn 버전 호환)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    return pipe, rmse, r2


# -------------------------------------------
# Streamlit UI
# -------------------------------------------
def main():
    st.title("📊 학생 성적 예측기 (Random Forest) — 완전한 안정 버전")
    st.write("CSV 파일만 넣으면 자동으로 돌아가는 안전한 버전입니다.")

    df = load_dataset()

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    target_col = st.selectbox(
        "예측할 과목을 선택하세요",
        ["math_score", "reading_score", "writing_score"]
    )

    if st.button("모델 학습 & 성능 평가"):
        st.info("모델 학습 중...")

        model, rmse, r2 = train_single_target(df, target_col)

        st.success("학습 완료!")
        st.write(f"**RMSE:** {rmse:.3f}")
        st.write(f"**R²:** {r2:.3f}")

        st.subheader("입력값으로 점수 예측하기")

        if "gender" not in st.session_state:
    st.session_state.gender = sorted(df["gender"].unique())[0]

st.session_state.gender = st.selectbox(
    "Gender",
    sorted(df["gender"].unique()),
    index=sorted(df["gender"].unique()).index(st.session_state.gender),
    key="gender"
)

        race = st.selectbox("Race/Ethnicity", sorted(df["race_ethnicity"].unique()))
        pedu = st.selectbox("Parent Education", sorted(df["parental_level_of_education"].unique()))
        lunch = st.selectbox("Lunch", sorted(df["lunch"].unique()))
        prep = st.selectbox("Test Preparation", sorted(df["test_preparation_course"].unique()))

        if st.button("점수 예측하기"):
            input_df = pd.DataFrame([{
                "gender": gender,
                "race_ethnicity": race,
                "parental_level_of_education": pedu,
                "lunch": lunch,
                "test_preparation_course": prep,
            }])

            pred_score = model.predict(input_df)[0]
            st.success(f"예측된 {target_col}: **{pred_score:.2f} 점**")


if __name__ == "__main__":
    main()
