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
# 1) CSV 파일 불러오기 + 컬럼명 통일
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
            "CSV 파일을 찾을 수 없습니다. "
            "app.py와 같은 위치에 CSV 파일을 넣어주세요."
        )
        st.stop()

    # 컬럼명 통일
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
# 3) Session State 초기화
# ===============================

def init_session_state(df):
    defaults = {
        "gender": df["gender"].unique()[0],
        "race_ethnicity": df["race_ethnicity"].unique()[0],
        "parental_level_of_education": df["parental_level_of_education"].unique()[0],
        "lunch": df["lunch"].unique()[0],
        "test_preparation_course": df["test_preparation_course"].unique()[0],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ===============================
# 4) Streamlit UI
# ===============================
def main():
    st.title("📊 안정적 학생 성적 예측기 (Session-State 적용)")
    st.write("값이 초기화되지 않고, CSV만 넣으면 자동으로 동작합니다.")

    # 1) 데이터 불러오기
    df = load_dataset()

    # 2) Session State 초기화
    init_session_state(df)

    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    # 예측할 대상 선택
    target_col = st.selectbox(
        "예측할 과목을 선택하세요",
        ["math_score", "reading_score", "writing_score"],
        key="target_col"
    )

    # 모델 학습 버튼
    if st.button("모델 학습 및 성능 평가"):
        model, rmse, r2 = train_single_target(df, target_col)

        st.success("모델 학습 완료!")
        st.write(f"**RMSE:** {rmse:.3f}")
        st.write(f"**R²:** {r2:.3f}")

        st.subheader("값을 선택해서 점수 예측하기")

        # Session-State 기반 선택 UI
        st.session_state.gender = st.selectbox(
            "Gender",
            sorted(df["gender"].unique()),
            index=sorted(df["gender"].unique()).index(st.session_state.gender),
            key="gender"
        )

        st.session_state.race_ethnicity = st.selectbox(
            "Race/Ethnicity",
            sorted(df["race_ethnicity"].unique()),
            index=sorted(df["race_ethnicity"].unique()).index(st.session_state.race_ethnicity),
            key="race_ethnicity"
        )

        st.session_state.parental_level_of_education = st.selectbox(
            "Parent Education",
            sorted(df["parental_level_of_education"].unique()),
            index=sorted(df["parental_level_of_education"].unique()).index(st.session_state.parental_level_of_education),
            key="parental_level_of_education"
        )

        st.session_state.lunch = st.selectbox(
            "Lunch",
            sorted(df["lunch"].unique()),
            index=sorted(df["lunch"].unique()).index(st.session_state.lunch),
            key="lunch"
        )

        st.session_state.test_preparation_course = st.selectbox(
            "Test Preparation",
            sorted(df["test_preparation_course"].unique()),
            index=sorted(df["test_preparation_course"].unique()).index(st.session_state.test_preparation_course),
            key="test_preparation_course"
        )

        if st.button("점수 예측하기"):
            input_df = pd.DataFrame([{
                "gender": st.session_state.gender,
                "race_ethnicity": st.session_state.race_ethnicity,
                "parental_level_of_education": st.session_state.parental_level_of_education,
                "lunch": st.session_state.lunch,
                "test_preparation_course": st.session_state.test_preparation_course,
            }])

            pred = model.predict(input_df)[0]
            st.success(f"예측된 {target_col}: **{pred:.2f} 점**")


if __name__ == "__main__":
    main()
