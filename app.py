import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================
# CSV 로드 함수
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
        st.error("❌ CSV 파일을 찾을 수 없습니다. app.py와 같은 위치에 두세요.")
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
        "test_preparation_course"
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
def get_feature_importance(model):
    ohe = model.named_steps["prep"].named_transformers_["cat"]
    feature_names = ohe.get_feature_names_out()
    importances = model.named_steps["model"].feature_importances_

    return pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)


# ============================================
# Streamlit UI 시작
# ============================================
def main():

    st.set_page_config(page_title="학생 성적 예측 시스템", page_icon="📈")

    st.markdown("""
        <h1 style="text-align:center; color:#222;">📈 학생 성적 예측 시스템</h1>
        <p style="text-align:center; color:#555;">
            전문가 수준의 머신러닝 분석 · 변수 영향 분석 · 성능 평가 · 점수 예측 서비스
        </p>
        <hr>
    """, unsafe_allow_html=True)

    df = load_data()

    # 탭 UI
    tab1, tab2, tab3 = st.tabs([
        "📊 데이터 분석",
        "⚙️ 모델 학습 & 성능",
        "📝 점수 예측"
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
        for col, score in zip([col1, col2, col3],
                              ["math_score", "reading_score", "writing_score"]):
            with col:
                fig, ax = plt.subplots()
                ax.hist(df[score], bins=20, color="#4A90E2")
                ax.set_title(f"{score} Distribution")
                st.pyplot(fig)

        st.subheader("3) 부모 학력별 평균 점수")
        st.bar_chart(
            df.groupby("parental_level_of_education")[
                ["math_score", "reading_score", "writing_score"]
            ].mean()
        )

    # ============================================================
    # 2) 모델 학습 & 성능
    # ============================================================
    with tab2:
        st.header("⚙️ 모델 학습 및 성능 평가")

        target = st.selectbox("예측할 점수", ["math_score", "reading_score", "writing_score"])

        model, rmse, mae, r2 = train_model(df, target)

        st.subheader("📈 성능 지표")
        colA, colB, colC = st.columns(3)
        colA.metric("RMSE", f"{rmse:.2f}")
        colB.metric("MAE", f"{mae:.2f}")
        colC.metric("R² Score", f"{r2:.3f}")

        st.markdown("### 🔍 Feature Importance")
        fi = get_feature_importance(model)
        st.dataframe(fi)

        fig, ax = plt.subplots()
        ax.barh(fi["feature"], fi["importance"], color="#1A73E8")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

    # ============================================================
    # 3) 점수 예측
    # ============================================================
    with tab3:
        st.header("📝 학생 점수 예측")

        with st.form("predict_form"):
            gender = st.selectbox("Gender", df["gender"].unique())
            race = st.selectbox("Race/Ethnicity", df["race_ethnicity"].unique())
            pedu = st.selectbox("Parental Education", df["parental_level_of_education"].unique())
            lunch = st.selectbox("Lunch", df["lunch"].unique())
            prep = st.selectbox("Test Preparation", df["test_preparation_course"].unique())
            target2 = st.selectbox("예측 대상 점수", ["math_score", "reading_score", "writing_score"])

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

            st.success(f"📘 예측된 {target2}: **{pred:.2f}점**")


if __name__ == "__main__":
    main()
