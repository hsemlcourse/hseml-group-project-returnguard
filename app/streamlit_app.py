import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="ReturnGuard — Доход домохозяйства", layout="centered")
st.title("Предсказание дохода домохозяйства")
st.caption("Тульская область, 2023 — модели GBM (классификация) и XGBoost (регрессия)")

with st.form("predict_form"):
    st.subheader("Характеристики домохозяйства")

    col1, col2 = st.columns(2)
    with col1:
        chlico = st.number_input("Число лиц (chlico)", min_value=1, max_value=20, value=3)
        rasress = st.number_input("Располагаемые ресурсы, руб. (rasress)", min_value=0, value=200_000, step=1_000)
        potras = st.number_input("Расход на потребление, руб. (potras)", min_value=0, value=150_000, step=1_000)
    with col2:
        chdet = st.number_input("Число детей (chdet)", min_value=0, max_value=15, value=1)
        rasq = st.number_input("Среднедушевые ресурсы, руб. (rasq)", min_value=0, value=70_000, step=1_000)

    submitted = st.form_submit_button("Предсказать", use_container_width=True)

if submitted:
    payload = {
        "chlico": chlico,
        "chdet": chdet,
        "rasress": rasress,
        "rasq": rasq,
        "potras": potras,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.divider()
        st.subheader("Результат")

        reg_col, clf_col = st.columns(2)
        with reg_col:
            st.metric(
                label="Среднедушевой доход (XGBoost)",
                value=f"{result['income_predicted']:,.0f} руб.",
            )
        with clf_col:
            label = "Выше Q3 (>151 400 руб.)" if result["above_q3"] else "Ниже Q3"
            st.metric(
                label="Группа дохода (GBM)",
                value=label,
                delta=f"вероятность {result['above_q3_probability']:.1%}",
                delta_color="normal",
            )

        st.caption(f"Порог Q3: {result['q3_threshold']:,.0f} руб.")

    except requests.exceptions.ConnectionError:
        st.error("Не удалось подключиться к API. Убедитесь, что сервис запущен.")
    except Exception as exc:
        st.error(f"Ошибка: {exc}")
