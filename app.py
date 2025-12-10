import pickle
import pandas as pd
import sklearn
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.base import BaseEstimator, TransformerMixin

class CustomTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()

        if 'model' not in X_copy.columns:
            parts = str(name).split()

            def extract(model, pos=0):
                if pd.isna(model):
                    return 'unknown'

                return parts[pos].strip().lower() if len(parts) > 0 else 'unknown'

            X_copy['brand'] = X_copy['name'].apply(lambda x: extract(x, 0))
            X_copy['model'] = X_copy['name'].apply(lambda x: extract(x, 1))

        if 'max_power' in X_copy.columns and 'engine' in X_copy.columns:
            engine_safe = X_copy['engine'].replace(0, X_copy['engine'].median())

            X_copy['max_power/engine'] = X_copy['max_power'] / engine_safe
            X_copy['max_power/engine'] = X_copy['max_power/engine'].fillna(0)

        return X_copy

st.set_page_config(
    page_title="Сколько стоит твоя машина?",
    page_icon="🚗",
    layout="centered",
)

st.markdown(f"""
    <style>
        section[data-testid="stSidebar"] {{
            width: 400px !important;
            padding: 20px;
        }}
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model():
    with open('model/car_prediction_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('model/features.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    return model, feature_names

st.title('Сколько стоит твоя машина?')

if 'page' not in st.session_state:
    st.session_state.page = "eda"

with st.sidebar:
    st.title("🚗 Навигация")

    if st.button("📊 EDA", use_container_width=True):
        st.session_state.page = "eda"
        st.rerun()

    if st.button("🎯 Предсказание", use_container_width=True):
        st.session_state.page = "prediction"
        st.rerun()

    if st.button("📈 Модель", use_container_width=True):
        st.session_state.page = "model"
        st.rerun()

    st.markdown("---")
    st.info(f"Текущая страница: {st.session_state.page}")

if st.session_state.page == "eda":
    st.header("📊 Анализ данных")

    uploaded_file = st.file_uploader("Загрузите CSV файл c данными", type=["csv"])

    if uploaded_file is None:
        st.info("👈 Загрузите CSV файл для начала работы")
        st.stop()
    else:
        df = pd.read_csv(uploaded_file)

        st.subheader("📊 Результаты")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Всего машин", len(df))
        with col2:
            st.metric("Количество пропусков", df.isna().sum().sum())

        st.subheader("🔍 Предпросмотр данных")
        st.dataframe(df.head(10))

        st.subheader("Типы данных")

        dtype_df = pd.DataFrame({
            'Колонка': df.columns,
            'Тип': df.dtypes,
            'Уникальных': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtype_df)

        st.subheader("📈 Визуализации")
        fig = px.histogram(df, x='selling_price', nbins=50, title="💸 Распределение цен")
        fig.update_layout(xaxis_title='Стоимость машин', yaxis_title='Количество')
        st.plotly_chart(fig, width='content')

        fig = px.histogram(df, x='year', title="🗓️ Распределение по годам")
        fig.update_layout(xaxis_title='Год', yaxis_title='Количество')
        st.plotly_chart(fig, width='content')

        fig = px.histogram(df, x='max_power', title="💪️ Распределение по мощностям")
        fig.update_layout(xaxis_title='Мощность', yaxis_title='Количество')
        st.plotly_chart(fig, width='content')

        fig = px.scatter(df, x='year', y='selling_price', trendline="ols", trendline_color_override="red", title="Зависимость цены от года выпуска")
        fig.update_layout(xaxis_title='Год', yaxis_title='Стоимость')
        st.plotly_chart(fig, width='content')

        fig = px.box(df, x='seller_type', y='selling_price', color='owner', title="Кто продает и количество владельцев")
        fig.update_layout(xaxis_title='Продавец', yaxis_title='Стоимость')
        st.plotly_chart(fig, width='content')

        st.subheader("📉 Корреляционная матрица")

        corr = df.corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=True, height=800, width=1200)
        st.plotly_chart(fig, width='content')


elif st.session_state.page == "prediction":
    st.header("🎯 Предсказание стоимости автомобиля")

    with st.form("manual_input"):
        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Марка (name)", "Toyota")
            model = st.text_input("Модель (model)", "Camry")
            year = st.number_input("Год выпуска (year)", 1900, 2025, 2020)
            km_driven = st.number_input("Пробег (Km_driven)", 0, 1000000, 50000)
            seats = st.number_input("Количество мест", 0, 20, 5)

        with col2:
            mileage = st.number_input("Расход топлива (mileage)", 0.0, 50.0, 15.5)
            engine = st.number_input("Объем двигателя (engine)", 0, 10000, 3500)
            max_power = st.number_input("Мощность (max_power)", 0.0, 500.0, 180.0)
            torque = st.number_input("Крутящий момент (Nm)", 0.0, 500.0, 100.0)
            max_torque_rpm = st.number_input("Максимальный крутящий момент в минуту (RPM)", 0.0, 10000.0, 1300.0)

        submitted = st.form_submit_button("🎯 Предсказать цену")

        cat_features = ['name', 'model', 'seats']
        num_features = ['year', 'km_driven', 'mileage', 'torque', 'max_torque_rpm', 'max_power/engine']

        if submitted:
            max_power_to_engine = max_power / engine

            input_data = pd.DataFrame({
                'name': [name],
                'model': [model],
                'year': [year],
                'km_driven': [km_driven],
                'mileage': [mileage],
                'torque': [torque],
                'seats': [seats],
                'max_torque_rpm': [max_torque_rpm],
                'max_power/engine': [max_power_to_engine],
            })

            (model, features) = load_model()

            prediction = model.predict(input_data)[0]
            st.success(f"💰 Предсказанная стоимость: {prediction:,.2f}".replace(',', ' '))


elif st.session_state.page == "model":
    st.header("📊 Визуализация весов модели")

    (model, features) = load_model()

    best_pipeline = model.best_estimator_
    ridge_model = best_pipeline.named_steps['ridge']
    preprocessor = best_pipeline.named_steps['preprocessor']

    # Коэффициенты
    coefficients = ridge_model.coef_

    try:
        feature_names = preprocessor.get_feature_names_out()

        all_feature = pd.DataFrame({'feature': feature_names, 'coefficient': coefficients, 'abs': np.abs(coefficients)})

        num_coeff = []
        num_feature = []

        for name, coef in zip(feature_names, coefficients):
            if name.startswith('num__') or name.startswith('poly__'):
                num_coeff.append(coef)
                num_feature.append(name)
    except:
        feature_names = [f"Feature {i + 1}" for i in range(len(coefficients))]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Всего признаков", len(feature_names))
    with col2:
        st.metric("Вещественных", len(num_coeff))
    with col3:
        st.metric("Категориальных", len(feature_names) - len(num_coeff))

    coef_df = pd.DataFrame({
        'feature': num_feature,
        'coef': num_coeff,
        'abs_coef': np.abs(num_coeff)
    }).sort_values('abs_coef', ascending=False)

    st.subheader("Таблица коэффициентов вещественных признаков")
    st.dataframe(coef_df.style.format({'coefficient': '{:.6f}'}))

    st.subheader("Распределение вещественных признаков по значимости")

    fig = px.bar(
        coef_df,
        x='abs_coef',
        y='feature',
        orientation='h',
        color='coef',
        color_continuous_scale='RdBu',
        labels={'abs_coef': 'Абсолютный Коэффициент', 'feature': 'Признак'}
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, width='content')

    st.subheader("Все признаки модели")

    fig = px.scatter(
        all_feature,
        x='feature',
        y='abs',
        color='coefficient',
        size='abs',
        hover_data=['feature', 'coefficient'],
        labels={'coefficient': 'Значение коэффициента', 'feature': 'Признак'}
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(xaxis_tickangle=-90)
    st.plotly_chart(fig, width='content')

    st.subheader("Только вещественные")

    fig = px.scatter(
        coef_df,
        x='feature',
        y='abs_coef',
        color='coef',
        size='abs_coef',
        hover_data=['feature', 'coef'],
        labels={'coef': 'Значение коэффициента', 'feature': 'Признак'}
    )
    fig.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig.update_layout(xaxis_tickangle=-90)
    st.plotly_chart(fig, width='content')

    st.subheader("Распределение значений коэффициентов")

    fig4 = px.histogram(
        coef_df,
        x='coef',
        nbins=30,
        title="Гистограмма распределения коэффициентов",
        labels={'coefficient': 'Значение коэффициента'},
        marginal="box"
    )
    st.plotly_chart(fig4, width='content')

    st.subheader("Статистика")

    stats_df = pd.DataFrame({
        'Метрика': ['Среднее', 'Станд. отклонение', 'Медиана', 'Мин', 'Макс', 'Сумма abs'],
        'Значение': [
            coefficients.mean(),
            coefficients.std(),
            np.median(coefficients),
            coefficients.min(),
            coefficients.max(),
            np.abs(coefficients).sum()
        ]
    })
    st.dataframe(stats_df.style.format({'Значение': '{:.2f}'}))

