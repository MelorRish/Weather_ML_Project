import streamlit as st
import pandas as pd
import requests
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ==========================================
# 1. НАЛАШТУВАННЯ СТОРІНКИ 
# ==========================================
st.set_page_config(page_title="AI Прогноз Погоди", page_icon="🌦️", layout="wide")

# Ініціалізація змінних у пам'яті
if 't_max' not in st.session_state: st.session_state['t_max'] = 15.0
if 't_min' not in st.session_state: st.session_state['t_min'] = 5.0
if 'w_max' not in st.session_state: st.session_state['w_max'] = 10.0
if 'p_sum' not in st.session_state: st.session_state['p_sum'] = 0.0
if 'data_loaded' not in st.session_state: st.session_state['data_loaded'] = False 

# ==========================================
# 2. БІЧНА ПАНЕЛЬ (SIDEBAR)
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1146/1146869.png", width=100)
    st.title("⚙️ Налаштування")
    
    # --- НАВІГАЦІЙНЕ МЕНЮ ЗЛІВА ---
    st.markdown("---")
    st.subheader("🧭 Навігація")
    page = st.radio("Оберіть сторінку:", ["🔮 Прогноз на завтра", "📊 Аналітика та Дані"])
    
    st.markdown("---")
    
    # --- РОЗДІЛ 1: БАЗА ДАНИХ ---
    st.subheader("1. База даних ☁️")
    if st.button("⬇️ Отримати дані", use_container_width=True):
        with st.spinner("З'єднання з Open-Meteo..."):
            url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": 50.45, "longitude": 30.52,
                "start_date": "2025-01-01", "end_date": "2026-03-11",
                "daily": ["precipitation_sum", "rain_sum", "temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max"],
                "timezone": "auto"
            }
            response = requests.get(url, params=params)
            if response.status_code == 200:
                df = pd.DataFrame(response.json()["daily"])
                df.to_csv("weather_daily.csv", index=False)
                st.session_state['data_loaded'] = True 
            else:
                st.error("❌ Помилка API")

    if st.session_state['data_loaded'] and os.path.exists("weather_daily.csv"):
        try:
            df_info = pd.read_csv("weather_daily.csv")
            st.success("✅ Дані готові до роботи")
            st.caption(f"**Кількість записів:** {len(df_info)} днів")
            st.caption(f"**Період:** {df_info['time'].iloc[0]} ➡ {df_info['time'].iloc[-1]}")
        except:
            st.error("Файл пошкоджено. Завантажте дані ще раз.")
    else:
        st.info("ℹ️ Дані ще не завантажено.")

    st.markdown("---")
    
    # --- РОЗДІЛ 2: ШТУЧНИЙ ІНТЕЛЕКТ ---
    st.subheader("2. Штучний інтелект 🧠")
    if st.button("🚀 Навчити модель", use_container_width=True):
        if not os.path.exists("weather_daily.csv"):
            st.warning("⚠️ Спочатку завантажте дані!")
        else:
            with st.spinner("Модель навчається..."):
                df = pd.read_csv("weather_daily.csv").dropna()
                df['target_tomorrow'] = (df['precipitation_sum'].shift(-1) > 0).astype(int)
                df = df.dropna()
                
                features = ['temperature_2m_max', 'temperature_2m_min', 'wind_speed_10m_max', 'precipitation_sum']
                X, y = df[features], df['target_tomorrow']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                st.session_state['trained_model'] = model
                st.session_state['features_list'] = features
                st.session_state['accuracy'] = accuracy_score(y_test, model.predict(X_test))
                
                raw_report = classification_report(y_test, model.predict(X_test), target_names=["Немає", "Є опади"])
                ukr_report = raw_report.replace("precision", "Влучність")
                ukr_report = ukr_report.replace("recall", "Повнота  ")
                ukr_report = ukr_report.replace("f1-score", "F1-оцінка")
                ukr_report = ukr_report.replace("support", "Кількість")
                ukr_report = ukr_report.replace("accuracy", "Заг. точність")
                ukr_report = ukr_report.replace("macro avg", "Макро-сер.   ")
                ukr_report = ukr_report.replace("weighted avg", "Зважене сер. ")
                st.session_state['report'] = ukr_report
                
                st.success("✅ Готово!")

    if 'accuracy' in st.session_state:
        st.markdown("---")
        st.metric(label="🎯 Точність моделі", value=f"{st.session_state['accuracy']*100:.1f}%")

# ==========================================
# 3. ГОЛОВНИЙ ЕКРАН
# ==========================================
st.title("🌦️ Інтелектуальний прогноз опадів: Київ")
st.markdown("*Ваш персональний метеоролог на базі машинного навчання.*")
st.markdown("---")

# ЛОГІКА ПЕРЕМИКАННЯ СТОРІНОК
if page == "🔮 Прогноз на завтра":
    
    st.markdown("### 🌤️ Погодні умови за сьогодні")
    st.write("Щоб зробити прогноз на завтра, нам потрібні дані за сьогодні. Оберіть зручний для вас спосіб:")
    
    st.info("🤖 **СПОСІБ 1: Автоматично (Рекомендовано)** - програма сама знайде поточну погоду в Києві.")
    
    if st.button("🔄 Отримати поточну погоду з інтернету"):
        with st.spinner("Отримання поточних даних для Києва..."):
            forecast_url = "https://api.open-meteo.com/v1/forecast"
            forecast_params = {
                "latitude": 50.45, "longitude": 30.52,
                "daily": ["temperature_2m_max", "temperature_2m_min", "wind_speed_10m_max", "precipitation_sum"],
                "timezone": "auto",
                "forecast_days": 1 
            }
            res = requests.get(forecast_url, params=forecast_params)
            
            if res.status_code == 200:
                current_data = res.json()["daily"]
                st.session_state['t_max'] = float(current_data["temperature_2m_max"][0])
                st.session_state['t_min'] = float(current_data["temperature_2m_min"][0])
                st.session_state['w_max'] = float(current_data["wind_speed_10m_max"][0])
                st.session_state['p_sum'] = float(current_data["precipitation_sum"][0])
                st.success("✅ Дані успішно завантажено! Вони з'явилися у віконцях нижче.")
            else:
                st.warning("❌ Не вдалося отримати поточну погоду.")

    st.markdown("---")
    
    st.write("✍️ **СПОСІБ 2: Вручну** - впишіть свої значення або відредагуйте автоматично завантажені:")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        temp_max = st.number_input("Макс. темп. (°C)", key='t_max', step=0.5)
    with col2:
        temp_min = st.number_input("Мінім. темп. (°C)", key='t_min', step=0.5)
    with col3:
        wind_max = st.number_input("Вітер (км/год)", key='w_max', step=1.0)
    with col4:
        precip_today = st.number_input("Опади (мм)", key='p_sum', step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    
    _, btn_col, _ = st.columns([1, 2, 1])
    with btn_col:
        predict_btn = st.button("🔮 ДІЗНАТИСЯ ПРОГНОЗ НА ЗАВТРА", use_container_width=True, type="primary")

    if predict_btn:
        if 'trained_model' not in st.session_state:
            st.warning("👈 Спочатку навчіть модель у бічному меню (Крок 2)!")
        else:
            model = st.session_state['trained_model']
            input_data = pd.DataFrame([[temp_max, temp_min, wind_max, precip_today]], columns=st.session_state['features_list'])
            
            prediction = model.predict(input_data)[0]
            probabilities = model.predict_proba(input_data)[0] 
            
            st.markdown("---")
            if prediction == 1:
                st.info("### 🌧️ Увага! Завтра очікуються опади.")
                st.progress(float(probabilities[1]))
                st.write(f"**Ймовірність дощу/снігу: {probabilities[1]*100:.1f}%**")
            else:
                st.success("### ☀️ Чудові новини! Завтра опадів не очікується.")
                st.progress(float(probabilities[0]))
                st.write(f"**Ймовірність ясної погоди: {probabilities[0]*100:.1f}%**")

elif page == "📊 Аналітика та Дані":
    if os.path.exists("weather_daily.csv"):
        df_show = pd.read_csv("weather_daily.csv")
        st.markdown("### 📈 Температурний графік (Макс vs Мін)")
        st.line_chart(df_show[['time', 'temperature_2m_max', 'temperature_2m_min']].set_index('time'))
        
        col_data, col_metrics = st.columns(2)
        with col_data:
            st.markdown("### 📁 Всі завантажені дані")
            # Обмежуємо висоту таблиці, щоб було зручно гортати
            st.dataframe(df_show, use_container_width=True, height=300)
        with col_metrics:
            st.markdown("### 🤖 Метрики ML-моделі")
            if 'report' in st.session_state:
                st.metric("Загальна точність (Accuracy)", f"{st.session_state['accuracy']*100:.2f}%")
                with st.expander("Подивитися детальний звіт (Classification Report)"):
                    st.code(st.session_state['report'])
            else:
                st.info("Навчіть модель у бічному меню, щоб побачити метрики.")
    else:
        st.warning("👈 Спочатку завантажте дані у бічному меню (Крок 1)!")