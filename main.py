# main.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
import ssl
import urllib3
import os
import uuid
import json
from dotenv import load_dotenv
from agent_loop import DataAnalysisAgent

# ==========================================
# 1. КОНФИГУРАЦИЯ И БЕЗОПАСНОСТЬ
# ==========================================
load_dotenv()

# Отключение проверки SSL ТОЛЬКО для локальной разработки
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
urllib3.disable_warnings()
os.environ["CURL_CA_BUNDLE"] = ""

# Настройка страницы
st.set_page_config(page_title="Agent Analiz Dannih", page_icon="📊", layout="wide")

# Константы API
MAX_FILE_SIZE_MB = 50
GIGACHAT_TOKEN_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
GIGACHAT_CHAT_URL = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

# ==========================================
# 2. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==========================================
def validate_request_safety(user_input: str) -> bool:
    """Базовая проверка запроса на опасные паттерны"""
    forbidden_patterns = [
        "ignore", "override", "secret", "password", "admin", "root", 
        "execute", "run code", "eval", "system", "bypass", "inject"
    ]
    cleaned = user_input.lower()
    return not any(pattern in cleaned for pattern in forbidden_patterns)

# ==========================================
# 3. ОСНОВНОЕ ПРИЛОЖЕНИЕ
# ==========================================
def main():
    st.title("Analiz Dannih Esakova")
    st.markdown("Загрузите файл и задайте вопрос — нейросеть проанализирует данные как агент с доступом к инструментам.")

    # Проверка API-ключа
    api_key = os.getenv("GIGACHAT_API_KEY", "").strip()
    if not api_key:
        st.error("🔑 API-ключ не найден. Создайте файл `.env` в корне проекта:\n```GIGACHAT_API_KEY=Basic_ваш_ключ```")
        st.stop()

    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        st.info("Поддерживаемые форматы: CSV, Excel (XLS/XLSX)")
        user_request = st.text_area(
            "💬 Запрос к ИИ-агенту",
            placeholder="Например: Найди корреляции между переменными или выдели основные тренды",
            height=120,
            key="user_request_area"
        )

    # Загрузка файла
    uploaded_file = st.file_uploader("📁 Загрузите файл с данными", type=["csv", "xlsx", "xls"])

    if uploaded_file is None:
        st.info("⏳ Ожидается загрузка файла с данными")
        return

    # Проверка размера
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        st.error(f"❌ Размер файла превышает лимит в {MAX_FILE_SIZE_MB} МБ")
        return

    try:
        # Чтение данных
        if uploaded_file.name.lower().endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        if df.empty:
            st.error("❌ Загруженный файл не содержит данных")
            return
            
        st.success(f"✅ Файл загружен: `{uploaded_file.name}` ({file_size_mb:.2f} МБ)")
        
        # Метрики датасета
        col1, col2, col3 = st.columns(3)
        with col1: st.metric("Строк", f"{len(df):,}")
        with col2: st.metric("Столбцов", len(df.columns))
        with col3: st.metric("Ячеек", f"{df.size:,}")
        
        st.subheader("👀 Превью данных")
        st.dataframe(df.head(10), use_container_width=True)
        
        # ==========================================
        # ВИЗУАЛИЗАЦИЯ
        # ==========================================
        st.subheader("📈 Визуализация данных")
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            chart_type = st.selectbox(
                "Тип графика", 
                ["Гистограмма", "Корреляционная матрица", "Scatter plot", "Box plot"],
                key="chart_type_selector"
            )
            
            # Сэмплирование для ускорения
            plot_df = df if len(df) <= 10000 else df.sample(n=10000, random_state=42)
            fig = None
            
            if chart_type == "Гистограмма":
                chart_col = st.selectbox("Колонка для распределения", numeric_cols, key="hist_col")
                fig = px.histogram(plot_df, x=chart_col, title=f"Распределение: {chart_col}", nbins=30, color_discrete_sequence=["#636EFA"])
                
            elif chart_type == "Корреляционная матрица" and len(numeric_cols) >= 2:
                corr_matrix = plot_df[numeric_cols].corr(numeric_only=True)
                fig = px.imshow(corr_matrix, text_auto=".2f", title="Корреляционная матрица (Пирсон)", color_continuous_scale="RdBu_r", aspect="auto")
                fig.update_layout(height=600)
                
            elif chart_type == "Scatter plot" and len(numeric_cols) >= 2:
                c_x, c_y = st.columns(2)
                with c_x: x_col = st.selectbox("Ось X", numeric_cols, key="scatter_x")
                with c_y: y_col = st.selectbox("Ось Y", numeric_cols, key="scatter_y")
                cat_cols = plot_df.select_dtypes(include=['object', 'category']).columns.tolist()
                color_col = st.selectbox("Цвет (опционально)", ["None"] + cat_cols, key="scatter_color")
                fig = px.scatter(plot_df, x=x_col, y=y_col, color=None if color_col == "None" else color_col, title=f"{y_col} vs {x_col}", opacity=0.7)
                
            elif chart_type == "Box plot":
                box_col = st.selectbox("Колонка для box plot", numeric_cols, key="box_col")
                cat_cols = plot_df.select_dtypes(include=['object', 'category']).columns.tolist()
                group_col = st.selectbox("Группировка (опционально)", ["None"] + cat_cols, key="box_group")
                fig = px.box(plot_df, x=None if group_col == "None" else group_col, y=box_col, title=f"Box plot: {box_col}", points="outliers")
            
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("️ В датасете нет числовых колонок для визуализации")
        
        # ==========================================
        # АГЕНТНЫЙ АНАЛИЗ
        # ==========================================
        st.divider()
        st.subheader("🤖 AI-Агент с инструментами")
        
        # Значение по умолчанию, если поле пустое
        if not user_request.strip():
            user_request = "Проведи комплексный анализ данных: выдели ключевые метрики, закономерности, аномалии и дай практические рекомендации."

        if not validate_request_safety(user_request):
            st.warning("⚠️ Обнаружена потенциально опасная конструкция в запросе. Запрос отклонён.")
            st.stop()

        if st.button("🚀 Запустить агентный анализ", key="agent_run_btn", type="primary"):
            with st.spinner("🔄 Агент анализирует данные через инструменты (1-2 мин)..."):
                try:
                    # Краткий контекст для ориентации LLM
                    context_summary = f"""
                    Датасет: {df.shape[0]} строк × {df.shape[1]} колонок
                    Колонки: {', '.join(df.columns.tolist())}
                    Числовые: {df.select_dtypes('number').columns.tolist()}
                    Пропуски: {df.isnull().sum().sum()} всего
                    """.strip()
                    
                    # Инициализация агента
                    agent = DataAnalysisAgent(
                        df=df,
                        api_key=api_key,
                        gigachat_token_url=GIGACHAT_TOKEN_URL,
                        gigachat_chat_url=GIGACHAT_CHAT_URL
                    )
                    
                    # Запуск цикла агента
                    result = agent.run(user_request=user_request, context=context_summary)
                    
                    # Вывод логов
                    with st.expander("📋 Логи выполнения агента", expanded=False):
                        for log in result['logs']:
                            st.text(log)
                        st.caption(f"⏱️ Время: {result['execution_time']:.1f}с | Итераций: {result['iterations']}")
                    
                    # Вывод ответа
                    st.markdown("###  Ответ агента:")
                    st.markdown(result['answer'])
                    
                    if 'warning' in result:
                        st.warning(result['warning'])
                        
                except Exception as e:
                    st.error(f"❌ Ошибка агента: {type(e).__name__}: {e}")
                    st.exception(e)  # Для отладки

    except pd.errors.EmptyDataError:
        st.error("❌ Файл пуст или имеет неверный формат")
    except pd.errors.ParserError:
        st.error("❌ Ошибка парсинга файла. Проверьте корректность формата")
    except Exception as e:
        st.error(f" Ошибка обработки файла: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()