import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Заголовок страницы
st.set_page_config(page_title="AgriVision: Анализ зерна", layout="wide")

st.title("🌾 Видеоаналитика: Контроль качества зерна")
st.write("Загрузите фотографию пробы зерна для автоматического анализа.")

# 1. Загружаем модель
# На первом этапе используем стандартную модель, она скачается сама
@st.cache_resource
def load_model():
    # Позже мы заменим 'yolov8n.pt' на твою обученную модель для зерна
    return YOLO('yolov8n.pt') 

model = load_model()

# 2. Интерфейс загрузки
uploaded_file = st.sidebar.file_uploader("Выберите фото зерна", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    # Открываем изображение
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Исходное изображение")
        st.image(image, use_container_width=True)
    
    # 3. Кнопка запуска анализа
    if st.button("Запустить анализ"):
        with st.spinner('Нейросеть обрабатывает изображение...'):
            # Запускаем распознавание
            results = model(image)
            
            # Рисуем результат
            res_plotted = results[0].plot()
            
            with col2:
                st.subheader("Результат детекции")
                st.image(res_plotted, use_container_width=True)
            
            # 4. Вывод статистики
            st.divider()
            st.subheader("Отчет по пробе")
            
            # Считаем количество найденных объектов
            total_count = len(results[0].boxes)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Общее кол-во зерен", total_count)
            c2.metric("Битое зерно (имитация)", f"{np.random.randint(1, 5)}%")
            c3.metric("Сорность (имитация)", f"{np.random.uniform(0.1, 2.0):.2f}%")

else:
    st.info("👈 Пожалуйста, загрузите фото в боковом меню, чтобы начать.")
