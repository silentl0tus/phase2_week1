import streamlit as st
import torch
import gdown
import os
import sys
import requests
from PIL import Image

# 1. Поднимаемся на уровень выше, чтобы видеть папку models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Импортируем из папки models (имя_папки.имя_файла)
from models.resnet import MyResNet, get_preprocess

# --- ДАЛЕЕ ВАШ КОД ---
FILE_ID = '10u-lzjpiuDFeaguYs-9dZVp0vcPIsdop'
SAVE_PATH = 'model_weights.pth'

@st.cache_resource
def load_full_model():
    save_path = 'model_weights.pth'
    if not os.path.exists(save_path):
        # Ссылка на файл из GitHub Release
        url = "https://github.com/silentl0tus/phase2_week1/releases/download/v_1.0/plants_resnet50_final.pth"
        response = requests.get(url, stream=True)
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    model = MyResNet(num_classes=30)
    state_dict = torch.load(save_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    return model

# --- ФУНКЦИЯ ЗАГРУЗКИ КАРТИНКИ ПО ССЫЛКЕ ---
def load_image_from_url(url):
    """Загружает картинку по URL и возвращает объект PIL Image"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        st.error(f"Ошибка загрузки картинки: {e}")
        return None

# ✅ Загрузка модели
model = load_full_model()  # <-- было load_full_model(FILE_ID, SAVE_PATH)
preprocess = get_preprocess()


# --- ИНТЕРФЕЙС СТРИМЛИТ ---
st.title("🌾 Определение растений по фото")
st.write("Загрузите фото растения, и модель определит его вид")

upload_method = st.radio(
    "Выберите способ загрузки фото:",
    ["📁 Загрузить файл", "🔗 Ссылка на фото"]
)

image = None

# Способ 1: Загрузка файла
if upload_method == "📁 Загрузить файл":
    uploaded_file = st.file_uploader("Выберите фото (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Ваше изображение', use_container_width=True)

# Способ 2: Ссылка на фото
else:
    url = st.text_input("Введите ссылку на фото:", placeholder="https://example.com/photo.jpg")
    if url:
        if st.button("📥 Загрузить по ссылке"):
            with st.spinner("Загружаю фото..."):
                image = load_image_from_url(url)
            if image:
                st.image(image, caption='Загруженное изображение', use_container_width=True)
                st.success("Фото успешно загружено!")

# --- РАСПОЗНАВАНИЕ ---
if image is not None:
    if st.button("🚀 Запустить распознавание"):
        if model is None:
            st.error("Модель не загружена")
        else:
            with st.spinner("Распознаю растение..."):
                img_tensor = preprocess(image).unsqueeze(0)
                with torch.no_grad():
                    output = model(img_tensor)
                    predicted_idx = torch.argmax(output, 1).item()
                    
                    # Получаем уверенность модели
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    confidence = probabilities[predicted_idx].item()
            
            # Названия классов на русском
            names = ['Вишня', 'Кофейное дерево', 'Огурец', 'Фокс-нат (Махана)', 'Лимон',
                     'Оливковое дерево', 'Жемчужное просо (баджра)', 'Табак', 'Миндаль', 'Банан',
                     'Кардамон', 'Перец чили', 'Гвоздика', 'Кокос', 'Хлопок',
                     'Нут', 'Сорго', 'Джут', 'Кукуруза', 'Горчица',
                     'Папайя', 'Ананас', 'Рис', 'Соя', 'Сахарный тростник',
                     'Подсолнечник', 'Чай', 'Томат', 'Маш (бобы мунг)', 'Пшеница']
            
            # Вывод результата
            st.success(f"**🌿 Результат:** {names[predicted_idx]}")
            st.info(f"📊 Уверенность модели: {confidence:.2%}")
            
            # Показываем топ-3 альтернативы
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            st.write("**🔍 Альтернативные варианты:**")
            for i in range(3):
                if top3_idx[i] != predicted_idx:  # Не показываем основной результат повторно
                    st.write(f"- {names[top3_idx[i]]}: {top3_prob[i].item():.2%}")
