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

# ✅ ИСПРАВЛЕНО: убраны аргументы
model = load_full_model()  # <-- было load_full_model(FILE_ID, SAVE_PATH)
preprocess = get_preprocess()


# --- ИНТЕРФЕЙС СТРИМЛИТ ---
st.title("🌾 Определение растений по фото")
st.write("Загрузите фото растения, и модель определит его вид")

uploaded_file = st.file_uploader("Выберите фото (JPG, PNG)", type=["jpg", "jpeg", "png"])

# Список названий классов
names = ['Cherry', 'Coffee-plant', 'Cucumber', 'Fox_nut(Makhana)', 'Lemon',
         'Olive-tree', 'Pearl_millet(bajra)', 'Tobacco-plant', 'almond', 'banana',
         'cardamom', 'chilli', 'clove', 'coconut', 'cotton',
         'gram', 'jowar', 'jute', 'maize', 'mustard-oil',
         'papaya', 'pineapple', 'rice', 'soyabean', 'sugarcane',
         'sunflower', 'tea', 'tomato', 'vigna-radiati(Mung)', 'wheat']

if uploaded_file:
    # Отображение картинки
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ваше изображение', use_container_width=True)
    
    if st.button("🚀 Запустить распознавание"):
        # Проверка, что модель загружена
        if model is None:
            st.error("Модель не загружена. Проверьте файл весов.")
        else:
            with st.spinner("Распознаю растение..."):
                # Превращаем картинку в тензор (с добавлением батч-размерности)
                img_tensor = preprocess(image).unsqueeze(0)
                
                with torch.no_grad():
                    output = model(img_tensor)
                    # Получаем вероятности для всех классов
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    # Берем индекс самого вероятного класса
                    predicted_idx = torch.argmax(output, 1).item()
                    confidence = probabilities[predicted_idx].item()
                    
            # Вывод результата
            st.success(f"**Результат:** {names[predicted_idx]}")
            st.info(f"📊 Уверенность модели: {confidence:.2%}")
            
            # Дополнительно показываем топ-3 наиболее вероятных класса
            top3_prob, top3_idx = torch.topk(probabilities, 3)
            st.write("**Альтернативные варианты:**")
            for i in range(3):
                st.write(f"- {names[top3_idx[i]]}: {top3_prob[i].item():.2%}")