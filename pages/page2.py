
import streamlit as st
import torch
import gdown
import os
import sys
from PIL import Image

# 1. Поднимаемся на уровень выше, чтобы видеть папку models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Импортируем из папки models (имя_папки.имя_файла)
from models.resnet import MyResNet, get_preprocess

# --- ДАЛЕЕ ВАШ КОД ---
FILE_ID = '10u-lzjpiuDFeaguYs-9dZVp0vcPIsdop'
SAVE_PATH = 'model_weights.pth'

@st.cache_resource
def load_full_model(file_id, save_path):
    if not os.path.exists(save_path):
        url = f'https://google.com{file_id}'
        gdown.download(url, save_path, quiet=True)
    
    model = MyResNet(num_classes=30)
    # Обязательно загружаем на CPU для сервера Streamlit
    state_dict = torch.load(save_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_full_model(FILE_ID, SAVE_PATH)
preprocess = get_preprocess()

# --- ИНТЕРФЕЙС СТРИМЛИТ ---
uploaded_file = st.file_uploader("Выберите фото (JPG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Отображение картинки
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Ваше изображение', use_container_width=True)
    
    if st.button("🚀 Запустить распознавание"):
        # Превращаем картинку в тензор (с добавлением батч-размерности)
        img_tensor = preprocess(image).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img_tensor)
            # Берем индекс самого вероятного класса
            _, predicted_idx = torch.max(output, 1)
            
        # Вывод результата
        st.success(f"**Результат:** Модель определила класс №{predicted_idx.item()}")
        
 
        names = ['Cherry', 'Coffee-plant', 'Cucumber', 'Fox_nut(Makhana)', 'Lemon',
        'Olive-tree', 'Pearl_millet(bajra)', 'Tobacco-plant', 'almond', 'banana',
        'cardamom', 'chilli', 'clove', 'coconut', 'cotton',
        'gram', 'jowar', 'jute', 'maize', 'mustard-oil',
        'papaya', 'pineapple', 'rice', 'soyabean', 'sugarcane',
        'sunflower', 'tea', 'tomato', 'vigna-radiati(Mung)', 'wheat']
        st.info(f"Это похоже на: {names[predicted_idx.item()]}")