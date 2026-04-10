import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import sys
import os

# Добавляем путь к вашей папке, чтобы Python видел model.py
sys.path.append(os.path.join(os.getcwd(), 'models', 'Max'))
from model import get_resnet50_model

# Настройки
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
WEIGHTS_PATH = 'models/Max/model_weights.pth'

@st.cache_resource
def load_my_model():
    model = get_resnet50_model(num_classes=len(CLASSES))
    # Загружаем веса. map_location нужен для работы на CPU (если нет GPU на сервере)
    state_dict = torch.load(WEIGHTS_PATH, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

def predict(image, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(probabilities, 1)
    return CLASSES[pred.item()], conf.item()

# Интерфейс
st.title("Классификация Intel Images (Model by Max)")
st.write("Загрузите фото пейзажа, и модель определит, что на нем изображено.")

uploaded_file = st.file_uploader("Выберите изображение...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Загруженное фото', use_container_width=True)
    
    if st.button('Классифицировать'):
        model = load_my_model()
        label, confidence = predict(image, model)
        
        st.success(f"Результат: **{label}**")
        st.info(f"Уверенность: {confidence:.2%}")