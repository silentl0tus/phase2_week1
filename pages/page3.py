import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# ============================================
# КОНФИГУРАЦИЯ СТРАНИЦЫ
# ============================================
st.set_page_config(
    page_title="Skin Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

# ============================================
# ВАША АРХИТЕКТУРА МОДЕЛИ
# ============================================
class MySkinCancerModel(nn.Module):
    """Ваша собственная модель"""
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = models.resnet50(weights=None)
        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================
# ПУТИ И МЕТРИКИ
# ============================================
MODEL_PATH = Path(__file__).parent.parent / "models" / "my_skin_cancer_model.pth"

MY_METRICS = {
    'accuracy': 0.9076,
    'sensitivity': 0.8967,
    'specificity': 0.9167,
    'epoch': 12,
    'train_samples': 2637,
    'test_samples': 660
}

CLASS_NAMES = ['Benign (Доброкачественное)', 'Malignant (Злокачественное)']

# ============================================
# ЗАГРУЗКА МОДЕЛИ
# ============================================
@st.cache_resource
def load_my_model():
    """Загружает ВАШУ модель"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = MySkinCancerModel(num_classes=2)
    
    if MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        st.sidebar.success(f"✅ Модель загружена (Эпоха {MY_METRICS['epoch']})")
    else:
        st.sidebar.warning("⚠️ Используется непредобученная модель")
    
    model.eval()
    model.to(device)
    
    return model, device

# ============================================
# ТРАНСФОРМАЦИИ
# ============================================
def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# ============================================
# ПРЕДСКАЗАНИЕ
# ============================================
def predict(image, model, device):
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1).item()
    
    probs = probabilities[0].cpu().numpy()
    
    return CLASS_NAMES[prediction], probs, prediction

# ============================================
# ВИЗУАЛИЗАЦИЯ
# ============================================
def plot_probabilities(probs):
    fig = go.Figure(data=[
        go.Bar(
            x=['Benign', 'Malignant'],
            y=probs,
            text=[f'{p:.1%}' for p in probs],
            textposition='auto',
            marker_color=['#2ecc71', '#e74c3c'],
            opacity=0.8
        )
    ])
    
    fig.update_layout(
        title="Вероятности классов",
        yaxis_title="Вероятность",
        yaxis_tickformat='.0%',
        height=350,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def plot_gauge(value, title, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value * 100,
        title={'text': title, 'font': {'size': 14}},
        number={'suffix': '%', 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': '#ffebee'},
                {'range': [50, 75], 'color': '#fff3e0'},
                {'range': [75, 90], 'color': '#e8f5e9'},
                {'range': [90, 100], 'color': '#c8e6c9'}
            ]
        }
    ))
    fig.update_layout(height=200)
    return fig

# ============================================
# CSS СТИЛИ
# ============================================
def local_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 25px;
        border-radius: 15px;
        margin: 20px 0;
        text-align: center;
    }
    .benign {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    .malignant {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
    }
    .result-text {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #6b7280;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================
# ОСНОВНОЙ КОНТЕНТ
# ============================================
def main():
    local_css()
    
    st.markdown('<h1 class="main-header">🔬 Мой Skin Cancer Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Классификация: Benign vs Malignant | ResNet50 + Transfer Learning</p>', unsafe_allow_html=True)
    
    # Загружаем модель
    model, device = load_my_model()
    
    # ============================================
    # SIDEBAR
    # ============================================
    with st.sidebar:
        st.markdown("### 📊 Метрики моей модели")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("🎯 Accuracy", f"{MY_METRICS['accuracy']*100:.1f}%")
            st.metric("🔬 Sensitivity", f"{MY_METRICS['sensitivity']*100:.1f}%")
        with col2:
            st.metric("✅ Specificity", f"{MY_METRICS['specificity']*100:.1f}%")
            st.metric("📈 Эпоха", MY_METRICS['epoch'])
        
        st.markdown("---")
        st.markdown("### 🏗️ Архитектура")
        st.markdown("""
        - **Backbone**: ResNet50
        - **Transfer Learning**: ImageNet
        - **Разморожен**: Layer4 + FC
        - **Dropout**: 0.5
        """)
        
        st.markdown("---")
        st.markdown("### 📚 Данные")
        st.markdown(f"""
        - **Train**: {MY_METRICS['train_samples']} изображений
        - **Test**: {MY_METRICS['test_samples']} изображений
        - **Классы**: Benign / Malignant
        """)
        
        st.markdown("---")
        st.markdown("### ⚠️ Дисклеймер")
        st.info(
            "Модель предназначена для исследовательских целей. "
            "Не является медицинским диагнозом."
        )
    
    # ============================================
    # ОСНОВНАЯ ОБЛАСТЬ
    # ============================================
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Загрузите изображение")
        
        uploaded_file = st.file_uploader(
            "Выберите дерматоскопическое изображение",
            type=['jpg', 'jpeg', 'png', 'bmp']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Загруженное изображение", use_container_width=True)
            
            if st.button("🔍 Анализировать", type="primary", use_container_width=True):
                with st.spinner("Анализ..."):
                    prediction, probs, pred_class = predict(image, model, device)
                    st.session_state['prediction'] = prediction
                    st.session_state['probs'] = probs
                    st.session_state['pred_class'] = pred_class
    
    with col2:
        st.markdown("### 📊 Результаты")
        
        if 'prediction' in st.session_state:
            prediction = st.session_state['prediction']
            probs = st.session_state['probs']
            pred_class = st.session_state['pred_class']
            
            box_class = 'benign' if pred_class == 0 else 'malignant'
            confidence = probs[pred_class]
            
            st.markdown(f"""
            <div class="result-box {box_class}">
                <div class="result-text">{prediction}</div>
                <div class="confidence-text">Уверенность: {confidence:.1%}</div>
            </div>
            """, unsafe_allow_html=True)
            
            fig = plot_probabilities(probs)
            st.plotly_chart(fig, use_container_width=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #2ecc71;">{probs[0]:.1%}</div>
                    <div class="metric-label">Benign</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: #e74c3c;">{probs[1]:.1%}</div>
                    <div class="metric-label">Malignant</div>
                </div>
                """, unsafe_allow_html=True)
            
            if pred_class == 1 and confidence > 0.7:
                st.warning("⚠️ Высокая вероятность злокачественного образования. Рекомендуется консультация дерматолога.")
            elif pred_class == 1:
                st.info("ℹ️ Признаки требуют внимания. Рекомендуется консультация специалиста.")
        else:
            st.info("👆 Загрузите изображение и нажмите 'Анализировать'")

# ============================================
# ЗАПУСК
# ============================================
if __name__ == "__main__":
    main()