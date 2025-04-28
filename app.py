import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

# 🎯 Configuración de la página
st.set_page_config(page_title="Evaluador PPE Inteligente", layout="wide")

# ✨ Firma superior
st.markdown("""
<center>
    <p style='font-size:18px;'><strong>ANÁLISIS DE SENTIMIENTOS</strong><br>Todos los derechos reservados ©️</p>
</center>
""", unsafe_allow_html=True)

# 📂 Ruta a la carpeta del modelo
modelo_path = r"C:\Users\Angelly\OneDrive\Documentos\pysentimiento\content\pysentimiento\robertuito-base-uncased-save"

# 1️⃣ Cargar arquitectura
model = AutoModelForSequenceClassification.from_pretrained(modelo_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(modelo_path)

# 🔍 Instrucciones
with st.expander("📖 ¿Cómo se usa esta herramienta?"):
    st.markdown("""
    - 📄 Escribe un **texto** describiendo el equipo de protección personal que estás usando.
    - 🛡️ La herramienta analizará el texto y te indicará el sentimiento.
    """)

# 📄 Cuadro de texto para ingresar la descripción
texto_entrada = st.text_area("Escribe un texto:", height=200)

# Diccionario de emociones según tu modelo
id2emotion = {
    0: "neutral",
    1: "suicidal",
    2: "depressed",
    3: "lonely",
    4: "disappointment",
    5: "disgust",
    6: "fear",
    7: "anger",
    8: "sadness",
    9: "hopeless",
    10: "embarrassment",
    11: "remorse",
    12: "nervousness",
    13: "grief"
}


# 🔄 Botón para analizar el texto
if st.button("📤 Analizar Texto"):
    if texto_entrada.strip():
        # Tokenización y predicción
        inputs = tokenizer(texto_entrada, return_tensors="pt", truncation=True, padding=True)

        with torch.no_grad():
            logits = model(**inputs).logits
        prediccion = torch.argmax(logits, dim=-1).item()

        # Traducir predicción a emoción
        emocion_predicha = id2emotion.get(prediccion, "desconocido")

        # Mostrar resultado
        st.markdown(f"<center><h4>🎭 Emoción detectada: <strong>{emocion_predicha.capitalize()}</strong></h4></center>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ Por favor, escribe cómo te sientes.")