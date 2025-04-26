import numpy as np
import pandas as pd
import joblib
import streamlit as st
from tensorflow.keras.models import load_model

# CONFIGURACIÓN DE LA PÁGINA
st.set_page_config(
    page_title='Flight Experience Feedback Analyzer',  # Actualizado para reflejar el nuevo título
    page_icon='risk_score.jpg',
    layout='wide'
)

# Colocamos la imagen en la parte superior (antes del título) y la hacemos tres veces más grande
# Ahora la imagen y el título estarán centrados en la parte superior
st.image('risk_score.jpg', width=300)  # Tamaño más grande, ajustado a 300px de ancho

# Texto con el nombre del creador debajo de la imagen, más pequeño y alineado a la izquierda
st.markdown("<h5 style='color: #888;'>                                         Created by Álvaro Gamundi</h5>", unsafe_allow_html=True)

# Título de la app (más grande y llamativo) justo debajo de la imagen
st.markdown("<h1 style='text-align: center; color: #5F6368;'>Flight Experience Feedback Analyzer</h1>", unsafe_allow_html=True)

# Descripción
st.markdown("""
    <p style='text-align: center; font-size: 18px; color: #555;'>A simple tool to analyze the sentiment of your flight experience. Just write your comment and let us analyze it!</p>
""", unsafe_allow_html=True)

# Entrada de texto EN LA PÁGINA PRINCIPAL
texto_usuario = st.text_area('Write your experience here', height=150)

# Botón para hacer predicción
if st.button("Submit your comment"):
    if texto_usuario:
        x = pd.DataFrame([texto_usuario], columns=['text'])

        try:
            # Cargar stopwords y preprocesar el texto
            from stopwords_utils import preprocess_text
            from stopwords_utils import get_clean_stopwords
            stopwords = get_clean_stopwords()

            # Cargar el modelo tfidf
            tfidf_loaded = joblib.load('tfidf_vectorizer.pkl')
            x = tfidf_loaded.transform(x.text)
            x = x.toarray()
            x = pd.DataFrame(x, columns=tfidf_loaded.get_feature_names_out())
            x = x.to_numpy()

            # Cargar el modelo de Keras
            best_model = load_model('best_model_balanceado.keras', compile=True)
            
            predicciones = best_model.predict(x)

            # Lista de clases
            clases = ['neutral', 'positive', 'negative']

            # Obtener el índice de la clase con la mayor probabilidad
            predicciones_clases = np.argmax(predicciones, axis=1)

            # Mapear el índice a la clase correspondiente
            clases_predichas = [clases[i] for i in predicciones_clases]

            # Mostrar mensaje personalizado según la predicción (en negro)
            if clases_predichas[0] == 'negative':
                st.markdown("<h3 style='color: grey;'>We're truly sorry for your experience. We will get in touch with you to improve anything that needs attention.</h3>", unsafe_allow_html=True)
            elif clases_predichas[0] == 'positive':
                st.markdown("<h3 style='color: grey;'>We're happy to hear your experience was good! We hope to have you flying with us again soon.</h3>", unsafe_allow_html=True)
            else:  # 'neutral'
                st.markdown("<h3 style='color: grey;'>Thank you for your feedback! We are constantly working to improve our service.</h3>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")











