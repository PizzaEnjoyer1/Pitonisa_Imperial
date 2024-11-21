import os
import streamlit as st
from PIL import Image as Image, ImageOps as ImagOps
import pytesseract
import cv2
import glob
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd

def analizar_destino_maya(nombre):
    # [The entire existing analizar_destino_maya function remains the same]
    # ... [previous implementation]

st.title('Pitonisa Imperial: Descubre tu destino')
st.components.v1.html(open('maya_animation.html', 'r').read(), height=300)

st.subheader("Escribe tu nombre en el canvas")

# Replace slider with buttons for stroke size
col1, col2, col3 = st.columns(3)

with col1:
    pequeño = st.button("Pequeño (10)")
with col2:
    mediano = st.button("Mediano (20)")
with col3:
    grande = st.button("Grande (30)")

# Default stroke width
if 'stroke_width' not in st.session_state:
    st.session_state.stroke_width = 10

if pequeño:
    st.session_state.stroke_width = 10
elif mediano:
    st.session_state.stroke_width = 20
elif grande:
    st.session_state.stroke_width = 30

color = st.radio(
    "Selecciona el color de escritura", 
    ["Blanco", "Negro"], 
    horizontal=True
)

stroke_color = "black" if color == "Negro" else "white"

# Create a full-width canvas component
canvas_result = st_canvas(
    fill_color="white",
    stroke_width=st.session_state.stroke_width,
    stroke_color=stroke_color,
    background_color="white",
    height=200,
    width=st.config.get_option("server.maxUploadSize"),  # Use maximum possible width
    drawing_mode="freedraw",
    key="canvas",
)

if st.button("Predice tu futuro"):
    if canvas_result.image_data is not None:
        with st.spinner("Analizando..."):
            try:
                # Obtener el array numpy directamente del canvas
                input_numpy_array = np.array(canvas_result.image_data)
                
                # Convertir de RGBA a RGB
                rgb_array = input_numpy_array[:, :, :3]
                
                # Convertir a escala de grises
                gray_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2GRAY)
                
                # Aplicar umbralización
                _, threshold = cv2.threshold(gray_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                
                # Realizar OCR y procesar el texto
                text = pytesseract.image_to_string(threshold)
                
                # Limpiar y formatear el texto en una línea
                cleaned_text = ' '.join(
                    word.strip()
                    for line in text.splitlines()
                    for word in line.split()
                    if word.strip()
                )
                
                if cleaned_text:
                    st.write("¿Así que te llamas ", cleaned_text, "? ¡Ahora predeciré tu destino!")
                    prediccion = analizar_destino_maya(cleaned_text)
                    st.markdown(f"### 🔮 Los cristales de datos han hablado:\n\n{prediccion}")
                else:
                    st.write("No se detectó texto en la imagen")
                    
            except Exception as e:
                st.error(f"Error procesando la imagen: {str(e)}")

    else:
        st.write("El canvas no contiene texto")
