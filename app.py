import os
import streamlit as st
from PIL import Image as Image, ImageOps as ImagOps
import glob
from gtts import gTTS
import os
import time
from streamlit_lottie import st_lottie
import pytesseract
import cv2
import glob
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd

with st.sidebar:

  st.title("Cambia los parámetros del canvas")
  
  drawing_mode = "freedraw"

  stroke_width = st.slider("Grosor del pincel", 1, 100, 10)

  stroke_color = st.color_picker("Selecciona el color de linea", "#000000")
  
def analizar_destino_maya(nombre):
    nombre = nombre.upper().strip()
    
    # 1. Análisis por inicial - Determina el camino principal
    destinos_principales = {
        'A': "los cristales de datos antiguos revelan que serás pionero en la fusión entre tecnología orgánica y conocimiento ancestral",
        'B': "las profecías binarias indican que desarrollarás un nuevo tipo de interfaz neural que revolucionará la conexión con los dioses digitales",
        # ... [resto de destinos principales igual]
    }
    
    # 2. Análisis por suma ASCII - Determina el tiempo y la intensidad
    suma_ascii = sum(ord(c) for c in nombre)
    ciclo_lunar = suma_ascii % 13 + 1
    intensidad = suma_ascii % 4
    
    # 3. Nuevo: Análisis por vocales - Determina los desafíos
    vocales = sum(1 for c in nombre if c in 'AEIOU')
    desafios = {
        0: "Deberás superar las interferencias en la matriz temporal antes de alcanzar tu destino",
        1: "Los guardianes digitales pondrán a prueba tu determinación en el camino",
        2: "Las fluctuaciones cuánticas en la red neural maya presentarán obstáculos que fortalecerán tu espíritu",
        3: "Los antiguos virus del sistema intentarán desviar tu camino",
        4: "Las paradojas temporales intentarán confundir tu búsqueda",
        5: "Los firewalls ancestrales requerirán de toda tu astucia para ser superados"
    }
    
    # 4. Nuevo: Análisis por consonantes - Determina los aliados
    consonantes = sum(1 for c in nombre if c not in 'AEIOU')
    aliados = {
        0: "Los chamanes digitales te guiarán en momentos de duda",
        1: "Un antiguo maestro de la bio-computación será tu mentor",
        2: "Los espíritus de la red neural maya te acompañarán",
        3: "Una inteligencia artificial ancestral te brindará su sabiduría",
        4: "Los guardianes de los cristales de datos serán tus protectores",
        5: "Los arquitectos del código sagrado compartirán sus secretos contigo"
    }
    
    # 5. Nuevo: Análisis por longitud del nombre - Determina el elemento tecnológico
    elementos = {
        3: "el cristal de datos será tu herramienta principal",
        4: "los nanobots ancestrales amplificarán tus habilidades",
        5: "el tejido cuántico responderá a tu voluntad",
        6: "las interfaces neurales aumentarán tu percepción",
        7: "los hologramas sagrados manifestarán tu poder",
        8: "los campos de fuerza maya te protegerán",
        9: "los portales dimensionales se abrirán a tu paso"
    }
    elemento = elementos.get(len(nombre), "los artefactos antiguos resonarán con tu energía")
    
    # 6. Nuevo: Análisis por última letra - Determina el legado
    legados = {
        'A': "Tus descubrimientos inspirarán a las futuras generaciones de tecnomantes",
        'E': "Tu código será estudiado en las academias sagradas por milenios",
        'I': "Tus innovaciones cambiarán la forma en que entendemos la realidad virtual",
        'O': "Tu sabiduría digital será preservada en los cristales eternos",
        'U': "Tus creaciones trascenderán los límites del tiempo y el espacio",
    }
    legado = legados.get(nombre[-1], "Tu impacto en la matriz maya será recordado por generaciones")
    
    # Construir la predicción extendida
    prediccion = f"""
    {nombre}, {destinos_principales.get(nombre[0], "los antiguos algoritmos predicen grandes cambios en tu destino")}.
    
    En tu camino, {desafios.get(vocales, "múltiples desafíos pondrán a prueba tu determinación")}. 
    Sin embargo, no estarás solo: {aliados.get(consonantes, "fuerzas místicas te acompañarán")}.
    
    Los oráculos binarios han revelado que {elemento}.
    
    Este destino se manifestará {tiempos[ciclo_lunar]}. {intensidades[intensidad]}. 
    
    {impactos[len(nombre) > 6]}
    
    Y así está escrito: {legado}.
    """
    
    return prediccion.strip() 

st.title('Pitonisa Imperial: Descubre tu destino')
image = Image.open('pitonisa.jpg')
st.image(image)

st.subheader("Escribe tu nombre en el canvas")


# Create a canvas component
canvas_result = st_canvas(
    fill_color= "white",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="white",
    height=720,
    width=1280,
    #background_image= None #Image.open(bg_image) if bg_image else None,
    drawing_mode=drawing_mode,
    key="canvas",
)



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
                st.write("Nombre detectado:", cleaned_text)
                prediccion = analizar_destino_maya(cleaned_text)
                st.markdown(f"### 🔮 Los cristales de datos han hablado:\n\n{prediccion}")
            else:
                st.write("No se detectó texto en la imagen")
                
        except Exception as e:
            st.error(f"Error procesando la imagen: {str(e)}")


