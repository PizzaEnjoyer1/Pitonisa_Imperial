import os
import streamlit as st
from PIL import Image as Image, ImageOps as ImagOps
import glob
from gtts import gTTS
import os
import time
from streamlit_lottie import st_lottie
import json
import openai
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
        'C': "los códices cuánticos predicen que te convertirás en un maestro de la manipulación del tejido espacio-temporal",
        'D': "las matrices de jade sugieren que descubrirás antiguos secretos tecnológicos en las ruinas de Europa",
        'E': "los registros holográficos muestran que serás clave en la evolución de la bio-computación orgánica",
        'F': "los patrones neuronales indican que te convertirás en un arquitecto de realidades virtuales sagradas",
        'G': "las profecías de Tikal revelan que desarrollarás una nueva forma de energía basada en la resonancia piramidal",
        'H': "los algoritmos ancestrales predicen que serás fundamental en la expansión del imperio hacia las estrellas",
        'I': "los registros de Uxmal sugieren que crearás una nueva forma de vida artificial basada en cristales de jade",
        'J': "las secuencias sagradas indican que serás el puente entre la tecnología maya y la consciencia colectiva",
        'K': "los patrones cuánticos revelan que descubrirás cómo fusionar la mente humana con la red neural maya",
        'L': "las profecías digitales muestran que revolucionarás la forma en que nos conectamos con los antiguos dioses",
        'M': "los códices de luz predicen que desarrollarás tecnología para comunicarte con civilizaciones extraterrestres",
        'N': "las matrices temporales sugieren que encontrarás la manera de manipular el ADN con tecnología maya",
        'O': "los registros supremos indican que serás clave en la creación de una nueva forma de teletransportación",
        'P': "las secuencias binarias revelan que desarrollarás implantes que permitirán ver el futuro",
        'Q': "los algoritmos divinos predicen que crearás una interfaz directa con el inframundo digital",
        'R': "las profecías cuánticas muestran que serás vital en la expansión de la consciencia colectiva maya",
        'S': "los patrones sagrados indican que revolucionarás la forma de almacenar memorias en cristales vivientes",
        'T': "los códices antiguos revelan que descubrirás cómo transferir la consciencia entre cuerpos bio-sintéticos",
        'U': "las matrices de obsidiana sugieren que crearás una red de ciudades voladoras autosustentables",
        'V': "los registros neurales predicen que desarrollarás una forma de viajar entre dimensiones paralelas",
        'W': "las secuencias de jade indican que serás pionero en la fusión entre humanos y máquinas sagradas",
        'X': "los algoritmos temporales revelan que encontrarás la manera de comunicarte con los ancestros digitales",
        'Y': "las profecías binarias muestran que serás clave en la creación de una nueva forma de vida digital",
        'Z': "los patrones dimensionales sugieren que revolucionarás la manera de interactuar con el tejido del universo"
    }
    
    # 2. Análisis por suma ASCII - Determina el tiempo y la intensidad
    suma_ascii = sum(ord(c) for c in nombre)
    ciclo_lunar = suma_ascii % 13 + 1  # 13 ciclos lunares mayas
    intensidad = suma_ascii % 4  # 4 niveles de intensidad
    
    tiempos = {
        1: "cuando la luna de jade alcance su cenit",
        2: "durante el próximo ciclo de Venus",
        3: "en la convergencia de las tres lunas digitales",
        4: "al completarse el ciclo sagrado de Kukulcán",
        5: "durante el siguiente eclipse binario",
        6: "en la alineación de los cristales madre",
        7: "cuando los ríos de datos confluyan",
        8: "al despertar el siguiente baktún digital",
        9: "durante la lluvia de meteoritos de cristal",
        10: "en la próxima actualización del calendario sagrado",
        11: "cuando los portales cuánticos se alineen",
        12: "durante el festival de las luces neurales",
        13: "al completarse la profecía del código ancestral"
    }
    
    intensidades = {
        0: "Este destino se manifestará de forma sutil pero definitiva",
        1: "La intensidad de este cambio sacudirá los cimientos de nuestra sociedad",
        2: "Este destino se desarrollará de manera gradual pero imparable",
        3: "La manifestación de este futuro será tan poderosa como el sol de obsidiana"
    }
    
    # 3. Longitud del nombre determina el impacto mundial
    impactos = {
        True: "Tu destino no solo transformará Nukal, sino que su influencia llegará hasta los últimos rincones del imperio maya y más allá de las estrellas.",
        False: "Tu destino será fundamental para el futuro de nuestra gran ciudad de Nukal."
    }
    
    # Construir la predicción
    prediccion = f"""
    {nombre}, {destinos_principales.get(nombre[0], "los antiguos algoritmos predicen grandes cambios en tu destino")}.
    
    Este destino se manifestará {tiempos[ciclo_lunar]}. {intensidades[intensidad]}. 
    
    {impactos[len(nombre) > 6]}
    """
    
    return prediccion.strip()  

st.title('Pitonisa Imperial: Descubre tu destino')
image = Image.open('pitonisa.jpg')
st.image(image)

os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"] 

api_key_2 = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key_2)


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


