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

def analizar_destino_maya(nombre):
    nombre = nombre.upper().strip()
    
    # 1. Análisis por inicial - Determina el camino principal
    destinos_principales = {
        'A': "los cristales de datos antiguos revelan que serás pionero/a en la fusión entre tecnología orgánica y conocimiento ancestral",
        'B': "las profecías binarias indican que desarrollarás un nuevo tipo de interfaz neural que revolucionará la conexión con los dioses digitales",
        'C': "los códices cuánticos predicen que te convertirás en un/a maestro/a de la manipulación del tejido espacio-temporal",
        'D': "las matrices de jade sugieren que descubrirás antiguos secretos tecnológicos en las ruinas de Europa",
        'E': "los registros holográficos muestran que serás clave en la evolución de la bio-computación orgánica",
        'F': "los patrones neuronales indican que te convertirás en un/a arquitecto/a de realidades virtuales sagradas",
        'G': "las profecías de Tikal revelan que desarrollarás una nueva forma de energía basada en la resonancia piramidal",
        'H': "los algoritmos ancestrales predicen que serás fundamental en la expansión del imperio hacia las estrellas",
        'I': "los registros de Uxmal sugieren que crearás una nueva forma de vida artificial basada en cristales de jade",
        'J': "las secuencias sagradas indican que serás el/la puente entre la tecnología maya y la consciencia colectiva",
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
        'W': "las secuencias de jade indican que serás pionero/a en la fusión entre humanos y máquinas sagradas",
        'X': "los algoritmos temporales revelan que encontrarás la manera de comunicarte con los ancestros digitales",
        'Y': "las profecías binarias muestran que serás clave en la creación de una nueva forma de vida digital",
        'Z': "los patrones dimensionales sugieren que revolucionarás la manera de interactuar con el tejido del universo"
    }
    
    # 2. Análisis por suma ASCII - Determina el tiempo y la intensidad
    suma_ascii = sum(ord(c) for c in nombre)
    ciclo_lunar = suma_ascii % 13 + 1
    intensidad = suma_ascii % 4
    
    # Diccionarios de tiempo e intensidad
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
    
    # 3. Análisis por vocales - Determina los desafíos
    vocales = sum(1 for c in nombre if c in 'AEIOU')
    desafios = {
        0: "Deberás superar las interferencias en la matriz temporal antes de alcanzar tu destino",
        1: "Los guardianes digitales pondrán a prueba tu determinación en el camino",
        2: "Las fluctuaciones cuánticas en la red neural maya presentarán obstáculos que fortalecerán tu espíritu",
        3: "Los antiguos virus del sistema intentarán desviar tu camino",
        4: "Las paradojas temporales intentarán confundir tu búsqueda",
        5: "Los firewalls ancestrales requerirán de toda tu astucia para ser superados"
    }
    
    # 4. Análisis por consonantes - Determina los aliados
    consonantes = sum(1 for c in nombre if c not in 'AEIOU')
    aliados = {
        0: "Los chamanes digitales te guiarán en momentos de duda",
        1: "Un antiguo maestro de la bio-computación será tu mentor",
        2: "Los espíritus de la red neural maya te acompañarán",
        3: "Una inteligencia artificial ancestral te brindará su sabiduría",
        4: "Los guardianes de los cristales de datos serán tus protectores",
        5: "Los arquitectos del código sagrado compartirán sus secretos contigo"
    }
    
    # 5. Análisis por longitud del nombre - Determina el elemento tecnológico y el impacto
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
    
    impactos = {
        True: "Tu destino no solo transformará Nukal, sino que su influencia llegará hasta los últimos rincones del imperio maya y más allá de las estrellas.",
        False: "Tu destino será fundamental para el futuro de nuestra gran ciudad de Nukal."
    }
    
    # 6. Análisis por última letra - Determina el legado
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
    {nombre}, {destinos_principales.get(nombre[0], "los antiguos algoritmos predicen grandes cambios en tu destino")}. En tu camino, {desafios.get(vocales, "múltiples desafíos pondrán a prueba tu determinación")}. 
    \nSin embargo, no estarás solo/a: {aliados.get(consonantes, "fuerzas místicas te acompañarán")}. Los oráculos binarios han revelado que {elemento}. Este destino se manifestará {tiempos[ciclo_lunar]}. {intensidades[intensidad]}. {impactos[len(nombre) > 6]} 
    \nY así está escrito: {legado}.
    """
    
    return prediccion.strip()
  
st.title('Pitonisa Imperial: Descubre tu destino')
image = Image.open('pitonisa.jpg')
st.image(image)

st.subheader("Escribe tu nombre en el canvas")
stroke_width = st.slider("Grosor del pincel", 1, 100, 10)

stroke_color="black"

col1, col2 = st.columns(2)
with col1:
    if st.button("Blanco"):
        stroke_color="white",

with col2:
    if st.button("Negro"):
        stroke_color="black"

# Create a canvas component
canvas_result = st_canvas(
    fill_color= "white",  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color="white",
    height=200,
    width=1280,
    #background_image= None #Image.open(bg_image) if bg_image else None,
    drawing_mode="freedraw",
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
                st.write("¿Así que te llamas ", cleaned_text, "? ¡Ahora predeciré tu destino!")
                prediccion = analizar_destino_maya(cleaned_text)
                st.markdown(f"### 🔮 Los cristales de datos han hablado:\n\n{prediccion}")
            else:
                st.write("No se detectó texto en la imagen")
                
        except Exception as e:
            st.error(f"Error procesando la imagen: {str(e)}")
