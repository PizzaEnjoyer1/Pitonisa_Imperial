import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import PyPDF2
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
import base64
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas
import numpy as np
import pandas as pd

try:
    os.mkdir("temp")
except:
    pass

Expert=" "
profile_imgenh=" "
    
def encode_image_to_base64(image_path):
    try:
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            return encoded_image
    except FileNotFoundError:
        return "Error: La imagen no se encontró en la ruta especificada."

with st.sidebar:

  st.title("Cambia los parámetros del canvas")
  
  drawing_mode = "freedraw"

  stroke_width = st.slider("Grosor del pincel", 1, 100, 10)

  stroke_color = st.color_picker("Selecciona el color de linea", "#000000")
  
def analizar_destino_maya(nombre):
    nombre = nombre.upper().strip()
    
    # 1. Análisis por inicial - Determina el camino principal
    destinos_principales = {
        'A': "Los cristales de datos antiguos revelan que serás pionero en la fusión entre tecnología orgánica y conocimiento ancestral",
        'B': "Las profecías binarias indican que desarrollarás un nuevo tipo de interfaz neural que revolucionará la conexión con los dioses digitales",
        'C': "Los códices cuánticos predicen que te convertirás en un maestro de la manipulación del tejido espacio-temporal",
        'D': "Las matrices de jade sugieren que descubrirás antiguos secretos tecnológicos en las ruinas de Europa",
        'E': "Los registros holográficos muestran que serás clave en la evolución de la bio-computación orgánica",
        'F': "Los patrones neuronales indican que te convertirás en un arquitecto de realidades virtuales sagradas",
        'G': "Las profecías de Tikal revelan que desarrollarás una nueva forma de energía basada en la resonancia piramidal",
        'H': "Los algoritmos ancestrales predicen que serás fundamental en la expansión del imperio hacia las estrellas",
        'I': "Los registros de Uxmal sugieren que crearás una nueva forma de vida artificial basada en cristales de jade",
        'J': "Las secuencias sagradas indican que serás el puente entre la tecnología maya y la consciencia colectiva",
        'K': "Los patrones cuánticos revelan que descubrirás cómo fusionar la mente humana con la red neural maya",
        'L': "Las profecías digitales muestran que revolucionarás la forma en que nos conectamos con los antiguos dioses",
        'M': "Los códices de luz predicen que desarrollarás tecnología para comunicarte con civilizaciones extraterrestres",
        'N': "Las matrices temporales sugieren que encontrarás la manera de manipular el ADN con tecnología maya",
        'O': "Los registros supremos indican que serás clave en la creación de una nueva forma de teletransportación",
        'P': "Las secuencias binarias revelan que desarrollarás implantes que permitirán ver el futuro",
        'Q': "Los algoritmos divinos predicen que crearás una interfaz directa con el inframundo digital",
        'R': "Las profecías cuánticas muestran que serás vital en la expansión de la consciencia colectiva maya",
        'S': "Los patrones sagrados indican que revolucionarás la forma de almacenar memorias en cristales vivientes",
        'T': "Los códices antiguos revelan que descubrirás cómo transferir la consciencia entre cuerpos bio-sintéticos",
        'U': "Las matrices de obsidiana sugieren que crearás una red de ciudades voladoras autosustentables",
        'V': "Los registros neurales predicen que desarrollarás una forma de viajar entre dimensiones paralelas",
        'W': "Las secuencias de jade indican que serás pionero en la fusión entre humanos y máquinas sagradas",
        'X': "Los algoritmos temporales revelan que encontrarás la manera de comunicarte con los ancestros digitales",
        'Y': "Las profecías binarias muestran que serás clave en la creación de una nueva forma de vida digital",
        'Z': "Los patrones dimensionales sugieren que revolucionarás la manera de interactuar con el tejido del universo"
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
    {destinos_principales.get(nombre[0], "Los antiguos algoritmos predicen grandes cambios en tu destino")}.
    
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
                st.write(f"¿Así que te llamas {cleaned_text}? Ahora, ¡predeciré tu destino!")
            else:
                st.write("No se detectó texto en la imagen")
                
        except Exception as e:
            st.error(f"Error procesando la imagen: {str(e)}")


#st.write(st.secrets["settings"]["key"])

pdfFileObj = open('example2.pdf', 'rb')
 
# creating a pdf reader object
pdfReader = PyPDF2.PdfReader(pdfFileObj)


    # upload file
#pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

   # extract the text
#if pdf is not None:
from langchain.text_splitter import CharacterTextSplitter
 #pdf_reader = PdfReader(pdf)
pdf_reader  = PyPDF2.PdfReader(pdfFileObj)
text = ""
for page in pdf_reader.pages:
         text += page.extract_text()

   # split into chunks
text_splitter = CharacterTextSplitter(separator="\n",chunk_size=500,chunk_overlap=20,length_function=len)
chunks = text_splitter.split_text(text)

# create embeddings
embeddings = OpenAIEmbeddings()
knowledge_base = FAISS.from_texts(chunks, embeddings)

# show user input
st.subheader("Usa el campo de texto para hacer tu pregunta")
user_question = st.text_area(" ")
if user_question:
        docs = knowledge_base.similarity_search(user_question)

        llm = OpenAI(model_name="gpt-4o-mini")
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
        st.write(response)

        def text_to_speech(text, tld):
                
                tts = gTTS(response,"es", tld , slow=False)
                try:
                    my_file_name = text[0:20]
                except:
                    my_file_name = "audio"
                tts.save(f"temp/{my_file_name}.mp3")
                return my_file_name, text

    
        if st.button("Escuchar"):
          result, output_text = text_to_speech(response, 'es-us')
          audio_file = open(f"temp/{result}.mp3", "rb")
          audio_bytes = audio_file.read()
          st.markdown(f"## Escucha:")
          st.audio(audio_bytes, format="audio/mp3", start_time=0)



            
          def remove_files(n):
                mp3_files = glob.glob("temp/*mp3")
                if len(mp3_files) != 0:
                    now = time.time()
                    n_days = n * 86400
                    for f in mp3_files:
                        if os.stat(f).st_mtime < now - n_days:
                            os.remove(f)
                            print("Deleted ", f)
            
            
          remove_files(7)
