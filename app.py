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

  st.title("Cambia los parámetros de tu canvas")
  
  drawing_mode = st.selectbox(
    "Selecciona el modo de dibujo",
    ("freedraw", "line", "transform", "rect", "circle")
  )


  stroke_width = st.slider("Grosor del pincel", 1, 100, 10)

  stroke_color = st.color_picker("Selecciona el color de linea", "#000000")

  fill_color = st.color_picker("Selecciona el color de relleno", "#000000")
  
  bg_color = st.color_picker("Selecciona el color del fondo", "#FFFFFF")
          

st.title('Pitonisa Imperial: Descubre tu destino')
image = Image.open('pitonisa.jpg')
st.image(image)

os.environ['OPENAI_API_KEY'] = st.secrets["settings"]["key"] 

api_key_2 = os.environ['OPENAI_API_KEY']
client = OpenAI(api_key=api_key_2)


st.subheader("Escribe tu nombre en el canvas")


# Create a canvas component
canvas_result = st_canvas(
    fill_color=fill_color,  # Fixed fill color with some opacity
    stroke_width=stroke_width,
    stroke_color=stroke_color,
    background_color=bg_color,
    height=720,
    width=1280,
    #background_image= None #Image.open(bg_image) if bg_image else None,
    drawing_mode=drawing_mode,
    key="canvas",
)



if canvas_result.image_data is not None:
    # Convertir la imagen del canvas a formato RGB
    input_numpy_array = np.array(canvas_result.image_data)
    input_image = Image.fromarray(input_numpy_array.astype('uint8'), 'RGBA')
    img_rgb = input_image.convert("RGB")  # Asegurarse de que la imagen esté en formato RGB
    text = pytesseract.image_to_string(img_rgb)

# Solo mostrar los parámetros de traducción si se ha reconocido texto
if text.strip():  # Asegurarse de que el texto no esté vacío
    detected_language = translator.detect(text).lang  # Detectar el idioma del texto
    warning_message.empty()  # Limpiar el mensaje de advertencia
    with st.sidebar:
        st.subheader("Parámetros de traducción")
        
        try:
            os.mkdir("temp")
        except FileExistsError:
            pass

        # Cambiar automáticamente el lenguaje de entrada
        input_language = detected_language

        # Mostrar el idioma detectado
        lang_map = {
            "en": "Inglés",
            "es": "Español",
            "bn": "Bengalí",
            "ko": "Coreano",
            "zh-cn": "Mandarín",
            "ja": "Japonés",
            "fr": "Francés",
            "de": "Alemán",
            "pt": "Portugués",
            "ru": "Ruso"  # Agregar ruso
        }

        st.markdown(f"### El texto reconocido fue:")
        st.write(text)
        st.markdown(f"**Idioma detectado:** {lang_map.get(input_language, 'Desconocido')}")

        # Establecer el idioma de entrada en función del idioma detectado
        in_lang_name = lang_map.get(input_language, "Desconocido")
        
        # Seleccionar el idioma de entrada automáticamente
        in_lang_options = list(lang_map.values())
        in_lang_index = in_lang_options.index(in_lang_name) if in_lang_name in in_lang_options else 0

        # Desplegable para el lenguaje de entrada
        st.selectbox("Seleccione el lenguaje de entrada", in_lang_options, index=in_lang_index)

        out_lang = st.selectbox(
            "Selecciona tu idioma de salida",
            ("Inglés", "Español", "Bengalí", "Coreano", "Mandarín", "Japonés", "Francés", "Alemán", "Portugués", "Ruso"),  # Agregar ruso
        )
        output_language = {
            "Inglés": "en",
            "Español": "es",
            "Bengalí": "bn",
            "Coreano": "ko",
            "Mandarín": "zh-cn",
            "Japonés": "ja",
            "Francés": "fr",
            "Alemán": "de",
            "Portugués": "pt",
            "Ruso": "ru"  # Agregar ruso
        }.get(out_lang, "en")

        english_accent = st.selectbox(
            "Seleccione el acento (solo aplica para inglés; no afecta el audio en otros idiomas)",
            (
                "Default",
                "India",
                "United Kingdom",
                "United States",
                "Canada",
                "Australia",
                "Ireland",
                "South Africa",
            ),
        )
        
        tld = {
            "Default": "com",
            "India": "co.in",
            "United Kingdom": "co.uk",
            "United States": "com",
            "Canada": "ca",
            "Australia": "com.au",
            "Ireland": "ie",
            "South Africa": "co.za",
        }.get(english_accent, "com")

        if st.button("Convertir"):
            loading_placeholder.image("dog.gif")  # Mostrar el GIF de carga
            
            result, output_text = text_to_speech(input_language, output_language, text, tld)
            audio_file = open(f"temp/{result}.mp3", "rb")
            audio_bytes = audio_file.read()
            st.markdown(f"## Tu audio:")
            st.audio(audio_bytes, format="audio/mp3", start_time=0)

            # Limpiar el GIF de carga después de que se genera el audio
            loading_placeholder.empty()

            # Mostrar automáticamente el texto de salida
            st.markdown(f"## Texto de salida:")
            st.write(f"{output_text}")
else:
    warning_message.warning("No se ha reconocido texto aún.")


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



