import streamlit as st
import os
import requests
import concurrent.futures
from time import perf_counter, sleep
import logging
from datetime import datetime
import tiktoken  # Importa tiktoken

st.set_page_config(
    page_title="Texto Corto",
    layout="wide"
)

# Configuración de Logging
LOG_FILE = 'app.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Obtener la API Key de DeepSeek
try:
    DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
except KeyError:
    st.error("La variable de entorno DEEPSEEK_API_KEY no está configurada.")
    st.stop()

# Inicializa el tokenizador (¡IMPORTANTE!)
try:
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # O el modelo que uses
except Exception as e:
    st.error(f"Error al cargar el tokenizador: {e}. Asegúrate de tener tiktoken instalado (`pip install tiktoken`).")
    st.stop()

def contar_tokens(texto):
    """Cuenta tokens usando tiktoken."""
    return len(encoding.encode(texto))

def dividir_texto_dinamico(texto, tamano_fragmento_pequeno=750, tamano_fragmento_mediano=1500):
    """Divide el texto en fragmentos más pequeños dinámicamente."""
    longitud_texto = contar_tokens(texto)

    if longitud_texto < 1000:
        return [texto]  # No dividir si es muy corto
    elif longitud_texto < 5000:
        max_tokens = tamano_fragmento_mediano
        st.info(f"Dividiendo en fragmentos medianos (max {max_tokens} tokens).")
    else:
        max_tokens = tamano_fragmento_pequeno
        st.info(f"Dividiendo en fragmentos pequeños (max {max_tokens} tokens).")

    fragmentos = []
    fragmento_actual = []
    cuenta_tokens_actual = 0

    palabras = texto.split()  # Dividir por palabras para mejor control
    for palabra in palabras:
        tokens_palabra = contar_tokens(palabra)  # Tokenizar cada palabra

        if cuenta_tokens_actual + tokens_palabra <= max_tokens:
            fragmento_actual.append(palabra)
            cuenta_tokens_actual += tokens_palabra
        else:
            fragmentos.append(" ".join(fragmento_actual))
            fragmento_actual = [palabra]
            cuenta_tokens_actual = tokens_palabra

    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))

    st.info(f"Texto dividido en {len(fragmentos)} fragmentos.")
    return fragmentos

def limpiar_transcripcion_deepseek(texto, max_retries=3, initial_delay=1):
    """Limpia una transcripción usando DeepSeek con reintentos."""
    prompt = f"""
        Actúa como un escritor usando un tono conversacional y ameno,
        como si le contaras la historia a tus lectores. Escribe en primera persona, como si tú hubieras vivido la experiencia o reflexionado sobre los temas presentados.

        Sigue estas pautas:
        - El texto resultante debe tener al menos el 90% de la longitud del texto original, y preferiblemente entre el 100% 
        y el 120% (en número de tokens). Intenta expandir cada idea principal con 2-3 frases adicionales que ofrezcan ejemplos, 
        explicaciones o reflexiones personales. Si el texto original parece incompleto o le falta algo, añade detalles relevantes.
        - No reduzcas la información, e intenta expandir cada punto si es posible.
        - No me generes un resumen, quiero un texto parafraseado y expandido con una longitud comparable al texto original.
        - Dale un título preciso y llamativo.
        - Evita mencionar nombres de personajes o del autor.
        - Concentra el resumen en la experiencia general, las ideas principales, los temas y las emociones transmitidas por el texto.
        - Utiliza un lenguaje evocador y personal, como si estuvieras compartiendo tus propias conclusiones tras una profunda reflexión.
        - No uses nombres propios ni nombres de lugares específicos, refiérete a ellos como "un lugar", "una persona", "otro personaje", etc.
        - Usa un lenguaje claro y directo
        - Escribe como si estuvieras narrando una historia
        - Evita los asteriscos en el texto, dame tan solo el texto sin encabezados ni texto en negrita
        - Importante, el texto debe adaptarse para que el lector de voz de google lo lea lo mejor posible

        {texto}

        Texto corregido:
    """
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}]
    }
    
    retries = 0
    delay = initial_delay
    while retries <= max_retries:
        try:
            logging.info(f"Enviando solicitud a DeepSeek para texto: {texto[:50]}... (Intento {retries + 1})")
            response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            logging.info(f"Respuesta recibida de DeepSeek para texto: {texto[:50]}")
            return result['choices'][0]['message']['content']
        except requests.exceptions.RequestException as e:
            logging.error(f"Error en la solicitud a DeepSeek: {e} (Intento {retries + 1})")
            if e.response and e.response.status_code == 429:
              retries += 1
              sleep(delay)
              delay *= 2 # Delay exponencial
            else:
              return None
        except Exception as e:
            logging.error(f"Error al procesar la respuesta de DeepSeek: {e}")
            return None
    logging.error(f"Máximo número de reintentos alcanzado para el texto: {texto[:50]}.")
    return None

def procesar_transcripcion(texto):
    """Procesa el texto dividiendo en fragmentos y usando DeepSeek en paralelo."""
    fragmentos = dividir_texto_dinamico(texto)
    texto_limpio_completo = ""
    total_fragmentos = len(fragmentos)
    progress_bar = st.progress(0)

    with st.spinner("Procesando con DeepSeek..."):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(limpiar_transcripcion_deepseek, fragmento): fragmento
                for fragmento in fragmentos
            }
            
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                texto_limpio = future.result()
                if texto_limpio:
                    texto_limpio_completo += texto_limpio + " "
                progress_bar.progress(i / total_fragmentos)

    return texto_limpio_completo.strip()

def descargar_texto(texto_formateado):
    """Genera un enlace de descarga para el texto formateado."""
    return st.download_button(
        label="Descargar Texto",
        data=texto_formateado.encode('utf-8'),
        file_name="transcripcion_formateada.txt",
        mime="text/plain"
    )

def mostrar_logs():
    """Muestra los logs en Streamlit."""
    try:
      with open(LOG_FILE, 'r', encoding='utf-8') as f:
          log_content = f.read()
          st.subheader("Logs de la Aplicación:")
          st.text_area("Logs", value=log_content, height=300)
    except FileNotFoundError:
      st.error("El archivo de logs no fue encontrado.")

st.title("Limpiador de Transcripciones de YouTube (con DeepSeek)")

transcripcion = st.text_area("Pega aquí tu transcripción sin formato:")

if 'texto_procesado' not in st.session_state:
    st.session_state['texto_procesado'] = ""

if st.button("Procesar"):
    if transcripcion:
        # Antes de procesar, muestra la longitud del texto original
        longitud_original = contar_tokens(transcripcion)
        st.info(f"Longitud del texto original: {longitud_original} tokens.")

        start_time = perf_counter()
        texto_limpio = procesar_transcripcion(transcripcion)
        end_time = perf_counter()

        st.session_state['texto_procesado'] = texto_limpio
        st.success(f"Tiempo de procesamiento: {end_time - start_time:.2f} segundos")

        # Después de procesar, muestra la longitud del texto resultante
        longitud_resultante = contar_tokens(texto_limpio)
        st.info(f"Longitud del texto resultante: {longitud_resultante} tokens.")

    else:
        st.warning("Por favor, introduce el texto a procesar.")

if st.session_state['texto_procesado']:
    st.subheader("Transcripción Formateada:")
    st.write(st.session_state['texto_procesado'])
    descargar_texto(st.session_state['texto_procesado'])

if st.checkbox("Mostrar Logs"):
  mostrar_logs()
