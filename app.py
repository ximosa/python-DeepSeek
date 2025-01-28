import streamlit as st
import os
import requests
import concurrent.futures
from time import perf_counter

st.set_page_config(
    page_title="Texto Corto",
    layout="wide"
)

# Obtener la API Key de DeepSeek
try:
    DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
except KeyError:
    st.error("La variable de entorno DEEPSEEK_API_KEY no está configurada.")
    st.stop()

def dividir_texto(texto, max_tokens=3000):
    """Divide el texto en fragmentos más pequeños de manera inteligente."""
    tokens = texto.split()
    fragmentos = []
    fragmento_actual = []
    cuenta_tokens = 0
    
    for token in tokens:
      if cuenta_tokens + len(token.split()) <= max_tokens:
        fragmento_actual.append(token)
        cuenta_tokens += len(token.split())
      else:
         fragmentos.append(" ".join(fragmento_actual))
         fragmento_actual = [token]
         cuenta_tokens = len(token.split())

    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))
    return fragmentos

def limpiar_transcripcion_deepseek(texto):
    """Limpia una transcripción usando DeepSeek."""
    prompt = f"""
        Actúa como un lector profundo y reflexivo usando un tono conversacional y ameno, como si le contaras la historia a un amigo. Escribe en primera persona, como si tú hubieras vivido la experiencia o reflexionado sobre los temas presentados.

        Sigue estas pautas:
        - Reescribe el siguiente texto utilizando tus propias palabras, y asegúrate de mantener una longitud similar al texto original.
        No reduzcas la información, e intenta expandir cada punto si es posible.
        No me generes un resumen, quiero un texto parafraseado y expandido con una longitud comparable al texto original.
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

    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=data)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        st.error(f"Error al procesar con DeepSeek: {e}")
        return None

def procesar_transcripcion(texto):
    """Procesa el texto dividiendo en fragmentos y usando DeepSeek en paralelo."""
    fragmentos = dividir_texto(texto)
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

st.title("Limpiador de Transcripciones de YouTube (con DeepSeek)")

transcripcion = st.text_area("Pega aquí tu transcripción sin formato:")

if 'texto_procesado' not in st.session_state:
    st.session_state['texto_procesado'] = ""

if st.button("Procesar"):
    if transcripcion:
        start_time = perf_counter()
        texto_limpio = procesar_transcripcion(transcripcion)
        end_time = perf_counter()
        st.session_state['texto_procesado'] = texto_limpio
        st.success(f"Tiempo de procesamiento: {end_time - start_time:.2f} segundos")
    else:
        st.warning("Por favor, introduce el texto a procesar.")


if st.session_state['texto_procesado']:
    st.subheader("Transcripción Formateada:")
    st.write(st.session_state['texto_procesado'])
    descargar_texto(st.session_state['texto_procesado'])
