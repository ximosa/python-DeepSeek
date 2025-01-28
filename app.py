import streamlit as st
import os
import requests
import concurrent.futures
import time

# Configuración de la página
st.set_page_config(
    page_title="texto-corto",
    layout="wide"
)

# Obtener la API Key de DeepSeek
try:
    DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
    DEEPSEEK_API_URL = "https://api.deepseek.com/v1/chat/completions"
except KeyError:
    st.error("La variable de entorno DEEPSEEK_API_KEY no está configurada.")
    st.stop()

def dividir_texto(texto, max_tokens=1500):  # Tamaño intermedio
    """Divide el texto en fragmentos más pequeños."""
    tokens = texto.split()
    fragmentos = []
    fragmento_actual = []
    cuenta_tokens = 0

    for token in tokens:
        cuenta_tokens += 1
        if cuenta_tokens <= max_tokens:
            fragmento_actual.append(token)
        else:
            fragmentos.append(" ".join(fragmento_actual))
            fragmento_actual = [token]
            cuenta_tokens = 1
    if fragmento_actual:
        fragmentos.append(" ".join(fragmento_actual))
    return fragmentos

def limpiar_transcripcion_deepseek(texto):
    """
    Limpia una transcripción usando DeepSeek.
    """
    prompt = f"""
       Actúa como un lector profundo y reflexivo usando un tono conversacional y ameno, como si le contaras la historia a un amigo. Escribe en primera persona, como si tú hubieras vivido la experiencia o reflexionado sobre los temas presentados.
    Sigue estas pautas:
    - Reescribe el siguiente texto utilizando tus propias palabras, y asegúrate de mantener una longitud similar al texto original.
    No reduzcas la información, e intenta expandir cada punto si es posible.
    Quiero  una longitud comparable al texto original.
    - Dale un titulo preciso y llamativo.
    - Evita mencionar nombres de personajes o del autor.
    - Concentra el resumen en la experiencia general, las ideas principales, los temas y las emociones transmitidas por el texto.
    - Utiliza un lenguaje evocador y personal, como si estuvieras compartiendo tus propias conclusiones tras una profunda reflexión.
    - No uses nombres propios ni nombres de lugares específicos, refiérete a ellos como "un lugar", "una persona", "otro personaje", etc.
    - Usa un lenguaje claro y directo
    - Escribe como si estuvieras narrando una historia
    - Evita los asteriscos en el texto, dame tan solo el texto sin encabezados ni texto en negrita
    -Importante, el texto debe adaptarse para que el lector de voz de google lo lea lo mejor posible
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
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Limitar a 5 solicitudes simultáneas
        futures = {
            executor.submit(limpiar_transcripcion_deepseek, fragmento): fragmento
            for fragmento in fragmentos
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            st.write(f"Procesando fragmento {i+1}/{len(fragmentos)}")
            texto_limpio = future.result()
            if texto_limpio:
                texto_limpio_completo += texto_limpio + " "
            time.sleep(1)  # Pequeño retraso entre solicitudes
    
    return texto_limpio_completo.strip()

def descargar_texto(texto_formateado):
    """
    Genera un enlace de descarga para el texto formateado.
    """
    return st.download_button(
        label="Descargar Texto",
        data=texto_formateado.encode('utf-8'),
        file_name="transcripcion_formateada.txt",
        mime="text/plain"
    )

st.title("Limpiador de Transcripciones de YouTube (con DeepSeek)")

transcripcion = st.text_area("Pega aquí tu transcripción sin formato:")

# Botón para procesar el texto
if st.button("Procesar Texto"):
    if transcripcion:
        with st.spinner("Procesando con DeepSeek..."):
            texto_limpio = procesar_transcripcion(transcripcion)
            if texto_limpio:
                st.subheader("Transcripción Formateada:")
                st.write(texto_limpio)
                descargar_texto(texto_limpio)
    else:
        st.warning("Por favor, pega una transcripción para procesar.")
