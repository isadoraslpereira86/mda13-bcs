# ============================================================
# CABECERA
# ============================================================
# Alumno: Isadora de Sampaio Leite Correa
# URL Streamlit Cloud: https://...streamlit.app
# URL GitHub: https://github.com/...

# ============================================================
# IMPORTS
# ============================================================
# Streamlit: framework para crear la interfaz web
# pandas: manipulación de datos tabulares
# plotly: generación de gráficos interactivos
# openai: cliente para comunicarse con la API de OpenAI
# json: para parsear la respuesta del LLM (que llega como texto JSON)
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI
import json

# ============================================================
# CONSTANTES
# ============================================================
# Modelo de OpenAI. No lo cambies.
MODEL = "gpt-4.1-mini"

# -------------------------------------------------------
# >>> SYSTEM PROMPT — TU TRABAJO PRINCIPAL ESTÁ AQUÍ <<<
# -------------------------------------------------------
# El system prompt es el conjunto de instrucciones que recibe el LLM
# ANTES de la pregunta del usuario. Define cómo se comporta el modelo:
# qué sabe, qué formato debe usar, y qué hacer con preguntas inesperadas.
#
# Puedes usar estos placeholders entre llaves — se rellenan automáticamente
# con información real del dataset cuando la app arranca:
#   {fecha_min}             → primera fecha del dataset
#   {fecha_max}             → última fecha del dataset
#   {plataformas}           → lista de plataformas (Android, iOS, etc.)
#   {reason_start_values}   → valores posibles de reason_start
#   {reason_end_values}     → valores posibles de reason_end
#
# IMPORTANTE: como el prompt usa llaves para los placeholders,
# si necesitas escribir llaves literales en el texto (por ejemplo para
# mostrar un JSON de ejemplo), usa doble llave: {{ y }}
#
SYSTEM_PROMPT = """
You are a very experienced data analyst capable of answering questions about a user's Spotify listening history by consulting the data you are provided. You can understand both English and Spanish. You receive questions in natural language and respond by generating Python code that creates visualizations using Plotly.

The data is in a pandas DataFrame called `df` with these columns:
- ts (datetime): timestamp of when the play ended
- track (str): song name
- artist (str): artist name
- album (str): album name
- track_uri (str): unique Spotify track identifier
- reason_start (str): why the play started. Values: {reason_start_values}
- reason_end (str): why the play ended. Values: {reason_end_values}
- shuffle (bool): whether shuffle mode was on
- skipped (bool): True if the song was skipped, False if it was not
- platform (str): device used. Values: {plataformas}
- ms_played (int): milliseconds played
- minutes_played (float): minutes played
- hours_played (float): hours played
- hour (int): hour of day (0-23)
- day_of_week (str): day name (Monday, Tuesday, etc.)
- month (int): month number (1-12)
- month_name (str): month name (January, February, etc.)
- year (int): year

The dataset covers from {fecha_min} to {fecha_max}.
Podcasts have been filtered out — only music tracks are included.

You must respond ONLY with a valid JSON object in this exact format, no extra text before or after:

For questions you can answer:
{{"tipo": "grafico", "codigo": "<python code>", "interpretacion": "<brief text explanation>"}}

For questions outside your scope:
{{"tipo": "fuera_de_alcance", "codigo": "", "interpretacion": "<explanation of why you cannot answer>"}}

Rules for the Python code you generate:
- The DataFrame is already loaded as `df`. Do not load or create data.
- Use Plotly Express (px) or Plotly Graph Objects (go) for all charts.
- The chart must be stored in a variable called `fig`.
- Always add clear titles, axis labels, and readable formatting.
- Use .head(10) or .head(20) for rankings to keep charts readable.
- For time-based questions, order results chronologically.
- When showing artists or songs, sort by the relevant metric (play count, hours, etc.).
- Round decimal numbers to 2 decimal places for readability.
- Use Spanish for chart titles and labels.
- In the interpretacion field, you MUST ALWAYS directly answer the user's question using the actual data. State the specific answer first with real numbers from the data (e.g., "Your most listened artist is Bad Bunny with 6,757 minutes"), then add brief context and short description of what the chart is illustrating if needed. Never use hypothetical examples.
- For comparison questions (e.g., "first semester vs second", "summer vs winter"), always create a grouped bar chart comparing the two periods side by side.
If the user asks something unrelated to their Spotify listening data (e.g., weather, politics, math), respond with tipo "fuera_de_alcance" and a polite explanation that you can only answer questions about their listening history.
Do not generate code that modifies the DataFrame.
Do not use print() statements — only create the fig variable.
CRITICAL: Your response must contain ONLY the JSON object. No explanations, no markdown, no text before or after the JSON. Even for complex comparison questions, respond with the JSON object only.
The interpretacion field must be a plain text string. Never include Python code, .format(), f-strings, or any code expressions inside the JSON values.
When the user asks to compare two periods, determine the appropriate months yourself:
- Summer (verano) = June, July, August (months 6, 7, 8)
- Winter (invierno) = December, January, February (months 12, 1, 2)
- First semester (primer semestre) = January to June (months 1-6)
- Second semester (segundo semestre) = July to December (months 7-12)
"""


# ============================================================
# CARGA Y PREPARACIÓN DE DATOS
# ============================================================
# Esta función se ejecuta UNA SOLA VEZ gracias a @st.cache_data.
# Lee el fichero JSON y prepara el DataFrame para que el código
# que genere el LLM sea lo más simple posible.
#
@st.cache_data
def load_data():
    df = pd.read_json("streaming_history.json")
    # 1. Convertir 'ts' de string a datetime
    df["ts"] = pd.to_datetime(df["ts"])

    # 2. Crear columnas derivadas (hora, día de la semana, mes...)
    df["hour"] = df["ts"].dt.hour
    df["day_of_week"] = df["ts"].dt.day_name()
    df["month"] = df["ts"].dt.month
    df["month_name"] = df["ts"].dt.month_name()
    df["year"] = df["ts"].dt.year

    # 3. Convertir milisegundos a minutos
    df["minutes_played"] = df["ms_played"] / 60000
    df["hours_played"] = df["ms_played"] / 3600000

    # 4. Renombrar columnas largas para simplificar el código generado
    df = df.rename(columns={
        "master_metadata_track_name": "track",
        "master_metadata_album_artist_name": "artist",
        "master_metadata_album_album_name": "album",
        "spotify_track_uri": "track_uri",
    })

    # 5. Filtrar registros que no aportan al análisis (podcasts, etc.)
    df = df[df["track"].notna()]

    # 6. Cambiar 'skipped' a True/False
    df["skipped"] = df["skipped"].fillna(False).astype(bool)
    return df


def build_prompt(df):
    """
    Inyecta información dinámica del dataset en el system prompt.
    Los valores que calcules aquí reemplazan a los placeholders
    {fecha_min}, {fecha_max}, etc. dentro de SYSTEM_PROMPT.

    Si añades columnas nuevas en load_data() y quieres que el LLM
    conozca sus valores posibles, añade aquí el cálculo y un nuevo
    placeholder en SYSTEM_PROMPT.
    """
    fecha_min = df["ts"].min()
    fecha_max = df["ts"].max()
    plataformas = df["platform"].unique().tolist()
    reason_start_values = df["reason_start"].unique().tolist()
    reason_end_values = df["reason_end"].unique().tolist()

    return SYSTEM_PROMPT.format(
        fecha_min=fecha_min,
        fecha_max=fecha_max,
        plataformas=plataformas,
        reason_start_values=reason_start_values,
        reason_end_values=reason_end_values,
    )


# ============================================================
# FUNCIÓN DE LLAMADA A LA API
# ============================================================
# Esta función envía DOS mensajes a la API de OpenAI:
# 1. El system prompt (instrucciones generales para el LLM)
# 2. La pregunta del usuario
#
# El LLM devuelve texto (que debería ser un JSON válido).
# temperature=0.2 hace que las respuestas sean más predecibles.
#
# No modifiques esta función.
#
def get_response(user_msg, system_prompt):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


# ============================================================
# PARSING DE LA RESPUESTA
# ============================================================
# El LLM devuelve un string que debería ser un JSON con esta forma:
#
#   {"tipo": "grafico",          "codigo": "...", "interpretacion": "..."}
#   {"tipo": "fuera_de_alcance", "codigo": "",    "interpretacion": "..."}
#
# Esta función convierte ese string en un diccionario de Python.
# Si el LLM envuelve el JSON en backticks de markdown (```json...```),
# los limpia antes de parsear.
#
# No modifiques esta función.
#
def parse_response(raw):
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

    return json.loads(cleaned)


# ============================================================
# EJECUCIÓN DEL CÓDIGO GENERADO
# ============================================================
# El LLM genera código Python como texto. Esta función lo ejecuta
# usando exec() y busca la variable `fig` que el código debe crear.
# `fig` debe ser una figura de Plotly (px o go).
#
# El código generado tiene acceso a: df, pd, px, go.
#
# No modifiques esta función.
#
def execute_chart(code, df):
    local_vars = {"df": df, "pd": pd, "px": px, "go": go}
    exec(code, {}, local_vars)
    return local_vars.get("fig")


# ============================================================
# INTERFAZ STREAMLIT
# ============================================================
# Toda la interfaz de usuario. No modifiques esta sección.
#

# Configuración de la página
st.set_page_config(page_title="Spotify Analytics", layout="wide")

# --- Control de acceso ---
# Lee la contraseña de secrets.toml. Si no coincide, no muestra la app.
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 Acceso restringido")
    pwd = st.text_input("Contraseña:", type="password")
    if pwd:
        if pwd == st.secrets["PASSWORD"]:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Contraseña incorrecta.")
    st.stop()

# --- App principal ---
st.title("🎵 Spotify Analytics Assistant")
st.caption("Pregunta lo que quieras sobre tus hábitos de escucha")

# Cargar datos y construir el prompt con información del dataset
df = load_data()
system_prompt = build_prompt(df)

# Caja de texto para la pregunta del usuario
if prompt := st.chat_input("Ej: ¿Cuál es mi artista más escuchado?"):

    # Mostrar la pregunta en la interfaz
    with st.chat_message("user"):
        st.write(prompt)

    # Generar y mostrar la respuesta
    with st.chat_message("assistant"):
        with st.spinner("Analizando..."):
            try:
                # 1. Enviar pregunta al LLM
                raw = get_response(prompt, system_prompt)

                # 2. Parsear la respuesta JSON
                parsed = parse_response(raw)

                if parsed["tipo"] == "fuera_de_alcance":
                    # Pregunta fuera de alcance: mostrar solo texto
                    st.write(parsed["interpretacion"])
                else:
                    # Pregunta válida: ejecutar código y mostrar gráfico
                    fig = execute_chart(parsed["codigo"], df)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                        st.write(parsed["interpretacion"])
                        st.code(parsed["codigo"], language="python")
                    else:
                        st.warning("El código no produjo ninguna visualización. Intenta reformular la pregunta.")
                        st.code(parsed["codigo"], language="python")

            except json.JSONDecodeError:
                st.error("No he podido interpretar la respuesta. Intenta reformular la pregunta.")
            except Exception as e:
                st.error("Ha ocurrido un error al generar la visualización. Intenta reformular la pregunta.")


# ============================================================
# REFLEXIÓN TÉCNICA (máximo 30 líneas)
# ============================================================
#
# Responde a estas tres preguntas con tus palabras. Sé concreto
# y haz referencia a tu solución, no a generalidades.
# No superes las 30 líneas en total entre las tres respuestas.
#
# 1. ARQUITECTURA TEXT-TO-CODE
#    ¿Cómo funciona la arquitectura de tu aplicación? ¿Qué recibe
#    el LLM? ¿Qué devuelve? ¿Dónde se ejecuta el código generado?
#    ¿Por qué el LLM no recibe los datos directamente?
#
#    La arquitectura de la aplicación es la siguiente: un LLM recibe una pregunta
#    de un usuario y tiene un system prompt de como debe contestar a esta pregunta
#    (cual es su role, como están estructurados los datos, que formato debe de tener
#    la respuesta). Luego, el LLM devuelve un JSON que contiene el código de python
#    que analiza los datos y una breve explicación al usuario. El código se ejecuta
#    en la aplicación, utilizando exec() contra la base de datos y sus 15000 registros
#    directamente. El LLM nunca accede a los datos, solo sabe la estructura y eso
#    ahorra créditos de llamada, es mas rapido y evita cálculos por el LLM que suelen
#    tener mas errores.
#
# 2. EL SYSTEM PROMPT COMO PIEZA CLAVE
#    ¿Qué información le das al LLM y por qué? Pon un ejemplo
#    concreto de una pregunta que funciona gracias a algo específico
#    de tu prompt, y otro de una que falla o fallaría si quitases
#    una instrucción.
#
#    El LLM recibe su rol, una descripción de cada una de las columnas del dataset
#   y como funcionan, el formato exacto con el que debe contestar (un JSON con tipo,
#   codigo e interpretacion), reglas muy claras y específicas para el código Python
#   (formato gráfico, títulos, decimales), y guardrails para manejar preguntas fuera
#   de alcance y no modificar los datos. Como ejemplo, una pregunta como "Compara mi
#   top 5 de artistas en verano vs invierno" funciona porque el prompt describe la
#   columna month (número del mes), que permite al LLM filtrar por meses de verano
#   [6,7,8] e invierno [12,1,2]. Y si quitáramos las instrucciones del formato de
#   respuesta JSON, la app directamente rompería — parse_response() espera un JSON
#   exacto con los campos tipo, codigo e interpretacion, y si el LLM responde con
#   texto libre, json.loads() falla y el usuario ve un error.
#
# 3. EL FLUJO COMPLETO
#    Describe paso a paso qué ocurre desde que el usuario escribe
#    una pregunta hasta que ve el gráfico en pantalla.
#
#    El flujo se inicia con un usuario insertando su pregunta en la ventana de chat,
#    luego load_data() ya subió y preparo el dataset, la función build_prompt() agrega
#    las fechas y plataformas en mi system prompt y la función get_response() envía el
#    system prompt junto con la pregunta del usuario al LLM (GPT-4.1-mini). Eso permite
#    que el LLM conteste con el JSON. La función parse_response() limpia y pasa el código
#    a un diccionario de Python. La aplicación (streamlit) averigua el tipo de pregunta,
#    si es 'tipo' "fuera de alcance" solo contesta con texto, si es tipo "grafico",
#    ejecuta el código y produce la figura. Por ultimo, el gráfico y la interpretación
#    aparecen para el usuario en la interfaz.