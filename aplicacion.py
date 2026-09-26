import os

os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st

# --- PARCHE DE COMPATIBILIDAD PARA STREAMLIT ---
if not hasattr(st, 'rerun'):
    st.rerun = st.experimental_rerun

import ssl
import urllib.request

# --- PARCHE SSL PARA DESCARGA DE PESOS EN PC NUEVA ---
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np
import mysql.connector
import json
import hashlib
import glob
import cv2
from datetime import datetime
from PIL import Image
import pillow_heif
pillow_heif.register_heif_opener()
import rawpy
import altair as alt
import pandas as pd

# --- PARCHE DE COMPATIBILIDAD KERAS-VIT ---
import tensorflow as tf
import keras

keras.utils.register_keras_serializable = tf.keras.utils.register_keras_serializable


class KerasOpsTraductor:
    def __getattr__(self, nombre_funcion):
        if nombre_funcion == 'concatenate': return tf.concat
        if nombre_funcion == 'transpose': return lambda x, axes=None, **kw: tf.transpose(x, perm=axes, **kw)
        if nombre_funcion == 'softmax': return tf.nn.softmax
        return getattr(tf, nombre_funcion)


keras.ops = KerasOpsTraductor()

# --- IMPORTACIONES IA ---
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_prep
from vit_keras import vit
from tensorflow.keras import layers, models


# --- 1. CONFIGURACIÓN INICIAL Y RUTAS ABSOLUTAS ---
DIRECTORIO_BASE = os.path.dirname(os.path.abspath(__file__))
ARCHIVO_USUARIOS = os.path.join(DIRECTORIO_BASE, 'db_medicos.json')
CARPETA_PACIENTES = os.path.join(DIRECTORIO_BASE, 'db_pacientes')
os.makedirs(CARPETA_PACIENTES, exist_ok=True)

st.set_page_config(page_title="MELIA", layout="wide")

diccionario_resultados = {
    'MEL': 'ALERTA: Alta sospecha de melanoma',
    'NV': 'Nevus (lunar benigno)',
    'BKL': 'ℹ Lesión benigna (queratosis/verruga)'
}

# --- EL DICCIONARIO REAL DESCUBIERTO Y UMBRALES ---
DICCIONARIO_IA = {0: 'MEL', 1: 'NV', 2: 'BKL'}
indice_mel = 0
UMBRAL_MEL = 0.45  # Umbral para detectar Melanoma
UMBRAL_CONFIANZA = 0.35  # Filtro de seguridad (Out-of-Distribution)


# --- 2. AUTENTICACIÓN (NUBE AUTOMÁTICA) ---
def conectar_bd():
    conexion = mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        port=st.secrets["mysql"]["port"],
        database=st.secrets["mysql"]["database"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"]
    )
    # Crear la tabla automáticamente si es un servidor nuevo
    cursor = conexion.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS medicos (
            id_medico VARCHAR(50) PRIMARY KEY,
            password_hash VARCHAR(255) NOT NULL
        )
    """)
    conexion.commit()
    return conexion


def encriptar(password):
    return hashlib.sha256(password.encode()).hexdigest()


if 'autenticado' not in st.session_state:
    st.session_state['autenticado'] = False
    st.session_state['medico'] = ''

if not st.session_state['autenticado']:
    st.title("Portal médico - diagnóstico dermatológico")
    t1, t2 = st.tabs(["Ingresar", "Registrar médico"])

    with t1:
        with st.form("login"):
            usr = st.text_input("ID médico")
            pwd = st.text_input("Contraseña", type="password")
            if st.form_submit_button("Ingresar", type="primary"):
                try:
                    conexion = conectar_bd()
                    cursor = conexion.cursor()
                    cursor.execute("SELECT password_hash FROM medicos WHERE id_medico = %s", (usr,))
                    resultado = cursor.fetchone()
                    conexion.close()

                    if resultado and resultado[0] == encriptar(pwd):
                        st.session_state['autenticado'], st.session_state['medico'] = True, usr
                        st.rerun()
                    else:
                        st.error("Credenciales inválidas o usuario no registrado.")
                except Exception as e:
                    st.error(f"Error de conexión a la nube: {e}")

    with t2:
        with st.form("registro"):
            n_usr = st.text_input("Nuevo ID médico")
            n_pwd = st.text_input("Crear contraseña", type="password")
            if st.form_submit_button("Registrar"):
                if n_usr.strip() == "" or n_pwd.strip() == "":
                    st.error("Los campos no pueden estar vacíos.")
                else:
                    try:
                        conexion = conectar_bd()
                        cursor = conexion.cursor()
                        cursor.execute(
                            "INSERT INTO medicos (id_medico, password_hash) VALUES (%s, %s)",
                            (n_usr.strip(), encriptar(n_pwd))
                        )
                        conexion.commit()
                        conexion.close()
                        st.success("Médico registrado exitosamente en la nube. Ya puede iniciar sesión.")
                    except mysql.connector.IntegrityError:
                        st.error("Error: Ese ID de médico ya existe. Elija otro.")
                    except Exception as e:
                        st.error(f"Error al registrar: {e}")
    st.stop()


# --- 3. CARGA DE MOTOR IA HÍBRIDO (VISIÓN + CLÍNICA) ---
@st.cache_resource(show_spinner="Cargando ecosistema multimodal...")
def cargar_ecosistema_ia():
    import os
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # 1. Leer metadatos usando la ruta absoluta para evitar FileNotFoundError
    df_all = pd.concat([
            pd.read_csv(os.path.join(BASE_DIR, 'metadata_trainCLINICO.csv')),
            pd.read_csv(os.path.join(BASE_DIR, 'metadata_valCLINICO.csv')),
            pd.read_csv(os.path.join(BASE_DIR, 'metadata_testCLINICO.csv'))
        ], ignore_index=True)
    df_all['edad_norm'] = df_all['edad'] / 100.0
    df_all = pd.get_dummies(df_all, columns=['sexo', 'localizacion'])
    meta_cols = ['edad_norm'] + [c for c in df_all.columns if c.startswith('sexo_') or c.startswith('localizacion_')]

    # 2. Cargar ResNet50 Multimodal (Nuevo formato nativo .keras)
    # Toda la reconstrucción manual de capas ha sido eliminada
    ruta_resnet = os.path.join(BASE_DIR, 'resnet50_multimodal_final.keras') # Asegúrate de que el nombre coincida con tu archivo
    modelo_resnet = tf.keras.models.load_model(ruta_resnet)

    # 3. Reconstruir ViT Multimodal
    input_img_vit = layers.Input(shape=(224, 224, 3), name='input_imagen_vit')
    base_vit = vit.vit_b16(image_size=224, activation='linear', pretrained=False, include_top=False, pretrained_top=False)
    x_img_vit = base_vit(input_img_vit)
    x_img_vit = layers.Dense(256, activation='relu')(x_img_vit)
    x_img_vit = layers.Dropout(0.5)(x_img_vit)

    input_meta_vit = layers.Input(shape=(len(meta_cols),), name='input_clinico_vit')
    x_meta_vit = layers.Dense(32, activation='relu')(input_meta_vit)
    x_meta_vit = layers.Dropout(0.75)(x_meta_vit)

    concat_vit = layers.Concatenate()([x_img_vit, x_meta_vit])
    x_final_vit = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(concat_vit)
    x_final_vit = layers.Dropout(0.5)(x_final_vit)
    pred_vit = layers.Dense(3, activation='softmax')(x_final_vit)

    modelo_vit = models.Model(inputs=[input_img_vit, input_meta_vit], outputs=pred_vit)
    modelo_vit.load_weights(os.path.join(BASE_DIR, 'vit_multimodal_v2.h5'))

    return modelo_resnet, modelo_vit, meta_cols

modelo_resnet, modelo_vit, meta_cols = cargar_ecosistema_ia()


# --- FUNCIONES DE PROCESAMIENTO DE IMAGEN ---
def limpiar_captura(imagen_pil):
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    h, w, _ = img_cv.shape
    cx, cy = w // 2, h // 2
    lado = int(min(h, w) * 0.50)
    mitad = lado // 2
    y_min, y_max = max(0, cy - mitad), min(h, cy + mitad)
    x_min, x_max = max(0, cx - mitad), min(w, cx + mitad)
    recorte = img_cv[y_min:y_max, x_min:x_max]
    if recorte.size == 0: return imagen_pil
    return Image.fromarray(cv2.cvtColor(recorte, cv2.COLOR_BGR2RGB))


def validar_tejido_cutaneo(img_rgb):
    # 1. Extraer el 60% central para tener contexto de la lesión y la "piel" de fondo,
    # evadiendo las esquinas oscuras del dermatoscopio.
    h, w, _ = img_rgb.shape
    recorte = img_rgb[int(h * 0.2):int(h * 0.8), int(w * 0.2):int(w * 0.8)]

    # 2. Medir saturación general (debe tener ALGO de color)
    hsv = cv2.cvtColor(recorte, cv2.COLOR_RGB2HSV)
    saturacion_media = np.mean(hsv[:, :, 1])

    # 3. Calcular la dominancia térmica de los canales de color
    r = np.mean(recorte[:, :, 0])
    g = np.mean(recorte[:, :, 1])
    b = np.mean(recorte[:, :, 2])

    # 4. Desviación estándar de los canales (busca acromatismo)
    varianza_colores = np.std([r, g, b])

    # REGLAS DE RECHAZO CLÍNICO:
    # A) Es una imagen en escala de grises/plástico (varianza muy baja, < 10.0)
    # B) Es un objeto sin vida/color (saturación bajísima, < 10)
    # C) El rojo no es el color dominante (la piel real SIEMPRE refleja más rojo)

    if varianza_colores < 10.0 or saturacion_media < 10 or r <= b or r <= g:
        return False

    return True



def calcular_area_lesion(ruta_imagen):
    img = cv2.imread(ruta_imagen)
    if img is None: return 0, None, None
    img = cv2.resize(img, (500, 500))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 1. Usar el canal de Luminosidad (L) del espacio LAB (Inmune al tono de piel)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_blur = cv2.GaussianBlur(l, (21, 21), 0)

    # 2. Analizar únicamente el cuadro central
    h, w = l_blur.shape
    centro = l_blur[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]

    # 3. Matemática relativa: Buscar el punto más oscuro y la piel promedio
    min_val, _, _, _ = cv2.minMaxLoc(centro)  # El punto más negro (el lunar)
    mean_val = np.mean(centro)  # El promedio (la piel)

    # El umbral perfecto es el punto intermedio (tirando un poco hacia el lunar)
    umbral = min_val + ((mean_val - min_val) * 0.45)

    # 4. Crear la máscara aislando solo lo oscuro
    _, mascara = cv2.threshold(l_blur, umbral, 255, cv2.THRESH_BINARY_INV)

    # 5. Limpieza morfológica
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mascara_limpia = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    mascara_limpia = cv2.morphologyEx(mascara_limpia, cv2.MORPH_CLOSE, kernel)

    # 6. Buscar contornos y aplicar FILTRO GEOCÉNTRICO
    contornos, _ = cv2.findContours(mascara_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos: return 0, np.zeros_like(mascara_limpia), img_rgb

    centro_x, centro_y = w // 2, h // 2
    mejor_contorno = None
    menor_distancia = float('inf')

    # De todos los objetos detectados, elegimos SOLO el más central
    for c in contornos:
        if cv2.contourArea(c) > 30:  # Ignorar ruido diminuto
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                distancia = ((cx - centro_x) ** 2 + (cy - centro_y) ** 2) ** 0.5

                if distancia < menor_distancia:
                    menor_distancia = distancia
                    mejor_contorno = c

    if mejor_contorno is None: return 0, np.zeros_like(mascara_limpia), img_rgb

    # 7. Crear una máscara negra limpia y dibujar SOLO el lunar ganador en blanco
    mascara_final = np.zeros_like(mascara_limpia)
    cv2.drawContours(mascara_final, [mejor_contorno], -1, 255, thickness=cv2.FILLED)

    area = cv2.contourArea(mejor_contorno)
    cv2.drawContours(img_rgb, [mejor_contorno], -1, (0, 255, 0), 2)

    return area, mascara_final, img_rgb


def eliminar_vello(imagen_pil):
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    gris = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # 1. Kernel elíptico más grande para abarcar vellos gruesos
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
    bottom_hat = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, kernel)

    # 2. Desenfoque para conectar los trozos de vello separados
    bottom_hat_blur = cv2.GaussianBlur(bottom_hat, (3, 3), 0)

    # 3. Umbral adaptado
    _, mascara = cv2.threshold(bottom_hat_blur, 15, 255, cv2.THRESH_BINARY)

    # 4. Engrosar la máscara para que la restauración cubra las sombras del vello
    mascara = cv2.dilate(mascara, np.ones((3, 3), np.uint8), iterations=1)

    # 5. Restauración Telea con mayor radio de acción
    img_restaurada = cv2.inpaint(img_cv, mascara, 5, cv2.INPAINT_TELEA)
    return Image.fromarray(cv2.cvtColor(img_restaurada, cv2.COLOR_BGR2RGB))


def aplicar_clahe(imagen_pil):
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return Image.fromarray(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))


def aislar_canal_azul(imagen_pil):
    img_cv = cv2.cvtColor(np.array(imagen_pil), cv2.COLOR_RGB2BGR)
    b, g, r = cv2.split(img_cv)
    return Image.fromarray(b).convert("RGB")


# --- 4. INTERFAZ CLÍNICA PRINCIPAL ---
st.sidebar.success(f" Dr. {st.session_state['medico']}")
if st.sidebar.button("Cerrar sesión"):
    st.session_state['autenticado'] = False
    st.rerun()

st.title(" Sistema de triaje dermatológico")

tab_analisis, tab_historial, tab_evolucion = st.tabs(
    [" Nuevo análisis", " Historial de pacientes", " Evolución clínica"])

with tab_analisis:
    col_datos, col_escaneo = st.columns([1, 1])

    with col_datos:
        st.subheader("1. Datos del paciente")
        nombre = st.text_input("Nombre completo")
        edad = st.number_input("Edad (Expediente)", min_value=1, max_value=120, value=45)
        sexo_ui = st.selectbox("Sexo", ["Femenino", "Masculino"])
        loc_ui = st.selectbox("Localización de la lesión", [
            "Extremidad superior (Brazos)", "Extremidad inferior (Piernas)",
            "Torso anterior (Pecho/Abdomen)", "Torso posterior (Espalda)",
            "Cabeza / Cuello", "Cara", "Palmas / Plantas", "Oral / Genital",
            "Torso lateral"
        ])
        fototipo_ui = st.selectbox("Fototipo de piel (fitzpatrick)", ["1", "2", "3", "4", "5", "6"])

    with col_escaneo:
        st.subheader("2. Captura macroscópica")

        if 'imagen_actual' not in st.session_state:
            st.session_state['imagen_actual'] = None

        st.info("Haz clic abajo para subir una imagen desde tu PC o seleccionar una foto de la galería.")
        archivo_capturado = st.file_uploader("Cargar fotografía de la lesión",
                                             type=['jpg', 'jpeg', 'png', 'heic', 'heif', 'webp', 'dng'])

        if archivo_capturado:
            # Revisamos si el archivo es un DNG
            if archivo_capturado.name.lower().endswith('.dng'):
                # Revelado especial del RAW
                with rawpy.imread(archivo_capturado) as raw:
                    rgb = raw.postprocess()
                img_pil = Image.fromarray(rgb)
            else:
                # Lectura normal para JPG, HEIC, PNG...
                img_pil = Image.open(archivo_capturado).convert('RGB')

            st.session_state['imagen_actual'] = img_pil
        else:
            st.session_state['imagen_actual'] = None

        imagen_cargada = st.session_state['imagen_actual']

        if imagen_cargada:
            st.markdown("---")
            st.subheader("3. Procesamiento digital interactivo")

            col_controles1, col_controles2 = st.columns(2)
            with col_controles1:
                st.markdown("**Ajustes geométricos**")
                aplicar_recorte = st.checkbox(" Recortar marco del dispositivo (zoom clínico)", value=False)
                aplicar_depilacion = st.checkbox(" Depilación digital", value=False)
            with col_controles2:
                st.markdown("**Filtros dermatoscópicos**")
                filtro_avanzado = st.radio("Filtro visual:",
                                           ["Ninguno", "Contraste local (CLAHE)", "Aislamiento de canal"])

            # --- PROCESAMIENTO VISUAL EN TIEMPO REAL ---
            img_para_medico = imagen_cargada.copy()
            img_procesada = imagen_cargada.copy()  # La base limpia que usará la IA

            if aplicar_recorte:
                img_para_medico = limpiar_captura(img_para_medico)
                img_procesada = limpiar_captura(img_procesada)

            if aplicar_depilacion:
                img_para_medico = eliminar_vello(img_para_medico)
                img_procesada = eliminar_vello(img_procesada)

            if filtro_avanzado == "Contraste local (CLAHE)":
                img_para_medico = aplicar_clahe(img_para_medico)
            elif filtro_avanzado == "Aislamiento de canal":
                img_para_medico = aislar_canal_azul(img_para_medico)

            # Mostrar las imágenes dinámicamente antes de enviar a la IA
            c1, c2 = st.columns(2)
            c1.image(imagen_cargada, caption="Fotografía Original", use_column_width=True)
            c2.image(img_para_medico, caption="Vista Clínica Filtrada", use_column_width=True)

            st.markdown("---")

            # --- BOTÓN DE ANÁLISIS E INFERENCIA ---
            if st.button("Analizar y Guardar Expediente", type="primary"):
                if not nombre or loc_ui == "":
                    st.error(" ALERTA: Complete nombre y localización.")
                else:
                    # --- FILTRO BIOLÓGICO Y LLAMADA A LA IA ---
                    if not validar_tejido_cutaneo(np.array(img_procesada)):
                        st.error("ALERTA DE SEGURIDAD: Fotografía rechazada por el pre-filtro biológico.")
                        st.warning(
                            "El escáner no detectó propiedades compatibles con tejido cutáneo (ausencia de pigmentación humana). El análisis de Inteligencia Artificial ha sido bloqueado.")
                    else:
                        with st.spinner("Ejecutando ensamble multimodal..."):
                            # 1. Preparar imagen visual (TTA)
                            img_rgb_arr = np.array(img_procesada)
                            img_rgb_arr = cv2.resize(img_rgb_arr, (224, 224))
                            lote_crudo = np.array([img_rgb_arr, cv2.flip(img_rgb_arr, 1), cv2.flip(img_rgb_arr, 0)],
                                                  dtype=np.float32)

                            # 2. Preparar vector clínico
                            traductor_sexo = {"Femenino": "female", "Masculino": "male", "Desconocido": "unknown"}
                            traductor_loc = {
                                "Extremidad superior (Brazos)": "upper extremity",
                                "Extremidad inferior (Piernas)": "lower extremity",
                                "Torso anterior (Pecho/Abdomen)": "anterior torso",
                                "Torso posterior (Espalda)": "posterior torso",
                                "Cabeza / Cuello": "head/neck",
                                "Palmas / Plantas": "palms/soles",
                                "Oral / Genital": "oral/genital",
                                "Torso lateral": "lateral torso",
                                "Cara": "head/neck"
                            }

                            vector_clinico = np.zeros(len(meta_cols), dtype=np.float32)
                            vector_clinico[0] = edad / 100.0  # Edad normalizada

                            col_sexo = f"sexo_{traductor_sexo.get(sexo_ui, 'unknown')}"
                            if col_sexo in meta_cols: vector_clinico[meta_cols.index(col_sexo)] = 1.0

                            col_loc = f"localizacion_{traductor_loc.get(loc_ui, 'unknown')}"
                            if col_loc in meta_cols: vector_clinico[meta_cols.index(col_loc)] = 1.0

                            lote_clinico = np.array([vector_clinico, vector_clinico, vector_clinico], dtype=np.float32)

                            # 3. Inferencia de Doble Rama
                            lote_resnet = resnet_prep(lote_crudo.copy())
                            preds_res = modelo_resnet.predict([lote_resnet, lote_clinico], verbose=0)

                            lote_vit_inputs = vit.preprocess_inputs(lote_crudo.copy())
                            preds_vi = modelo_vit.predict([lote_vit_inputs, lote_clinico], verbose=0)

                            # 4. Promedio Final
                            pred_promedio_resnet = np.mean(preds_res, axis=0)
                            pred_promedio_vit = np.mean(preds_vi, axis=0)
                            prob_promedio = (pred_promedio_resnet + pred_promedio_vit) / 2.0

                        # --- FILTRO ANTI-BASURA (OOD DETECTION) ---
                        confianza_maxima = np.max(prob_promedio)
                        if confianza_maxima < UMBRAL_CONFIANZA:
                            st.error("ALERTA DE SEGURIDAD: Imagen no reconocida.")
                            st.warning(
                                "El motor de inteligencia artificial no ha detectado patrones compatibles con tejido cutáneo o lesiones dermatológicas válidas. Para evitar diagnósticos erróneos o usos indebidos del sistema, el análisis ha sido bloqueado. Por favor, asegúrese de subir una fotografía clara de la piel.")
                        else:
                            # --- DECISIÓN CLÍNICA Y GUARDADO ---
                            if prob_promedio[indice_mel] >= UMBRAL_MEL:
                                indice_clase = indice_mel
                            else:
                                prob_sin_mel = prob_promedio.copy()
                                prob_sin_mel[indice_mel] = 0.0
                                indice_clase = np.argmax(prob_sin_mel)

                            pred_cruda = DICCIONARIO_IA[indice_clase]
                            diagnostico_final = diccionario_resultados.get(pred_cruda, pred_cruda)

                            timestamp_actual = int(datetime.now().timestamp())
                            nombre_base = f"exp_{nombre.replace(' ', '_')}_{timestamp_actual}"
                            ruta_imagen_guardada = os.path.join(CARPETA_PACIENTES, f"{nombre_base}.jpg")
                            img_procesada.save(ruta_imagen_guardada)

                            expediente = {
                                "medico": st.session_state['medico'],
                                "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "paciente": {"nombre": nombre, "edad": edad, "sexo": sexo_ui, "fototipo": fototipo_ui},
                                "lesion": {"localizacion": loc_ui, "diagnostico_crudo": pred_cruda,
                                           "diagnostico_completo": diagnostico_final,
                                           "ruta_imagen": ruta_imagen_guardada,
                                           "probabilidades": prob_promedio.tolist()}  # <-- NUEVA LÍNEA AÑADIDA
                            }

                            with open(os.path.join(CARPETA_PACIENTES, f"{nombre_base}.json"), 'w') as f:
                                json.dump(expediente, f, indent=4)

                            st.success(" Análisis híbrido completado.")
                            st.markdown(f"### Resultado del triaje: **{diagnostico_final}**")

                            with st.expander("Ver desglose de probabilidades visuales", expanded=True):
                                df_probs = pd.DataFrame({
                                    'Diagnóstico': ['1. Melanoma (MEL)', '2. Nevus (NV)',
                                                    '3. Queratosis/Verruga (BKL)'],
                                    'Probabilidad (%)': [prob_promedio[0] * 100, prob_promedio[1] * 100,
                                                         prob_promedio[2] * 100],
                                    'Color': ['#ef4444', '#22c55e', '#3b82f6']
                                })
                                barras = alt.Chart(df_probs).mark_bar(cornerRadiusEnd=4, height=30).encode(
                                    x=alt.X('Probabilidad (%):Q', scale=alt.Scale(domain=[0, 100]),
                                            title="Confianza (%)"),
                                    y=alt.Y('Diagnóstico:N', sort='-x', title=None),
                                    color=alt.Color('Color:N', scale=None),
                                    tooltip=[alt.Tooltip('Diagnóstico:N'),
                                             alt.Tooltip('Probabilidad (%):Q', format='.2f')]
                                )
                                linea_umbral = alt.Chart(pd.DataFrame({'x': [UMBRAL_MEL * 100]})).mark_rule(
                                    color='#ef4444', strokeDash=[5, 5], size=2).encode(x='x:Q')
                                st.altair_chart((barras + linea_umbral).properties(height=200), use_container_width=True)
                                st.caption(f"*Línea roja: Umbral clínico de detección ({UMBRAL_MEL * 100}%).*")
                                st.caption(
                                    "*Este resultado combina la precisión local (ResNet) y el análisis global (Transformer).*")

with tab_historial:
    st.subheader(f"Expedientes clínicos - Dr. {st.session_state['medico']}")
    if 'expediente_activo' not in st.session_state: st.session_state['expediente_activo'] = None

    # Filtrar archivos para que solo aparezcan los del médico logueado
    archivos_todos = glob.glob(os.path.join(CARPETA_PACIENTES, "*.json"))
    expedientes_del_medico = []
    for arch in archivos_todos:
        try:
            d_temp = json.load(open(arch))
            if d_temp.get('medico') == st.session_state['medico']:
                d_temp['ruta_json_fisica'] = arch
                expedientes_del_medico.append(d_temp)
        except:
            pass

    if not expedientes_del_medico:
        st.info("No tiene pacientes registrados en su base de datos.")
    else:
        if st.session_state['expediente_activo']:
            datos = st.session_state['expediente_activo']
            p = datos.get('paciente', {})
            l = datos.get('lesion', {})

            st.markdown(f"### Paciente: {p.get('nombre', 'Desconocido')}")
            col_info, col_img = st.columns([1, 1])

            with col_info:
                st.markdown("#### Datos Demográficos y Clínicos")
                st.markdown(f"""
                - **Fecha de estudio:** {datos.get('fecha', 'Sin fecha')}
                - **Edad:** {p.get('edad', 'N/A')} años
                - **Sexo:** {p.get('sexo', 'N/A')}
                - **Fototipo (Fitzpatrick):** {p.get('fototipo', 'N/A')}
                - **Localización anatómica:** {l.get('localizacion', 'N/A')}
                """)
                st.divider()
                st.markdown(f"#### Diagnóstico Final: **{l.get('diagnostico_completo', 'N/A')}**")

                # Desplegar gráfica si existe en el JSON
                probs = l.get('probabilidades')
                if probs and len(probs) == 3:
                    df_probs = pd.DataFrame({
                        'Diagnóstico': ['MEL', 'NV', 'BKL'],
                        'Probabilidad (%)': [probs[0] * 100, probs[1] * 100, probs[2] * 100],
                        'Color': ['#ef4444', '#22c55e', '#3b82f6']
                    })
                    barras = alt.Chart(df_probs).mark_bar(cornerRadiusEnd=4, height=20).encode(
                        x=alt.X('Probabilidad (%):Q', scale=alt.Scale(domain=[0, 100]), title="Confianza (%)"),
                        y=alt.Y('Diagnóstico:N', sort='-x', title=None),
                        color=alt.Color('Color:N', scale=None)
                    )
                    st.altair_chart(barras.properties(height=150), use_container_width=True)

                st.divider()
                if st.button("← Regresar al listado general"):
                    st.session_state['expediente_activo'] = None
                    st.rerun()

            with col_img:
                ruta_img = l.get('ruta_imagen')
                if ruta_img and os.path.exists(ruta_img):
                    img_historial = Image.open(ruta_img)
                    st.image(img_historial, use_column_width=True, caption="Captura almacenada")
        else:
            # Ordenar del más reciente al más antiguo
            expedientes_del_medico.sort(key=lambda x: x.get('fecha', ''), reverse=True)
            for datos in expedientes_del_medico:
                col_resumen, col_boton = st.columns([4, 1])
                with col_resumen:
                    st.markdown(
                        f" **{datos.get('fecha', '')}** |  **{datos.get('paciente', {}).get('nombre', 'Desconocido')}** - {datos.get('lesion', {}).get('diagnostico_completo', '')}")
                with col_boton:
                    if st.button("Abrir Expediente", key=datos['ruta_json_fisica']):
                        st.session_state['expediente_activo'] = datos
                        st.rerun()
                st.divider()

with tab_evolucion:
    st.subheader("Análisis longitudinal de evolución")

    # 1. Filtrar solo pacientes del médico actual
    archivos_todos = glob.glob(os.path.join(CARPETA_PACIENTES, "*.json"))
    expedientes_medico = [json.load(open(a)) for a in archivos_todos if
                          json.load(open(a)).get('medico') == st.session_state['medico']]
    pacientes_nombres = {e['paciente']['nombre'] for e in expedientes_medico if 'paciente' in e}

    if not pacientes_nombres:
        st.info("No hay pacientes registrados con su perfil para analizar.")
    else:
        paciente_seleccionado = st.selectbox("Seleccione paciente:", sorted(list(pacientes_nombres)))
        visitas = sorted(
            [e for e in expedientes_medico if e.get('paciente', {}).get('nombre') == paciente_seleccionado],
            key=lambda x: x.get('fecha', ''))

        if len(visitas) < 2:
            st.warning("Se requieren al menos 2 estudios del mismo paciente para evaluar crecimiento.")
        else:
            fechas_disponibles = [v.get('fecha') for v in visitas]
            st.markdown("Seleccione las fechas que desea incluir en el seguimiento de este paciente:")

            # Selector múltiple (por defecto toma la más vieja y la más nueva)
            fechas_seleccionadas = st.multiselect("Estudios a comparar:", fechas_disponibles,
                                                  default=[fechas_disponibles[0], fechas_disponibles[-1]])

            if len(fechas_seleccionadas) >= 2:
                fechas_seleccionadas.sort()
                fecha_base = fechas_seleccionadas[0]
                fecha_actual = fechas_seleccionadas[-1]

                # Lógica Dermatológica: Revisión de tiempo
                d_base = datetime.strptime(fecha_base, "%Y-%m-%d %H:%M:%S")
                d_actual = datetime.strptime(fecha_actual, "%Y-%m-%d %H:%M:%S")
                dias_transcurridos = (d_actual - d_base).days

                bloqueo_clinico = False
                forzar_analisis = False

                if dias_transcurridos < 30:
                    st.warning(
                        " **Protocolo dermatológico:** El tiempo transcurrido entre el estudio inicial y el final es muy corto (menor a 1 mes). Los cambios estructurales significativos en tejido pigmentado requieren un periodo de observación estándar de 3 a 6 meses.")
                    bloqueo_clinico = True
                    forzar_analisis = st.checkbox("Forzar análisis computacional (omisión clínica)")

                if not bloqueo_clinico or forzar_analisis:
                    if st.button("Ejecutar análisis de área multitemporal", type="primary"):
                        areas = []
                        fechas_graf = []

                        # Extraer áreas de todas las selecciones
                        for fecha in fechas_seleccionadas:
                            exp = next(v for v in visitas if v['fecha'] == fecha)
                            ruta = exp.get('lesion', {}).get('ruta_imagen')
                            area, _, _ = calcular_area_lesion(ruta)
                            areas.append(area)
                            fechas_graf.append(fecha.split(" ")[0])  # Solo tomar el día para la gráfica

                        area_inicial = areas[0]
                        area_final = areas[-1]

                        if area_inicial == 0 or area_final == 0:
                            st.error("Error de segmentación en las imágenes seleccionadas.")
                        else:
                            variacion_total = ((area_final - area_inicial) / area_inicial) * 100

                            # Mostrar comparativa visual de los extremos
                            st.markdown("### Comparativa de extremos")
                            col_img1, col_img2, col_metricas = st.columns([1.5, 1.5, 1])

                            exp_base = next(v for v in visitas if v['fecha'] == fecha_base)
                            exp_actual = next(v for v in visitas if v['fecha'] == fecha_actual)
                            _, mask_base, img_base = calcular_area_lesion(exp_base['lesion']['ruta_imagen'])
                            _, mask_actual, img_actual = calcular_area_lesion(exp_actual['lesion']['ruta_imagen'])

                            with col_img1:
                                st.image(img_base, caption=f"Base: {fechas_graf[0]}", use_column_width=True)
                            with col_img2:
                                st.image(img_actual, caption=f"Actual: {fechas_graf[-1]}", use_column_width=True)
                            with col_metricas:
                                st.metric("Variación total", f"{variacion_total:+.2f}%")
                                if variacion_total > 15.0:
                                    st.error("ALERTA CLÍNICA: Crecimiento acelerado")
                                    st.warning(
                                        f"La lesión ha incrementado su tamaño en un *{variacion_total:.2f}%*. Según el criterio 'E' (Evolución) de la regla ABCDE, los cambios rápidos en la extensión geométrica son un marcador de alto riesgo de malignidad. Requiere revisión dermatoscópica urgente.")
                                elif variacion_total < -10.0:
                                    st.success("Reducción detectada")
                                    st.info(
                                        "El área ha disminuido. Común en lesiones inflamatorias que se resuelven o en nevos que entran en fase de regresión fibrótica.")
                                else:
                                    st.info("Geometría estable")
                                    st.caption(
                                        "No se detectan cambios morfológicos significativos. La variación se encuentra dentro del margen de error natural del tejido y el escáner.")

                            # Gráfica de tendencia multitemporal
                            st.markdown("### Tendencia de crecimiento")
                            df_tendencia = pd.DataFrame({"Fecha": fechas_graf, "Área estructural (píxeles)": areas})
                            linea_tendencia = alt.Chart(df_tendencia).mark_line(point=True, color='#ef4444').encode(
                                x=alt.X('Fecha:N', title='Fecha de Consulta'),
                                y=alt.Y('Área estructural (píxeles):Q', scale=alt.Scale(zero=False)),
                                tooltip=['Fecha', 'Área estructural (píxeles)']
                            )
                            st.altair_chart(linea_tendencia.properties(height=250), use_container_width=True)
