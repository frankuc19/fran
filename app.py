# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import traceback
from PIL import Image
import os

# --- Configuración de Página (solo una vez) ---
# Debe ser la primera llamada a Streamlit
st.set_page_config(
    page_title="PreRoute | Transvip", # Título más descriptivo en la pestaña
    page_icon="🚀",
    layout="wide"
)

# --- Título y navegación lateral ---
st.title("Bienvenido a PreRoute de Transvip")
st.sidebar.success("Ajusta los parámetros necesarios para realizar el ruteo.")

st.markdown(
    """
    Esta aplicación asigna reservas de transporte a móviles disponibles
    según parámetros configurables y reglas de negocio.
    Sube los archivos CSV requeridos y haz clic en 'Ejecutar Asignación'.
    """
) # Descripción más útil

# --- Logo y Título ---
LOGO_PATH = "transvip.png"  # Ruta relativa
LOGO_WIDTH = 90
COLUMN_RATIO = [12, 1] # Ajusta según preferencia visual

# Usar un bloque try-except más específico para la carga del logo
# y asegurar que el título siempre se muestre.
try:
    col_title, col_logo = st.columns(COLUMN_RATIO)
    with col_title:
        st.title("PreRoute 2.0") # Título principal de la app visible
    with col_logo:
        if os.path.exists(LOGO_PATH):
            try:
                logo_image = Image.open(LOGO_PATH)
                st.image(logo_image, width=LOGO_WIDTH)
            except Exception as e:
                st.warning(f"⚠️ No se pudo cargar la imagen del logo '{LOGO_PATH}': {e}")
        else:
            st.warning(f"⚠️ Logo no encontrado en '{LOGO_PATH}'.")
except Exception as e:
    st.warning(f"No se pudo crear el layout para el título y logo: {e}")
    # Mostrar el título igualmente si las columnas fallan
    st.title("PreRoute 2.0")

# --- Constantes ---
RADIO_TIERRA_KM = 6371
PRECISION_SIMULATE_H3 = 3 # Precisión para simular H3 (más decimales = más granularidad)

# Intervalos base en minutos entre servicios
INTERVALO_CAMBIO_INTERREGIONAL = 270 # Intervalo al cambiar de/a Interregional o Divisiones
INTERVALO_URBANO_NOCTURNO = 70      # Intervalo para Urbano entre 00:00 y 05:59
INTERVALO_URBANO_DIURNO = 80       # Intervalo para Urbano fuera de horario nocturno
INTERVALO_GENERAL = 80             # Intervalo por defecto para otras categorías
INTERVALO_MIN_DEFAULT_FACTOR = 1.5 # Factor para ajustar intervalo mínimo basado en tiempo de viaje estimado

# Reglas de negocio por móvil
MAX_INTERREGIONALES_POR_MOVIL = 2
MAX_OTRAS_DIVISIONES_POR_MOVIL = 2 # Máximo de categorías únicas que NO son Interregional ni Urbano

# Columnas requeridas en los archivos CSV
REQUIRED_HIST_COLS = [
    'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'tiempoestimada'
]
REQUIRED_PRED_COLS_ORIGINAL = [
    'pickup_datetime', 'job_id', 'estimated_payment',
    'Categoria_viaje', 'latrecogida', 'lonrecogida',
    'latdestino', 'londestino'
    # Añadir 'Convenio', 'Tipo_servicio' si son *siempre* requeridas
    # 'Convenio', 'Tipo_servicio'
]
RENAME_MAP_PRED = {
    'pickup_datetime': 'HoraFecha',
    'job_id': 'reserva',
    # Añadir aquí otros mapeos si son necesarios
}
# Construir lista de columnas renombradas dinámicamente
# Incluye las claves renombradas (nuevos nombres) y las columnas que no se renombran
REQUIRED_PRED_COLS_RENAMED = list(RENAME_MAP_PRED.values()) + [
    col for col in REQUIRED_PRED_COLS_ORIGINAL if col not in RENAME_MAP_PRED.keys()
]

# --- Parámetros configurables por el usuario ---
st.sidebar.header("Parámetros de Asignación")
max_moviles_param = st.sidebar.slider('Máximo de Móviles:', min_value=1, max_value=500, value=100, step=10) # Ajuste step
max_monto_param = st.sidebar.slider('Monto Máximo por Móvil ($):', min_value=100000, max_value=1000000, value=500000, step=50000)
max_reservas_param = st.sidebar.slider('Máximo de Reservas por Móvil:', min_value=1, max_value=20, value=5) # Aumentado rango y default
max_horas_param = st.sidebar.slider('Máximo de Horas por Ruta (desde 1ra recogida):', min_value=1, max_value=24, value=10)

# --- Funciones Auxiliares ---
def check_columns(df, required_columns, filename):
    """Verifica si todas las columnas requeridas existen en el DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Error Crítico: Faltan columnas en '{filename}': {', '.join(missing_cols)}.")
        st.stop() # Detiene la ejecución si faltan columnas esenciales

def haversine_vectorized(lat1, lon1, lat2, lon2):
    """Calcula la distancia Haversine entre dos puntos (o arrays de puntos)."""
    # Convertir a radianes
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Diferencias
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Fórmula Haversine
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    # Resultado en kilómetros
    return RADIO_TIERRA_KM * c

def simulate_h3_vectorized(lats, lons, precision=PRECISION_SIMULATE_H3):
    """Crea un identificador de 'celda' simple basado en coordenadas redondeadas."""
    # Asegurar que sean numéricos, convertir errores a NaN
    lats = pd.to_numeric(lats, errors='coerce')
    lons = pd.to_numeric(lons, errors='coerce')
    # Redondear y concatenar como string. Manejar NaNs para que no rompan el string.
    return lats.round(precision).astype(str) + "_" + lons.round(precision).astype(str)

def calcular_intervalo(ultima_reserva, nueva_reserva):
    """
    Determina el intervalo de tiempo base requerido entre dos reservas.
    Retorna: (string: tipo de intervalo, int: minutos de intervalo base)
    """
    cat_nueva = nueva_reserva.get("Categoria_viaje", "Desconocida")
    # La categoría de la última reserva ya no es necesaria aquí si el chequeo principal
    # se basa en la llegada de la última y recogida de la nueva.
    # cat_ultima = ultima_reserva.get("Categoria_viaje", "Desconocida") # No usada directamente aquí
    hora_nueva = nueva_reserva.get("HoraFecha") # Hora de recogida de la nueva reserva

    if pd.isna(hora_nueva):
        return "Error Hora", 99999 # Intervalo muy grande para evitar asignación

    # Lógica simplificada: el intervalo depende principalmente de la *nueva* reserva
    # y si hubo un cambio "mayor" (Interregional/Divisiones)
    # NOTA: Asume que la lógica de cambio de categoría es simétrica
    # (Urbano -> Interregional tiene el mismo intervalo que Interregional -> Urbano)
    if cat_nueva in ["Interregional", "Divisiones"]:
        return "Cambio/Especial", INTERVALO_CAMBIO_INTERREGIONAL

    if cat_nueva == "Urbano":
        # Chequeo de hora nocturna (00:00 a 05:59)
        if 0 <= hora_nueva.hour < 6:
            return "Urbano nocturno", INTERVALO_URBANO_NOCTURNO
        else:
            return "Urbano diurno", INTERVALO_URBANO_DIURNO

    # Para cualquier otra categoría o si es desconocida
    return "General", INTERVALO_GENERAL

def monto_total_movil(movil_reservas):
    """Calcula el monto total acumulado de las reservas de un móvil."""
    monto = 0
    for r in movil_reservas:
        pago = r.get("estimated_payment", 0)
        if pd.notnull(pago): # Solo sumar si no es NaN
            monto += pago
    return monto

def puede_agregarse_a_movil(movil_reservas, nueva_reserva):
    """
    Verifica si una nueva reserva puede ser añadida a la ruta existente de un móvil.
    Retorna: (bool: puede agregarse, string: tipo intervalo, int: intervalo aplicado, string: motivo rechazo)
    """
    # --- Chequeo 1: Máximo de reservas por móvil ---
    if len(movil_reservas) >= max_reservas_param:
        return False, None, None, f"Máximo de {max_reservas_param} reservas alcanzado"

    # --- Obtener datos relevantes (con manejo de faltantes) ---
    ultima_reserva = movil_reservas[-1]
    nueva_hora_recogida = nueva_reserva.get("HoraFecha")
    nueva_monto = nueva_reserva.get("estimated_payment", 0) # Default 0 si falta
    nueva_cat = nueva_reserva.get("Categoria_viaje", "Desconocida")
    nueva_tiempo_viaje_estimado = nueva_reserva.get("avg_travel_time") # Puede ser NaN

    ultima_hora_llegada_estimada = ultima_reserva.get("estimated_arrival") # Hora estimada de llegada del último servicio

    # Validar datos críticos de la nueva reserva
    if pd.isna(nueva_hora_recogida) or nueva_monto is None: # Chequear None para monto tambien
        return False, None, None, "Datos inválidos (hora/monto) en nueva reserva"

    # Validar datos críticos de la última reserva del móvil
    if pd.isna(ultima_hora_llegada_estimada):
         # Si no se pudo calcular la llegada de la última, no se puede saber si hay tiempo
         return False, None, None, "Hora de llegada inválida en última reserva del móvil"

    # --- Chequeo 2: Intervalo de tiempo ---
    tipo_int_base, intervalo_base = calcular_intervalo(ultima_reserva, nueva_reserva) # Obtener intervalo base

    # Ajustar intervalo mínimo si hay tiempo de viaje estimado para la *nueva* reserva
    # (Esto podría ser una heurística para añadir un buffer si el siguiente viaje es largo)
    intervalo_min_requerido = intervalo_base
    if pd.notnull(nueva_tiempo_viaje_estimado) and nueva_tiempo_viaje_estimado > 0:
        intervalo_min_requerido = max(intervalo_base, int(nueva_tiempo_viaje_estimado * INTERVALO_MIN_DEFAULT_FACTOR))
        # Comentario: Esta lógica usa el tiempo del *nuevo* viaje para determinar el gap *después* del anterior.
        # Asegurarse que esta sea la regla de negocio deseada.

    # *** LÓGICA CLAVE CORREGIDA ***
    # La hora de recogida de la nueva reserva debe ser MAYOR O IGUAL que
    # la hora de LLEGADA ESTIMADA de la última reserva + el intervalo mínimo requerido.
    hora_minima_recogida = ultima_hora_llegada_estimada + timedelta(minutes=intervalo_min_requerido)

    if nueva_hora_recogida < hora_minima_recogida:
        motivo = f"Intervalo < {intervalo_min_requerido} min (Necesita recoger a las {hora_minima_recogida.strftime('%H:%M:%S')}, llega a las {ultima_hora_llegada_estimada.strftime('%H:%M:%S')})"
        return False, None, None, motivo

    # --- Chequeo 3: Monto máximo ---
    monto_actual = monto_total_movil(movil_reservas)
    nuevo_monto_valido = nueva_monto if pd.notnull(nueva_monto) else 0 # Usar 0 si el pago es NaN
    if monto_actual + nuevo_monto_valido > max_monto_param:
        return False, None, None, f"Excede monto máximo (${max_monto_param:,.0f})"

    # --- Chequeo 4: Horas máximas de ruta ---
    primera_hora_recogida = movil_reservas[0].get("HoraFecha")
    if pd.isna(primera_hora_recogida):
         return False, None, None, "Datos inválidos en primera reserva del móvil" # Error si la primera no tiene hora

    # La duración se mide desde la recogida de la primera hasta la recogida de la nueva
    # (Se podría considerar la llegada estimada de la nueva si estuviera disponible aquí)
    duracion_total_horas = (nueva_hora_recogida - primera_hora_recogida).total_seconds() / 3600
    if duracion_total_horas > max_horas_param:
        return False, None, None, f"Excede {max_horas_param} horas de ruta (desde 1ra recogida)"

    # --- Chequeo 5: Reglas de Categoría/División ---
    categorias_actuales = [r.get("Categoria_viaje", "Desconocida") for r in movil_reservas]
    num_interregional_actual = categorias_actuales.count("Interregional")
    # Contar divisiones "otras" (ni Interregional, ni Urbano, ni Desconocida) únicas
    otras_divisiones_unicas_actuales = set(cat for cat in categorias_actuales if cat not in ["Interregional", "Urbano", "Desconocida"])
    num_otras_divisiones_unicas_actual = len(otras_divisiones_unicas_actuales)

    es_nueva_interregional = nueva_cat == "Interregional"
    es_nueva_otra_division = nueva_cat not in ["Interregional", "Urbano", "Desconocida"]

    # Regla 5a: Límite de Interregionales
    if es_nueva_interregional and num_interregional_actual >= MAX_INTERREGIONALES_POR_MOVIL:
        return False, None, None, f"Máximo {MAX_INTERREGIONALES_POR_MOVIL} Interregionales"

    # Regla 5b: Límite de otras divisiones distintas
    # Si la nueva es otra división y ya alcanzamos el límite Y la nueva categoría no está ya en el set
    if es_nueva_otra_division and \
       num_otras_divisiones_unicas_actual >= MAX_OTRAS_DIVISIONES_POR_MOVIL and \
       nueva_cat not in otras_divisiones_unicas_actuales:
        return False, None, None, f"Máximo {MAX_OTRAS_DIVISIONES_POR_MOVIL} divisiones distintas (otras)"

    # Regla 5c: Restricciones si hay Urbanos involucrados (actuales o el nuevo)
    # Simplificado: Si hay o habrá urbanos, se aplican límites más estrictos a las otras categorías.
    # (Original parecía un poco complejo, esta es una interpretación común de esas reglas)
    # if "Urbano" in categorias_actuales or nueva_cat == "Urbano":
        # Verificar si añadir la nueva reserva violaría los límites *combinados*
        # total_interregional_proyectado = num_interregional_actual + int(es_nueva_interregional)
        # total_otras_divisiones_proyectado = num_otras_divisiones_unicas_actual + int(es_nueva_otra_division and nueva_cat not in otras_divisiones_unicas_actuales)

        # if total_interregional_proyectado > MAX_INTERREGIONALES_POR_MOVIL:
        #     return False, None, None, f"Con Urbanos, máx {MAX_INTERREGIONALES_POR_MOVIL} Interregionales"
        # if total_otras_divisiones_proyectado > MAX_OTRAS_DIVISIONES_POR_MOVIL:
        #     return False, None, None, f"Con Urbanos, máx {MAX_OTRAS_DIVISIONES_POR_MOVIL} divisiones distintas (otras)"
        # Podrían existir reglas más complejas aquí como no mezclar cierto tipo si hay urbano.
        # La regla original "no >1 Interregional y >1 división distinta (otra) simultáneamente"
        # es difícil de implementar proyectivamente sin más contexto. Se omite por simplicidad
        # pero podría re-añadirse si es crucial.

    # --- Si pasó todos los chequeos ---
    # Retorna True, el tipo de intervalo base, el intervalo mínimo que se aplicó, y None como motivo.
    return True, tipo_int_base, intervalo_min_requerido, None
# --- Fin Funciones Auxiliares ---

# --- Interfaz Streamlit ---
st.header("Cargar Archivos CSV para Asignación de Móviles")

uploaded_file_hist = st.file_uploader("1. Subir archivo Históricos (ej: distancias H3 1.7 (Hist).csv)", type="csv", key="hist_uploader")
uploaded_file_pred = st.file_uploader("2. Subir archivo Predicciones (ej: distancias H3 1.5 (pred).csv)", type="csv", key="pred_uploader")

# Solo proceder si ambos archivos están cargados
if uploaded_file_hist is not None and uploaded_file_pred is not None:

    boton_ejecutar = st.button("🚀 Ejecutar Asignación")

    if boton_ejecutar:
        # Inicializar variables dentro del botón para asegurar estado limpio en cada ejecución
        df_hist = None
        df_pred = None
        summary_df = None # DataFrame de tiempos promedio por ruta H3
        df_resultado = None # DataFrame final con asignaciones y cálculos
        moviles = [] # Lista de listas, cada sublista es la ruta de un móvil
        rutas_asignadas_list = [] # Lista de diccionarios para las rutas asignadas
        reservas_no_asignadas_list = [] # Lista de diccionarios para no asignadas

        st.write("---") # Separador visual

        # --- Fase 1: Lectura y Validación Inicial ---
        with st.expander("👁️ FASE 1: Lectura y Validación de Archivos", expanded=False): # Expandido por defecto
            with st.spinner('Leyendo y validando archivos...'):
                # Leer archivo histórico
                try:
                    df_hist = pd.read_csv(uploaded_file_hist)
                    st.write(f"✔️ Archivo histórico '{uploaded_file_hist.name}' leído ({len(df_hist)} filas).")
                    check_columns(df_hist, REQUIRED_HIST_COLS, uploaded_file_hist.name)
                    st.write(f"✔️ Columnas requeridas encontradas en archivo histórico.")
                except pd.errors.EmptyDataError:
                    st.error(f"Error Crítico: El archivo histórico '{uploaded_file_hist.name}' está vacío.")
                    st.stop()
                except Exception as e:
                    st.error(f"Error Crítico al leer o validar el archivo histórico '{uploaded_file_hist.name}': {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()

                # Leer archivo de predicciones
                try:
                    df_pred = pd.read_csv(uploaded_file_pred)
                    st.write(f"✔️ Archivo de predicciones '{uploaded_file_pred.name}' leído ({len(df_pred)} filas).")
                    # Validar columnas *originales* antes de renombrar
                    check_columns(df_pred, REQUIRED_PRED_COLS_ORIGINAL, uploaded_file_pred.name)
                    st.write(f"✔️ Columnas originales requeridas encontradas en archivo de predicciones.")
                except pd.errors.EmptyDataError:
                    st.error(f"Error Crítico: El archivo de predicciones '{uploaded_file_pred.name}' está vacío.")
                    st.stop()
                except Exception as e:
                    st.error(f"Error Crítico al leer o validar el archivo de predicciones '{uploaded_file_pred.name}': {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()

                # Renombrar columnas de predicciones
                try:
                    df_pred.rename(columns=RENAME_MAP_PRED, inplace=True)
                    # Validar columnas *después* de renombrar
                    check_columns(df_pred, REQUIRED_PRED_COLS_RENAMED, f"{uploaded_file_pred.name} (después de renombrar)")
                    st.write(f"✔️ Columnas renombradas y verificadas en predicciones.")
                except KeyError as e:
                    st.error(f"Error Crítico al renombrar columna: La columna original '{e}' definida en RENAME_MAP_PRED no se encontró en '{uploaded_file_pred.name}'. Ajusta RENAME_MAP_PRED en el script.")
                    st.stop()
                except Exception as e:
                     st.error(f"Error inesperado durante el renombrado de columnas: {e}")
                     st.error(f"Traceback: {traceback.format_exc()}")
                     st.stop()

                # Convertir tipos de datos importantes y manejar errores
                try:
                    # Convertir HoraFecha a datetime
                    df_pred["HoraFecha"] = pd.to_datetime(df_pred["HoraFecha"], errors='coerce')
                    num_invalid_dates = df_pred["HoraFecha"].isnull().sum()
                    if num_invalid_dates > 0:
                        st.warning(f"⚠️ {num_invalid_dates} fechas en 'HoraFecha' (predicciones) no pudieron ser convertidas y serán ignoradas.")
                        df_pred.dropna(subset=["HoraFecha"], inplace=True) # Eliminar filas con fechas inválidas
                        st.write(f"Filas restantes después de eliminar fechas inválidas: {len(df_pred)}")


                    # Convertir tiempo histórico a numérico
                    df_hist['tiempoestimada'] = pd.to_numeric(df_hist['tiempoestimada'], errors='coerce')
                    num_invalid_times = df_hist['tiempoestimada'].isnull().sum()
                    if num_invalid_times > 0:
                        st.warning(f"⚠️ {num_invalid_times} valores en 'tiempoestimada' (histórico) no son numéricos y serán ignorados en los promedios.")
                        # No eliminamos filas aquí, el dropna en groupby se encargará

                    # Convertir pago estimado a numérico
                    df_pred['estimated_payment'] = pd.to_numeric(df_pred['estimated_payment'], errors='coerce')
                    num_invalid_payments = df_pred['estimated_payment'].isnull().sum()
                    if num_invalid_payments > 0:
                         st.warning(f"⚠️ {num_invalid_payments} valores en 'estimated_payment' (predicciones) no son numéricos. Se tratarán como 0.")
                         df_pred['estimated_payment'].fillna(0, inplace=True) # Llenar NaN con 0

                    st.write(f"✔️ Tipos de datos convertidos (Fecha, Tiempo Histórico, Pago).")

                    if df_pred.empty:
                        st.error("Error Crítico: No quedaron predicciones válidas después de la limpieza inicial.")
                        st.stop()

                except Exception as e:
                    st.error(f"Error Crítico durante la conversión de tipos de datos: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()
            st.success("Fase 1 completada.")

        # --- Fase 2: Procesamiento de Datos Históricos ---
        with st.expander("⚙️ FASE 2: Procesamiento Histórico", expanded=False):
            with st.spinner('Calculando rutas y promedios históricos...'):
                try:
                    # Calcular H3 simulado y distancia para datos históricos
                    df_hist['h3_origin'] = simulate_h3_vectorized(df_hist['latrecogida'], df_hist['lonrecogida'])
                    df_hist['h3_destino'] = simulate_h3_vectorized(df_hist['latdestino'], df_hist['londestino'])
                    # df_hist['distance_km'] = haversine_vectorized(df_hist['latrecogida'], df_hist['lonrecogida'], df_hist['latdestino'], df_hist['londestino']) # El cálculo de distancia no parece usarse después
                    st.write(f"✔️ H3 simulado calculado para datos históricos.")

                    # Calcular tiempo promedio por ruta H3 origen-destino
                    # Importante: dropna quita filas donde 'tiempoestimada' es NaN ANTES de agrupar
                    summary_df = df_hist.dropna(subset=['tiempoestimada', 'h3_origin', 'h3_destino']) \
                                        .groupby(['h3_origin', 'h3_destino'], as_index=False)['tiempoestimada'] \
                                        .mean() \
                                        .rename(columns={'tiempoestimada': 'avg_travel_time'})

                    st.write(f"✔️ Tiempo promedio por ruta H3 calculado ({len(summary_df)} rutas únicas encontradas en históricos).")
                    if summary_df.empty:
                         st.warning("⚠️ No se pudieron calcular rutas promedio desde los datos históricos (puede que no haya datos válidos). La asignación usará intervalos base sin ajuste por tiempo de viaje.")
                    # Mostrar una muestra de los tiempos calculados (opcional)
                    # st.write("Muestra de tiempos promedio por ruta H3:")
                    # st.dataframe(summary_df.head())

                except Exception as e:
                    st.error(f"Error Crítico durante el procesamiento de datos históricos: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()
            st.success("Fase 2 completada.")

# --- Fase 3: Procesamiento de Predicciones y Enriquecimiento ---
        with st.expander("📈 FASE 3: Enriquecimiento de Predicciones", expanded=False):
            with st.spinner('Calculando rutas H3, tiempos promedio y horas de llegada...'): # Texto del spinner actualizado
                try:
                    # Calcular H3 simulado para predicciones
                    df_pred['h3_origin'] = simulate_h3_vectorized(df_pred['latrecogida'], df_pred['lonrecogida'])
                    df_pred['h3_destino'] = simulate_h3_vectorized(df_pred['latdestino'], df_pred['londestino'])
                    st.write(f"✔️ H3 simulado calculado para predicciones.")

                    # *** LÓGICA MEJORADA: Usar pd.merge para eficiencia ***
                    # Cruzar (merge) las predicciones con los tiempos promedio históricos
                    if summary_df is not None and not summary_df.empty:
                        df_resultado = pd.merge(df_pred, summary_df, on=['h3_origin', 'h3_destino'], how='left')
                        num_matched = df_resultado['avg_travel_time'].notna().sum()
                        st.write(f"✔️ Tiempos promedio históricos asociados a predicciones ({num_matched} de {len(df_resultado)} coincidencias encontradas).")
                    else:
                        st.warning("⚠️ No hay datos de tiempos promedio históricos para asociar. 'avg_travel_time' será NaN.")
                        df_resultado = df_pred.copy() # Usar df_pred directamente
                        df_resultado['avg_travel_time'] = np.nan # Asegurar que la columna exista pero con NaN

                    # *** NUEVA LÓGICA PARA HORA DE LLEGADA ESTIMADA ***
                    # Definir el tiempo de viaje por defecto en minutos si no hay histórico
                    DEFAULT_TRAVEL_TIME_MIN = 70
                    default_timedelta = timedelta(minutes=DEFAULT_TRAVEL_TIME_MIN)
                    st.write(f"ℹ️ Se usará un tiempo de viaje por defecto de {DEFAULT_TRAVEL_TIME_MIN} minutos si no se encuentra histórico.")

                    # 1. Calcular hora de llegada usando el tiempo histórico (resulta en NaT si avg_travel_time es NaN)
                    time_delta_hist = pd.to_timedelta(df_resultado['avg_travel_time'], unit='m', errors='coerce')
                    df_resultado['estimated_arrival'] = df_resultado['HoraFecha'] + time_delta_hist

                    # 2. Identificar dónde NO se pudo calcular la llegada (es NaT porque avg_travel_time era NaN)
                    mask_na_arrival = df_resultado['estimated_arrival'].isna()
                    num_default_applied = mask_na_arrival.sum()
                    num_hist_applied = len(df_resultado) - num_default_applied

                    # 3. Aplicar el tiempo de viaje por defecto (70 min) a esas filas
                    if num_default_applied > 0:
                        # Usar .loc para asignar el valor calculado a las filas donde la máscara es True
                        df_resultado.loc[mask_na_arrival, 'estimated_arrival'] = df_resultado.loc[mask_na_arrival, 'HoraFecha'] + default_timedelta
                        st.write(f"✔️ Hora de llegada calculada: {num_hist_applied} con tiempo histórico, {num_default_applied} con tiempo por defecto ({DEFAULT_TRAVEL_TIME_MIN} min).")
                    else:
                        st.write(f"✔️ Hora de llegada calculada para todas las {num_hist_applied} reservas usando tiempo histórico.")
                    
                    # Añadir una columna para indicar qué tipo de tiempo se usó (opcional, para análisis)
                    df_resultado['tiempo_usado'] = np.where(mask_na_arrival, 'Default (70min)', 'Historico')


                    # Ordenar las reservas para procesar en orden cronológico y por pago (descendente)
                    df_resultado_sorted = df_resultado.sort_values(
                        by=["HoraFecha", "estimated_payment"],
                        ascending=[True, False], # HoraFecha ascendente, Pago descendente
                        na_position='last' # Aunque HoraFecha ya no debería tener NaNs aquí
                    ).reset_index(drop=True)

                    # Filtrar por si acaso (aunque ya no debería haber NaNs en HoraFecha o estimated_arrival)
                    # Es importante asegurarse que estimated_arrival no tenga NaNs antes de la Fase 4
                    initial_rows = len(df_resultado_sorted)
                    df_resultado_sorted.dropna(subset=['HoraFecha', 'estimated_arrival'], inplace=True)
                    if len(df_resultado_sorted) < initial_rows:
                         st.warning(f"⚠️ Se eliminaron {initial_rows - len(df_resultado_sorted)} filas debido a valores nulos inesperados en HoraFecha o estimated_arrival después del cálculo.")

                    num_valid_to_assign = len(df_resultado_sorted)
                    st.write(f"✔️ Predicciones enriquecidas y ordenadas, listas para asignar: {num_valid_to_assign} reservas.")

                    if num_valid_to_assign == 0:
                        st.error("Error Crítico: No hay reservas válidas para intentar asignar después del enriquecimiento.")
                        st.stop()

                except Exception as e:
                    st.error(f"Error Crítico durante el procesamiento y enriquecimiento de predicciones: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()
            st.success("Fase 3 completada.")

# --- Fase 4: Algoritmo de Asignación ---
# (Esta fase no requiere modificaciones, ya que ahora recibe 'estimated_arrival' siempre calculado)
        with st.expander("🚚 FASE 4: Asignación de Reservas", expanded=False):
            with st.spinner('Asignando reservas a móviles...'):
                try:
                    # Convertir el DataFrame ordenado a lista de diccionarios para iterar
                    reservas_a_procesar = df_resultado_sorted.to_dict('records')
                    num_total_reservas = len(reservas_a_procesar)
                    st.write(f"Iniciando asignación para {num_total_reservas} reservas válidas...")

                    # Barra de progreso y texto de estado
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # Iterar sobre cada reserva ordenada
                    for i, reserva_actual in enumerate(reservas_a_procesar):
                        # Actualizar progreso
                        progress_percentage = (i + 1) / num_total_reservas
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Procesando reserva {i+1}/{num_total_reservas} (ID: {reserva_actual.get('reserva', 'N/A')})...")

                        asignado = False
                        mejor_motivo_no_asignado = "No se encontró móvil compatible o se alcanzó límite de móviles" # Default

                        # Intentar asignar a un móvil existente
                        for idx, movil_actual in enumerate(moviles):
                            # La función `puede_agregarse_a_movil` contiene toda la lógica de validación
                            # Esta función usará la 'estimated_arrival' de la última reserva del movil_actual,
                            # la cual ahora siempre tendrá un valor (histórico o default).
                            puede_agregar, tipo_rel, int_aplicado, motivo_rechazo = puede_agregarse_a_movil(movil_actual, reserva_actual)

                            if puede_agregar:
                                # ¡Asignar! Añadir la reserva al final de la ruta del móvil
                                movil_actual.append(reserva_actual)
                                # Guardar registro de la asignación
                                rutas_asignadas_list.append({
                                    "movil_id": idx + 1, # ID legible del móvil (1-based)
                                    **reserva_actual, # Añadir todos los datos de la reserva
                                    "tipo_relacion": tipo_rel, # Cómo se relaciona con la anterior
                                    "min_intervalo_aplicado": int_aplicado # Qué intervalo se usó
                                })
                                asignado = True
                                break # Pasar a la siguiente reserva una vez asignada
                            else:
                                # Guardar el motivo del rechazo (puede ser útil para el análisis posterior)
                                # Si falla con varios móviles, el último motivo será el guardado si no se asigna.
                                mejor_motivo_no_asignado = motivo_rechazo


                        # Si no se pudo asignar a un móvil existente, intentar crear uno nuevo
                        if not asignado and len(moviles) < max_moviles_param:
                            # Crear un nuevo móvil con esta reserva como la primera
                            moviles.append([reserva_actual])
                            # Guardar registro de la asignación (inicio de ruta)
                            rutas_asignadas_list.append({
                                "movil_id": len(moviles), # El ID del nuevo móvil
                                **reserva_actual,
                                "tipo_relacion": "Inicio Ruta",
                                "min_intervalo_aplicado": 0 # No aplica intervalo para la primera
                            })
                            asignado = True

                        # Si después de intentar todo, no se asignó
                        if not asignado:
                            # Añadir a la lista de no asignadas con el último motivo conocido
                            reserva_actual["motivo_no_asignado"] = mejor_motivo_no_asignado
                            reservas_no_asignadas_list.append(reserva_actual)

                    # Limpiar barra de progreso y texto final
                    status_text.text(f"Asignación completada. {len(rutas_asignadas_list)} asignadas, {len(reservas_no_asignadas_list)} no asignadas.")
                    progress_bar.empty() # Ocultar la barra de progreso

                    st.write(f"✔️ Asignación finalizada.")
                except Exception as e:
                    st.error(f"Error Crítico durante la asignación de reservas: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()
                st.success("Fase 4 completada.")


        # --- Fase 5: Resultados (Fuera de los expanders) ---
        st.subheader("🏁 Fase 5: Resultados Finales")
        try:
            st.success("✅ Proceso de asignación finalizado.")

            # Crear DataFrames a partir de las listas de resultados
            df_rutas = pd.DataFrame(rutas_asignadas_list) if rutas_asignadas_list else pd.DataFrame()
            df_no_asignadas = pd.DataFrame(reservas_no_asignadas_list) if reservas_no_asignadas_list else pd.DataFrame()

            st.subheader("📊 Resumen de la Asignación")
            num_asignadas = len(df_rutas)
            num_no_asignadas = len(df_no_asignadas)
            total_reservas_intentadas = num_asignadas + num_no_asignadas # Total que entraron a la fase 4
            num_moviles_usados = len(moviles) # Número de listas en 'moviles'
            monto_total_asignado = df_rutas['estimated_payment'].sum() if not df_rutas.empty else 0

            # Calcular porcentajes con seguridad (evitar división por cero)
            perc_asignadas = (num_asignadas / total_reservas_intentadas * 100) if total_reservas_intentadas > 0 else 0
            perc_no_asignadas = (num_no_asignadas / total_reservas_intentadas * 100) if total_reservas_intentadas > 0 else 0

            # Mostrar métricas clave en columnas
            col1, col2, col3 = st.columns(3)
            col1.metric("Reservas Procesadas (Fase 4)", f"{total_reservas_intentadas}")
            col2.metric("Reservas Asignadas", f"{num_asignadas} ({perc_asignadas:.1f}%)")
            col3.metric("Reservas No Asignadas", f"{num_no_asignadas} ({perc_no_asignadas:.1f}%)")

            col1b, col2b, col3b = st.columns(3)
            col1b.metric("Móviles Utilizados", f"{num_moviles_usados} / {max_moviles_param}")
            col2b.metric("Monto Total Asignado", f"${monto_total_asignado:,.0f}")
            # col3b podría usarse para otra métrica, e.g., promedio de reservas por móvil

            st.subheader("📋 Reservas Asignadas por Móvil")
            # Columnas a mostrar (asegurarse que existan en df_rutas antes de seleccionar)
            cols_deseadas_rutas = [
                'movil_id', 'reserva', 'HoraFecha', 'estimated_arrival', 'estimated_payment',
                'Categoria_viaje', 'tipo_relacion', 'min_intervalo_aplicado',
                 'avg_travel_time', 'Convenio', 'Tipo_servicio'
                # Añadir 'Convenio', 'Tipo_servicio' si existen y son relevantes
                # 'Convenio', 'Tipo_servicio'
            ]
            if not df_rutas.empty:
                # Filtrar columnas deseadas que realmente existen en el DataFrame
                cols_mostrar_rutas = [col for col in cols_deseadas_rutas if col in df_rutas.columns]
                # Formatear columnas de fecha/hora para mejor legibilidad
                if 'HoraFecha' in cols_mostrar_rutas:
                    df_rutas['HoraFecha'] = df_rutas['HoraFecha'].dt.strftime('%Y-%m-%d %H:%M:%S')
                if 'estimated_arrival' in cols_mostrar_rutas:
                    df_rutas['estimated_arrival'] = df_rutas['estimated_arrival'].dt.strftime('%Y-%m-%d %H:%M:%S')

                st.dataframe(df_rutas[cols_mostrar_rutas])
                # Botón de descarga
                csv_rutas = df_rutas[cols_mostrar_rutas].to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Descargar rutas_asignadas.csv",
                    data=csv_rutas,
                    file_name="rutas_asignadas.csv",
                    mime="text/csv"
                )
            else:
                st.info("No se asignaron rutas.")


            st.subheader("🚨 Reservas No Asignadas")
            # Columnas a mostrar para no asignadas
            cols_deseadas_no_asignadas = [
                'reserva', 'HoraFecha', 'estimated_payment', 'Categoria_viaje',
                 'avg_travel_time', 'motivo_no_asignado', 'Convenio', 'Tipo_servicio'
                # Añadir 'Convenio', 'Tipo_servicio' si existen y son relevantes
            ]
            if not df_no_asignadas.empty:
                 # Filtrar columnas deseadas que realmente existen
                cols_mostrar_no_asignadas = [col for col in cols_deseadas_no_asignadas if col in df_no_asignadas.columns]
                 # Formatear fecha/hora
                if 'HoraFecha' in cols_mostrar_no_asignadas:
                   df_no_asignadas['HoraFecha'] = df_no_asignadas['HoraFecha'].dt.strftime('%Y-%m-%d %H:%M:%S')

                st.dataframe(df_no_asignadas[cols_mostrar_no_asignadas])
                 # Botón de descarga
                csv_no_asignadas = df_no_asignadas[cols_mostrar_no_asignadas].to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Descargar reservas_no_asignadas.csv",
                    data=csv_no_asignadas,
                    file_name="reservas_no_asignadas.csv",
                    mime="text/csv"
                )
            else:
                st.info("🎉 Todas las reservas válidas fueron asignadas o no hubo reservas para procesar.")

        except Exception as e:
            st.error(f"❌ Ocurrió un error inesperado durante la presentación de resultados:")
            st.error(e)
            st.error(f"Traceback: {traceback.format_exc()}")


    # El manejo general de errores fuera del botón podría ser útil si algo falla
    # *antes* de que se presione el botón, pero después de cargar los archivos.
    # Sin embargo, la estructura actual con `st.stop()` dentro de los `try-except`
    # maneja la mayoría de los fallos durante la ejecución principal.

else:
    st.info("Por favor, carga ambos archivos CSV (Históricos y Predicciones) para habilitar la ejecución.")
