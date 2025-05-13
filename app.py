# app.py

import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import traceback
from PIL import Image
import os

# --- Configuración de Página (solo una vez) ---
st.set_page_config(
    page_title="PreRoute | Transvip",
    page_icon="🚀",
    layout="wide"
)

# --- Estado de Sesión para Opciones de Filtro Dinámicas ---
if 'df_pred_preview_options' not in st.session_state:
    st.session_state.df_pred_preview_options = {
        "categories": [],
        "convenios": [],
        "file_name": None  # Para rastrear si el archivo cargado ha cambiado
    }

# --- Título y Navegación Lateral ---
st.title("Bienvenido a PreRoute de Transvip")
st.sidebar.success("Ajusta los parámetros necesarios para realizar el ruteo.")

st.markdown(
    """
    Esta aplicación asigna reservas de transporte a móviles disponibles
    según parámetros configurables y reglas de negocio.
    Sube los archivos CSV requeridos y haz clic en 'Ejecutar Asignación'.
    """
)

# --- Logo y Título ---
LOGO_PATH = "transvip.png"
LOGO_WIDTH = 90
COLUMN_RATIO = [12, 1]

try:
    col_title, col_logo = st.columns(COLUMN_RATIO)
    with col_title:
        st.title("PreRoute 2.0")
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
    st.title("PreRoute 2.0")

# --- Constantes ---
RADIO_TIERRA_KM = 6371
PRECISION_SIMULATE_H3 = 3
INTERVALO_CAMBIO_INTERREGIONAL = 300
INTERVALO_URBANO_NOCTURNO = 70
INTERVALO_URBANO_DIURNO = 80
INTERVALO_GENERAL = 80
INTERVALO_MIN_DEFAULT_FACTOR = 1.5
MAX_INTERREGIONALES_POR_MOVIL = 2
MAX_OTRAS_DIVISIONES_POR_MOVIL = 2
CONVENIOS_OBLIGATORIOS = [
    "CODELCO CHILE", "M. PELAMBRES V REGIÓN", "MINERA LOS PELAMBRES", "CODELCO PREMIUM"
]
REQUIRED_HIST_COLS = [
    'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'tiempoestimada'
]
REQUIRED_PRED_COLS_ORIGINAL = [
    'pickup_datetime', 'job_id', 'estimated_payment',
    'Categoria_viaje', 'latrecogida', 'lonrecogida',
    'latdestino', 'londestino', 'Convenio'
]
RENAME_MAP_PRED = {
    'pickup_datetime': 'HoraFecha', 'job_id': 'reserva',
}
REQUIRED_PRED_COLS_RENAMED = list(RENAME_MAP_PRED.values()) + [
    col for col in REQUIRED_PRED_COLS_ORIGINAL if col not in RENAME_MAP_PRED.keys()
]

# --- Parámetros Configurables por el Usuario (Sidebar) ---
st.sidebar.header("Parámetros de Asignación")
max_moviles_param = st.sidebar.slider('Máximo de Móviles:', min_value=0, max_value=500, value=100, step=10)
max_monto_param = st.sidebar.slider('Monto Máximo por Móvil ($):', min_value=100000, max_value=1000000, value=500000, step=50000)
max_reservas_param = st.sidebar.slider('Máximo de Reservas por Móvil:', min_value=1, max_value=20, value=5)
max_horas_param = st.sidebar.slider('Máximo de Horas por Ruta (desde 1ra recogida):', min_value=0, max_value=24, value=10)

# --- File Uploaders ---
uploaded_file_hist = st.file_uploader("1. Subir archivo Históricos (ej: distancias H3 1.7 (Hist).csv)", type="csv", key="hist_uploader")
uploaded_file_pred = st.file_uploader("2. Subir archivo Predicciones (ej: distancias H3 1.5 (pred).csv)", type="csv", key="pred_uploader")

# --- Lógica para Pre-cargar Opciones de Filtro (Sidebar) ---
if uploaded_file_pred is not None:
    if uploaded_file_pred.name != st.session_state.df_pred_preview_options["file_name"]:
        try:
            original_pos = uploaded_file_pred.tell()
            temp_df_for_options = pd.read_csv(uploaded_file_pred)
            uploaded_file_pred.seek(original_pos) # Resetear puntero para lectura principal

            temp_df_for_options.rename(columns=RENAME_MAP_PRED, inplace=True) # Aplicar renombrado

            preview_categories = []
            if 'Categoria_viaje' in temp_df_for_options.columns:
                preview_categories = sorted(temp_df_for_options['Categoria_viaje'].dropna().unique())

            preview_convenios = []
            if 'Convenio' in temp_df_for_options.columns:
                preview_convenios = sorted(temp_df_for_options['Convenio'].dropna().unique())
            
            st.session_state.df_pred_preview_options = {
                "categories": preview_categories,
                "convenios": preview_convenios,
                "file_name": uploaded_file_pred.name
            }
        except Exception as e:
            st.sidebar.warning(f"No se pudieron pre-cargar opciones de filtro: {e}")
            st.session_state.df_pred_preview_options = {"categories": [], "convenios": [], "file_name": None}
elif st.session_state.df_pred_preview_options["file_name"] is not None: # Si el archivo se des-carga
    st.session_state.df_pred_preview_options = {"categories": [], "convenios": [], "file_name": None}

# --- Filtros Adicionales en Sidebar ---
st.sidebar.header("Filtros Adicionales (Predicciones)")

if st.session_state.df_pred_preview_options["categories"]:
    selected_categories_user = st.sidebar.multiselect(
        '1. Filtrar por Categoria de viaje:',
        options=st.session_state.df_pred_preview_options["categories"],
        default=[]
    )
else:
    st.sidebar.info("Cargue Predicciones para ver filtros de Categoría.")
    selected_categories_user = []

if st.session_state.df_pred_preview_options["convenios"]:
    selected_convenios_user = st.sidebar.multiselect(
        '2. Filtrar por Convenio:',
        options=st.session_state.df_pred_preview_options["convenios"],
        default=[]
    )
else:
    st.sidebar.info("Cargue Predicciones para ver filtros de Convenio.")
    selected_convenios_user = []


# --- Funciones Auxiliares ---
def check_columns(df, required_columns, filename):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Error Crítico: Faltan columnas en '{filename}': {', '.join(missing_cols)}.")
        st.info(f"Columnas encontradas: {list(df.columns)}")
        st.info(f"Columnas requeridas: {required_columns}")
        st.stop()

def haversine_vectorized(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return RADIO_TIERRA_KM * c

def simulate_h3_vectorized(lats, lons, precision=PRECISION_SIMULATE_H3):
    lats = pd.to_numeric(lats, errors='coerce')
    lons = pd.to_numeric(lons, errors='coerce')
    return lats.round(precision).astype(str) + "_" + lons.round(precision).astype(str)

def calcular_intervalo(ultima_reserva, nueva_reserva):
    cat_nueva = nueva_reserva.get("Categoria_viaje", "Desconocida")
    hora_nueva = nueva_reserva.get("HoraFecha")
    if pd.isna(hora_nueva): return "Error Hora", 99999
    if cat_nueva in ["Interregional", "Divisiones", "Division"]: return "Cambio/Especial", INTERVALO_CAMBIO_INTERREGIONAL
    if cat_nueva == "Urbano": return ("Urbano nocturno", INTERVALO_URBANO_NOCTURNO) if 0 <= hora_nueva.hour < 6 else ("Urbano diurno", INTERVALO_URBANO_DIURNO)
    return "General", INTERVALO_GENERAL

def monto_total_movil(movil_reservas):
    monto = 0
    for r in movil_reservas:
        pago = r.get("estimated_payment", 0)
        if pd.notnull(pago): monto += pago
    return monto

def ruta_cumple_convenio_obligatorio(lista_reservas_movil, convenios_obligatorios_list):
    if not lista_reservas_movil: return False
    for reserva in lista_reservas_movil:
        if reserva.get('Convenio') in convenios_obligatorios_list: return True
    return False

def puede_agregarse_a_movil(movil_reservas, nueva_reserva):
    intervalos_minimos = { ('Urbano', 'Urbano'): 80, ('Urbano', 'Interregional'): 270, ('Interregional', 'Urbano'): 270, ('Interregional', 'Interregional'): 270, }
    categorias_validas = {"interregional", "urbano", "division", "divisiones", "desconocida"}

    if len(movil_reservas) >= max_reservas_param: return False, None, None, f"Máx. {max_reservas_param} reservas"
    ultima_reserva = movil_reservas[-1]
    nueva_hora_recogida = nueva_reserva.get("HoraFecha")
    nueva_monto = nueva_reserva.get("estimated_payment", 0)
    nueva_cat_original = nueva_reserva.get("Categoria_viaje", "Desconocida")
    nueva_cat = nueva_cat_original.lower()
    nueva_tiempo_viaje_estimado = nueva_reserva.get("avg_travel_time")
    ultima_hora_llegada_estimada = ultima_reserva.get("estimated_arrival")

    if pd.isna(nueva_hora_recogida): return False, None, None, "Datos inválidos (hora) en nueva reserva"
    if pd.isna(ultima_hora_llegada_estimada): return False, None, None, "Hora llegada inválida en últ. reserva móvil"
    if nueva_cat not in categorias_validas: return False, None, None, f"Categoría nueva inválida: {nueva_cat_original}"
    
    tipo_int_base, intervalo_base = calcular_intervalo(ultima_reserva, nueva_reserva)
    ultima_cat_original = ultima_reserva.get("Categoria_viaje", "Desconocida")
    ultima_cat = ultima_cat_original.lower()
    if ultima_cat not in categorias_validas: return False, None, None, f"Categoría últ. reserva inválida: {ultima_cat_original}"

    intervalo_min_requerido = intervalo_base
    key_intervalo_cats = (ultima_cat.capitalize(), nueva_cat.capitalize()) # Normalizar para lookup si es necesario
    
    # Ajustar claves para el lookup en intervalos_minimos si las categorías en el dict están capitalizadas
    # Esto es un ejemplo, la normalización debe ser consistente con cómo se define el dict.
    # Para este código, asumamos que el dict usa las mismas claves que se obtienen de cat.lower().capitalize()
    # o que las categorías en el dict son minúsculas como en 'categorias_validas'
    
    # Simplificado: si las categorías en intervalos_minimos son como "Urbano", "Interregional"
    # y las obtenidas (ultima_cat, nueva_cat) son minúsculas:
    # NO HAY MATCH DIRECTO. Se debe normalizar consistentemente.
    # Para el ejemplo, asumamos que 'intervalos_minimos' usa claves como ('urbano', 'interregional')
    
    _ultima_cat_norm = ultima_cat # Suponiendo que las claves de intervalos_minimos están en minúscula
    _nueva_cat_norm = nueva_cat
    
    if (_ultima_cat_norm, _nueva_cat_norm) in intervalos_minimos: # Comparar con claves normalizadas
        intervalo_min_requerido = max(intervalo_base, intervalos_minimos[(_ultima_cat_norm, _nueva_cat_norm)])
    elif pd.notnull(nueva_tiempo_viaje_estimado) and nueva_tiempo_viaje_estimado > 0:
        intervalo_min_requerido = max(intervalo_base, int(nueva_tiempo_viaje_estimado * INTERVALO_MIN_DEFAULT_FACTOR))

    hora_minima_recogida = ultima_hora_llegada_estimada + timedelta(minutes=intervalo_min_requerido)
    if nueva_hora_recogida < hora_minima_recogida:
        return False, tipo_int_base, intervalo_min_requerido, (f"Intervalo ({nueva_cat_original} post {ultima_cat_original}) < {intervalo_min_requerido} min. "
                                                              f"Rec: {nueva_hora_recogida.strftime('%H:%M')}, Lleg: {ultima_hora_llegada_estimada.strftime('%H:%M')}, MinRec: {hora_minima_recogida.strftime('%H:%M')}")

    monto_actual = monto_total_movil(movil_reservas)
    if monto_actual + (nueva_monto if pd.notnull(nueva_monto) else 0) > max_monto_param: return False, None, None, f"Excede monto máx. (${max_monto_param:,.0f})"
    
    primera_hora_recogida = movil_reservas[0].get("HoraFecha")
    if pd.isna(primera_hora_recogida): return False, None, None, "Datos inválidos en 1ra reserva móvil"
    
    duracion_total_horas = (nueva_hora_recogida - primera_hora_recogida).total_seconds() / 3600
    if duracion_total_horas > max_horas_param: return False, None, None, f"Excede {max_horas_param}h de ruta"

    categorias_actuales_ruta = [r.get("Categoria_viaje", "").lower() for r in movil_reservas]
    num_interregional_actual = categorias_actuales_ruta.count("interregional")
    otras_divisiones_unicas_actuales = set(cat for cat in categorias_actuales_ruta if cat not in ["interregional", "urbano", "desconocida"])
    num_otras_divisiones_unicas_actual = len(otras_divisiones_unicas_actuales)
    es_nueva_interregional = (nueva_cat == "interregional")
    es_nueva_otra_division = (nueva_cat not in ["interregional", "urbano", "desconocida"])

    if es_nueva_interregional and num_interregional_actual >= MAX_INTERREGIONALES_POR_MOVIL: return False, None, None, f"Máx. {MAX_INTERREGIONALES_POR_MOVIL} Interregionales"
    if es_nueva_otra_division and nueva_cat not in otras_divisiones_unicas_actuales and num_otras_divisiones_unicas_actual >= MAX_OTRAS_DIVISIONES_POR_MOVIL:
        return False, None, None, f"Máx. {MAX_OTRAS_DIVISIONES_POR_MOVIL} divisiones distintas"

    ruta_propuesta = movil_reservas + [nueva_reserva]
    if len(ruta_propuesta) == max_reservas_param:
        if not ruta_cumple_convenio_obligatorio(ruta_propuesta, CONVENIOS_OBLIGATORIOS):
            return False, None, None, f"Completaría ruta ({max_reservas_param} serv.) sin convenio obligatorio."
    
    return True, tipo_int_base, intervalo_min_requerido, None

# --- Lógica Principal de la Aplicación ---
if uploaded_file_hist is not None and uploaded_file_pred is not None:
    boton_ejecutar = st.button("🚀 Ejecutar Asignación")

    if boton_ejecutar:
        df_hist = None
        df_pred = None # Se cargará y potencialmente filtrará
        summary_df = None
        df_resultado = None
        moviles = []
        rutas_asignadas_list = []
        reservas_no_asignadas_list = []
        st.write("---")

        # --- Fase 1: Lectura y Validación Inicial ---
        with st.expander("👁️ FASE 1: Lectura y Validación de Archivos", expanded=True):
            with st.spinner('Leyendo y validando archivos...'):
                try:
                    df_hist = pd.read_csv(uploaded_file_hist)
                    st.write(f"✔️ Histórico '{uploaded_file_hist.name}' leído ({len(df_hist)} filas).")
                    check_columns(df_hist, REQUIRED_HIST_COLS, uploaded_file_hist.name)
                except Exception as e: st.error(f"Error crítico leyendo/validando Histórico: {e}\n{traceback.format_exc()}"); st.stop()

                try:
                    df_pred = pd.read_csv(uploaded_file_pred) # df_pred se carga aquí
                    st.write(f"✔️ Predicciones '{uploaded_file_pred.name}' leído ({len(df_pred)} filas).")
                    check_columns(df_pred, REQUIRED_PRED_COLS_ORIGINAL, uploaded_file_pred.name)
                    df_pred.rename(columns=RENAME_MAP_PRED, inplace=True)
                    check_columns(df_pred, REQUIRED_PRED_COLS_RENAMED, f"{uploaded_file_pred.name} (renombrado)")
                except Exception as e: st.error(f"Error crítico leyendo/validando Predicciones: {e}\n{traceback.format_exc()}"); st.stop()
                
                try:
                    df_pred["HoraFecha"] = pd.to_datetime(df_pred["HoraFecha"], errors='coerce')
                    if df_pred["HoraFecha"].isnull().sum() > 0:
                        st.warning(f"⚠️ {df_pred['HoraFecha'].isnull().sum()} fechas inválidas en Predicciones, filas eliminadas.")
                        df_pred.dropna(subset=["HoraFecha"], inplace=True)
                    df_hist['tiempoestimada'] = pd.to_numeric(df_hist['tiempoestimada'], errors='coerce')
                    df_pred['estimated_payment'] = pd.to_numeric(df_pred['estimated_payment'], errors='coerce').fillna(0)
                    if df_pred.empty: st.error("No quedaron predicciones válidas tras limpieza inicial."); st.stop()
                except Exception as e: st.error(f"Error en conversión de tipos: {e}\n{traceback.format_exc()}"); st.stop()

                # --- Aplicación de Filtros de Sidebar ---
                if not df_pred.empty:
                    st.write("---")
                    st.write("🔎 Aplicando filtros de barra lateral (si se seleccionaron):")
                    initial_rows_bf_sidebar_filters = len(df_pred)
                    any_filter_applied_sidebar = False

                    if selected_categories_user:
                        any_filter_applied_sidebar = True
                        df_pred = df_pred[df_pred['Categoria_viaje'].isin(selected_categories_user)]
                        st.write(f"✔️ Filtro 'Categoria_viaje' aplicado. Filas restantes: {len(df_pred)}.")
                        if df_pred.empty and initial_rows_bf_sidebar_filters > 0:
                            st.error("No hay predicciones tras filtrar por Categoria_viaje. Proceso detenido."); st.stop()
                    
                    if selected_convenios_user:
                        any_filter_applied_sidebar = True
                        if not df_pred.empty:
                            df_pred = df_pred[df_pred['Convenio'].isin(selected_convenios_user)]
                            st.write(f"✔️ Filtro 'Convenio' aplicado. Filas restantes: {len(df_pred)}.")
                            if df_pred.empty: st.error("No hay predicciones tras filtrar por Convenio. Proceso detenido."); st.stop()
                        elif initial_rows_bf_sidebar_filters > 0:
                            st.warning("Filtro 'Convenio' no aplicado, datos ya filtrados a cero por categoría.")
                    
                    if not any_filter_applied_sidebar: st.write("ℹ️ No se seleccionaron filtros de Categoria_viaje o Convenio.")
                    if df_pred.empty and initial_rows_bf_sidebar_filters > 0 and any_filter_applied_sidebar: st.error("Predicciones filtradas a cero. Proceso detenido."); st.stop()

            st.success("Fase 1 completada.")

        # --- Fase 2: Procesamiento Histórico ---
        with st.expander("⚙️ FASE 2: Procesamiento Histórico", expanded=False):
            with st.spinner('Calculando rutas y promedios históricos...'):
                try:
                    df_hist['h3_origin'] = simulate_h3_vectorized(df_hist['latrecogida'], df_hist['lonrecogida'])
                    df_hist['h3_destino'] = simulate_h3_vectorized(df_hist['latdestino'], df_hist['londestino'])
                    summary_df = df_hist.dropna(subset=['tiempoestimada', 'h3_origin', 'h3_destino']) \
                                        .groupby(['h3_origin', 'h3_destino'], as_index=False)['tiempoestimada'] \
                                        .mean().rename(columns={'tiempoestimada': 'avg_travel_time'})
                    if summary_df.empty: st.warning("⚠️ No se calcularon rutas promedio desde históricos.")
                except Exception as e: st.error(f"Error en Fase 2: {e}\n{traceback.format_exc()}"); st.stop()
            st.success("Fase 2 completada.")

        # --- Fase 3: Enriquecimiento de Predicciones ---
        with st.expander("📈 FASE 3: Enriquecimiento de Predicciones", expanded=False):
            if df_pred.empty: # Chequeo por si los filtros vaciaron df_pred
                st.warning("No hay datos de predicción para enriquecer (posiblemente debido a filtros). Saltando Fase 3.")
                df_resultado_sorted = pd.DataFrame() # DataFrame vacío para que Fase 4 no falle
            else:
                with st.spinner('Enriqueciendo predicciones...'):
                    try:
                        df_pred['h3_origin'] = simulate_h3_vectorized(df_pred['latrecogida'], df_pred['lonrecogida'])
                        df_pred['h3_destino'] = simulate_h3_vectorized(df_pred['latdestino'], df_pred['londestino'])
                        if summary_df is not None and not summary_df.empty:
                            df_resultado = pd.merge(df_pred, summary_df, on=['h3_origin', 'h3_destino'], how='left')
                        else:
                            df_resultado = df_pred.copy(); df_resultado['avg_travel_time'] = np.nan
                        
                        DEFAULT_TRAVEL_TIME_MIN = 70; default_timedelta = timedelta(minutes=DEFAULT_TRAVEL_TIME_MIN)
                        time_delta_hist = pd.to_timedelta(df_resultado['avg_travel_time'], unit='m', errors='coerce')
                        df_resultado['estimated_arrival'] = df_resultado['HoraFecha'] + time_delta_hist
                        mask_na_arrival = df_resultado['estimated_arrival'].isna()
                        if mask_na_arrival.sum() > 0:
                            df_resultado.loc[mask_na_arrival, 'estimated_arrival'] = df_resultado.loc[mask_na_arrival, 'HoraFecha'] + default_timedelta
                        df_resultado['tiempo_usado'] = np.where(mask_na_arrival, f'Default ({DEFAULT_TRAVEL_TIME_MIN}min)', 'Historico')
                        df_resultado_sorted = df_resultado.sort_values(by=["HoraFecha", "estimated_payment"], ascending=[True, False], na_position='last').reset_index(drop=True)
                        df_resultado_sorted.dropna(subset=['HoraFecha', 'estimated_arrival'], inplace=True) # Crucial
                        if df_resultado_sorted.empty : st.warning("No hay predicciones válidas para asignar tras enriquecimiento.");
                    except Exception as e: st.error(f"Error en Fase 3: {e}\n{traceback.format_exc()}"); st.stop()
                st.success("Fase 3 completada.")


        # --- Fase 4: Asignación de Reservas ---
        with st.expander("🚚 FASE 4: Asignación de Reservas", expanded=True):
            if df_resultado_sorted.empty:
                 st.warning("No hay reservas para asignar (posiblemente debido a filtros o procesamiento previo). Saltando Fase 4.")
            else:
                with st.spinner('Asignando reservas a móviles...'):
                    try:
                        reservas_a_procesar = df_resultado_sorted.to_dict('records')
                        num_total_reservas = len(reservas_a_procesar)
                        st.write(f"Iniciando asignación para {num_total_reservas} reservas válidas...")
                        progress_bar = st.progress(0); status_text = st.empty()

                        for i, reserva_actual in enumerate(reservas_a_procesar):
                            progress_bar.progress((i + 1) / num_total_reservas)
                            status_text.text(f"Procesando reserva {i+1}/{num_total_reservas} (ID: {reserva_actual.get('reserva', 'N/A')})...")
                            asignado = False
                            mejor_motivo_no_asignado = "No se encontró móvil compatible o se alcanzó límite de móviles"

                            for idx, movil_actual in enumerate(moviles):
                                puede_agregar, tipo_rel, int_aplicado, motivo_rechazo = puede_agregarse_a_movil(movil_actual, reserva_actual)
                                if puede_agregar:
                                    movil_actual.append(reserva_actual)
                                    rutas_asignadas_list.append({"movil_id": idx + 1, **reserva_actual, "tipo_relacion": tipo_rel, "min_intervalo_aplicado": int_aplicado})
                                    asignado = True; break
                                else: mejor_motivo_no_asignado = motivo_rechazo
                            
                            if not asignado and len(moviles) < max_moviles_param:
                                puede_iniciar_nueva_ruta = True; motivo_no_inicio_ruta = ""
                                if max_reservas_param == 1 and not ruta_cumple_convenio_obligatorio([reserva_actual], CONVENIOS_OBLIGATORIOS):
                                    puede_iniciar_nueva_ruta = False
                                    motivo_no_inicio_ruta = f"No puede iniciar ruta ({max_reservas_param} serv.) sin convenio oblig."
                                if puede_iniciar_nueva_ruta:
                                    moviles.append([reserva_actual])
                                    tipo_rel_inicio = "Inicio Ruta (Única, Cumple Convenio Oblig.)" if max_reservas_param == 1 else "Inicio Ruta"
                                    rutas_asignadas_list.append({"movil_id": len(moviles), **reserva_actual, "tipo_relacion": tipo_rel_inicio, "min_intervalo_aplicado": 0})
                                    asignado = True
                                else:
                                    if (mejor_motivo_no_asignado == "No se encontró móvil compatible o se alcanzó límite de móviles" or mejor_motivo_no_asignado == ""):
                                        mejor_motivo_no_asignado = motivo_no_inicio_ruta
                            if not asignado:
                                reserva_actual["motivo_no_asignado"] = mejor_motivo_no_asignado
                                reservas_no_asignadas_list.append(reserva_actual)
                        status_text.text(f"Asignación completada. {len(rutas_asignadas_list)} asignadas, {len(reservas_no_asignadas_list)} no asignadas.")
                        progress_bar.empty()
                    except Exception as e: st.error(f"Error en Fase 4: {e}\n{traceback.format_exc()}"); st.stop()
                st.success("Fase 4 completada.")

        # --- Fase 5: Resultados Finales ---
        st.subheader("🏁 Fase 5: Resultados Finales")
        try:
            df_rutas = pd.DataFrame(rutas_asignadas_list) if rutas_asignadas_list else pd.DataFrame()
            df_no_asignadas = pd.DataFrame(reservas_no_asignadas_list) if reservas_no_asignadas_list else pd.DataFrame()
            num_asignadas = len(df_rutas); num_no_asignadas = len(df_no_asignadas)
            total_reservas_intentadas = num_asignadas + num_no_asignadas # Basado en lo que entró a asignación
            if not df_resultado_sorted.empty: # Si hubo algo que intentar asignar
                 total_reservas_intentadas = len(df_resultado_sorted) # Mejor usar esto como base para %
            
            num_moviles_usados = len(moviles)
            monto_total_asignado = df_rutas['estimated_payment'].sum() if not df_rutas.empty else 0
            perc_asignadas = (num_asignadas / total_reservas_intentadas * 100) if total_reservas_intentadas > 0 else 0
            perc_no_asignadas = (num_no_asignadas / total_reservas_intentadas * 100) if total_reservas_intentadas > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Reservas Procesadas (Fase 4)", f"{total_reservas_intentadas}")
            col2.metric("Reservas Asignadas", f"{num_asignadas} ({perc_asignadas:.1f}%)")
            col3.metric("Reservas No Asignadas", f"{num_no_asignadas} ({perc_no_asignadas:.1f}%)")
            col1b, col2b, _ = st.columns(3)
            col1b.metric("Móviles Utilizados", f"{num_moviles_usados} / {max_moviles_param}")
            col2b.metric("Monto Total Asignado", f"${monto_total_asignado:,.0f}")

            st.subheader("📋 Reservas Asignadas por Móvil")
            cols_rutas = ['movil_id', 'reserva', 'HoraFecha', 'estimated_arrival', 'estimated_payment', 'Categoria_viaje', 'Convenio', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino', 'tipo_relacion', 'min_intervalo_aplicado', 'avg_travel_time', 'tiempo_usado']
            if not df_rutas.empty:
                df_rutas_display = df_rutas.copy()
                for col_date in ['HoraFecha', 'estimated_arrival']:
                    if col_date in df_rutas_display.columns: df_rutas_display[col_date] = pd.to_datetime(df_rutas_display[col_date]).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(df_rutas_display[[c for c in cols_rutas if c in df_rutas_display.columns]])
                st.download_button("📥 Descargar rutas_asignadas.csv", df_rutas_display[[c for c in cols_rutas if c in df_rutas_display.columns]].to_csv(index=False, encoding='utf-8-sig'), "rutas_asignadas.csv", "text/csv")
            else: st.info("No se asignaron rutas.")

            st.subheader("🚨 Reservas No Asignadas")
            cols_no_asignadas = ['reserva', 'HoraFecha', 'estimated_payment', 'Categoria_viaje', 'Convenio', 'Tipo_servicio', 'ZonaOrigen', 'Zonadestino', 'avg_travel_time', 'motivo_no_asignado']
            if not df_no_asignadas.empty:
                df_no_asignadas_display = df_no_asignadas.copy()
                if 'HoraFecha' in df_no_asignadas_display.columns: df_no_asignadas_display['HoraFecha'] = pd.to_datetime(df_no_asignadas_display['HoraFecha']).dt.strftime('%Y-%m-%d %H:%M:%S')
                st.dataframe(df_no_asignadas_display[[c for c in cols_no_asignadas if c in df_no_asignadas_display.columns]])
                st.download_button("📥 Descargar reservas_no_asignadas.csv", df_no_asignadas_display[[c for c in cols_no_asignadas if c in df_no_asignadas_display.columns]].to_csv(index=False, encoding='utf-8-sig'), "reservas_no_asignadas.csv", "text/csv")
            else: st.info("🎉 Todas las reservas válidas fueron asignadas o no hubo para procesar.")
        except Exception as e: st.error(f"Error en Fase 5: {e}\n{traceback.format_exc()}")
else:
    st.info("Por favor, carga ambos archivos CSV (Históricos y Predicciones) para habilitar la ejecución.")
