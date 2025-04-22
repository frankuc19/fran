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
    page_title="PrePlanIt",
    page_icon="🚀",
    layout="wide"
)

# --- Título y navegación lateral ---
st.title("Bienvenido a PreRoute de transvip")
st.sidebar.success("Ajusta los parametros necesarios para realizar el ruteo.")

st.markdown(
    """
    Esta es la página principal de la aplicación.
    Usa la barra lateral para navegar a otras secciones
    """
)

# --- Logo y Título ---
LOGO_PATH = "transvip.png"  # Ruta relativa
LOGO_WIDTH = 90
COLUMN_RATIO = [12, 1]

try:
    col_title, col_logo = st.columns(COLUMN_RATIO)

    with col_title:
        st.title("PreRoute 2.0")

    with col_logo:
        try:
            if os.path.exists(LOGO_PATH):
                logo_image = Image.open(LOGO_PATH)
                st.image(logo_image, width=LOGO_WIDTH)
            else:
                st.error(f"⚠️ Error: No se encontró el logo en '{LOGO_PATH}'. Verifica la ruta.")
        except Exception as e:
            st.error(f"⚠️ Error al cargar el logo: {e}")

except Exception as e:
    st.warning(f"No se pudo crear el layout para el logo: {e}")
    st.title("PreRoute 2.0")

# --- Constantes ---
RADIO_TIERRA_KM = 6371
PRECISION_SIMULATE_H3 = 3

INTERVALO_CAMBIO_INTERREGIONAL = 270
INTERVALO_URBANO_NOCTURNO = 70
INTERVALO_URBANO_DIURNO = 80
INTERVALO_GENERAL = 80
INTERVALO_MIN_DEFAULT_FACTOR = 1.5

MAX_INTERREGIONALES_POR_MOVIL = 2
MAX_OTRAS_DIVISIONES_POR_MOVIL = 2

REQUIRED_HIST_COLS = [
    'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'tiempoestimada'
]
REQUIRED_PRED_COLS_ORIGINAL = [
    'pickup_datetime', 'job_id', 'estimated_payment',
    'Categoria_viaje', 'latrecogida', 'lonrecogida',
    'latdestino', 'londestino'
]
RENAME_MAP_PRED = {
    'pickup_datetime': 'HoraFecha',
    'job_id': 'reserva',
}
REQUIRED_PRED_COLS_RENAMED = list(RENAME_MAP_PRED.values()) + [
    'estimated_payment', 'Categoria_viaje',
    'latrecogida', 'lonrecogida', 'latdestino', 'londestino'
]

# --- Parámetros configurables por el usuario ---
st.sidebar.header("Parámetros de Asignación")
max_moviles_param = st.sidebar.slider('Máximo de Móviles:', 1, 500, 100)
max_monto_param = st.sidebar.slider('Monto Máximo por Móvil ($):', 100000, 1000000, 500000, step=50000)
max_reservas_param = st.sidebar.slider('Máximo de Reservas por Móvil:', 1, 10, 3)
max_horas_param = st.sidebar.slider('Máximo de Horas por Móvil:', 1, 24, 10)

# --- Funciones Auxiliares ---
def check_columns(df, required_columns, filename):
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Error Crítico: Faltan columnas en '{filename}': {', '.join(missing_cols)}.")
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
    cat_ultima = ultima_reserva.get("Categoria_viaje", "Desconocida")
    hora_nueva = nueva_reserva.get("HoraFecha")

    if pd.isna(hora_nueva):
        return "Error Hora", 99999

    if cat_nueva != cat_ultima or cat_nueva in ["Interregional", "Divisiones"]:
        return "Cambio/Especial", INTERVALO_CAMBIO_INTERREGIONAL

    if cat_nueva == "Urbano":
        if 0 <= hora_nueva.hour < 6:
            return "Urbano nocturno", INTERVALO_URBANO_NOCTURNO
        else:
            return "Urbano diurno", INTERVALO_URBANO_DIURNO

    return "General", INTERVALO_GENERAL

def monto_total_movil(movil_reservas):
    monto = 0
    for r in movil_reservas:
        pago = r.get("estimated_payment", 0)
        if pd.notnull(pago):
            monto += pago
    return monto

def puede_agregarse_a_movil(movil_reservas, nueva_reserva):
    if len(movil_reservas) >= max_reservas_param:
        return False, None, None, "Máximo de reservas alcanzado"

    ultima_reserva = movil_reservas[-1]
    nueva_hora = nueva_reserva.get("HoraFecha")
    avg_travel_time = nueva_reserva.get("avg_travel_time")
    nueva_monto = nueva_reserva.get("estimated_payment", 0)
    cat_nueva = nueva_reserva.get("Categoria_viaje", "Desconocida")

    if pd.isna(nueva_hora) or pd.isna(nueva_monto):
        return False, None, None, "Datos inválidos en reserva"

    ultima_hora = ultima_reserva.get("HoraFecha")
    if pd.isna(ultima_hora):
        return False, None, None, "Hora inválida en última reserva"

    # Chequeo 2: Intervalo de tiempo
    tipo_int, intervalo_base = calcular_intervalo(ultima_reserva, nueva_reserva)
    intervalo_min_requerido = intervalo_base
    if pd.notnull(avg_travel_time) and avg_travel_time > 0:
        intervalo_min_requerido = max(intervalo_base, int(avg_travel_time * INTERVALO_MIN_DEFAULT_FACTOR))
    if nueva_hora < ultima_hora + timedelta(minutes=intervalo_min_requerido):
        return False, None, None, f"Intervalo < {intervalo_min_requerido} min"

    # Chequeo 3: Monto máximo
    if monto_total_movil(movil_reservas) + (nueva_monto if pd.notnull(nueva_monto) else 0) > max_monto_param:
        return False, None, None, f"Excede monto máximo (${max_monto_param:,.0f})"

    # Chequeo 4: Horas máximas de ruta
    primera_hora = movil_reservas[0].get("HoraFecha")
    if pd.isna(primera_hora):
         return False, None, None, "Datos inválidos en primera reserva del móvil"
    duracion_total_horas = (nueva_hora - primera_hora).total_seconds() / 3600
    if duracion_total_horas > max_horas_param:
        return False, None, None, f"Excede {max_horas_param} horas de ruta"

    # Chequeo 5: Reglas de Categoría/División
    categorias_actuales = [r.get("Categoria_viaje", "Desconocida") for r in movil_reservas]
    num_interregional_actual = categorias_actuales.count("Interregional")
    num_urbano_actual = categorias_actuales.count("Urbano")
    otras_divisiones_unicas_actuales = set(d for d in categorias_actuales if d not in ["Interregional", "Urbano", "Desconocida"])
    es_nueva_interregional = cat_nueva == "Interregional"
    es_nueva_urbano = cat_nueva == "Urbano"
    es_nueva_otra_division = cat_nueva not in ["Interregional", "Urbano", "Desconocida"]

    if es_nueva_interregional and num_interregional_actual >= MAX_INTERREGIONALES_POR_MOVIL:
        return False, None, None, f"Máximo {MAX_INTERREGIONALES_POR_MOVIL} Interregionales"
    if es_nueva_otra_division and len(otras_divisiones_unicas_actuales) >= MAX_OTRAS_DIVISIONES_POR_MOVIL and cat_nueva not in otras_divisiones_unicas_actuales:
        return False, None, None, f"Máximo {MAX_OTRAS_DIVISIONES_POR_MOVIL} divisiones distintas (otras)"
    if num_urbano_actual >= 1 or es_nueva_urbano:
        total_interregional = num_interregional_actual + int(es_nueva_interregional)
        total_otras_divisiones = len(otras_divisiones_unicas_actuales) + int(es_nueva_otra_division and cat_nueva not in otras_divisiones_unicas_actuales)
        if total_interregional > MAX_INTERREGIONALES_POR_MOVIL:
            return False, None, None, f"Con Urbanos, máx {MAX_INTERREGIONALES_POR_MOVIL} Interregionales"
        if total_otras_divisiones > MAX_OTRAS_DIVISIONES_POR_MOVIL:
            return False, None, None, f"Con Urbanos, máx {MAX_OTRAS_DIVISIONES_POR_MOVIL} divisiones distintas (otras)"
        if total_interregional > 1 and total_otras_divisiones > 1:
             return False, None, None, "Con Urbanos: no >1 Interregional y >1 división distinta (otra) simultáneamente"

    return True, tipo_int, intervalo_min_requerido, None
# --- Fin Funciones Auxiliares ---

# --- Interfaz Streamlit ---
st.header("Cargar Archivos CSV para Asignación de Móviles")

uploaded_file_hist = st.file_uploader("1. Subir archivo Históricos (ej: distancias H3 1.7 (Hist).csv)", type="csv", key="hist_uploader")
uploaded_file_pred = st.file_uploader("2. Subir archivo Predicciones (ej: distancias H3 1.5 (pred).csv)", type="csv", key="pred_uploader")

if uploaded_file_hist is not None and uploaded_file_pred is not None:

    boton_ejecutar = st.button("🚀 Ejecutar Asignación")

    if boton_ejecutar:
        df_hist = None
        df_pred = None
        summary_df = None
        df_sorted = None
        st.write("---") # Separador visual

        # --- Fase 1: Lectura y Validación Inicial ---
        # Usamos st.expander para hacer esta sección desplegable
        with st.expander("👁️ FASE 1: Lectura y Validación de Archivos", expanded=False):
            with st.spinner('Leyendo y validando archivos...'):
                # Leer archivo histórico
                try:
                    df_hist = pd.read_csv(uploaded_file_hist)
                    st.write(f"✔️ Archivo histórico '{uploaded_file_hist.name}' leído.")
                    check_columns(df_hist, REQUIRED_HIST_COLS, uploaded_file_hist.name)
                    st.write(f"✔️ Columnas requeridas encontradas en archivo histórico.")
                except pd.errors.EmptyDataError:
                    st.error(f"Error Crítico: El archivo histórico '{uploaded_file_hist.name}' está vacío.")
                    st.stop()
                except Exception as e:
                    st.error(f"Error Crítico al leer o validar el archivo histórico '{uploaded_file_hist.name}': {e}")
                    st.stop()

                # Leer archivo de predicciones
                try:
                    df_pred = pd.read_csv(uploaded_file_pred)
                    st.write(f"✔️ Archivo de predicciones '{uploaded_file_pred.name}' leído.")
                    check_columns(df_pred, REQUIRED_PRED_COLS_ORIGINAL, uploaded_file_pred.name)
                    st.write(f"✔️ Columnas originales requeridas encontradas en archivo de predicciones.")
                except pd.errors.EmptyDataError:
                    st.error(f"Error Crítico: El archivo de predicciones '{uploaded_file_pred.name}' está vacío.")
                    st.stop()
                except Exception as e:
                    st.error(f"Error Crítico al leer o validar el archivo de predicciones '{uploaded_file_pred.name}': {e}")
                    st.stop()

                # Renombrar columnas de predicciones
                try:
                    df_pred.rename(columns=RENAME_MAP_PRED, inplace=True)
                    check_columns(df_pred, REQUIRED_PRED_COLS_RENAMED, f"{uploaded_file_pred.name} (después de renombrar)")
                    st.write(f"✔️ Columnas renombradas y verificadas en predicciones.")
                except KeyError as e:
                    st.error(f"Error Crítico al renombrar columna: La columna original '{e}' definida en RENAME_MAP_PRED no se encontró en '{uploaded_file_pred.name}'. Ajusta RENAME_MAP_PRED en el script.")
                    st.stop()
                except Exception as e:
                     st.error(f"Error inesperado durante el renombrado de columnas: {e}")
                     st.stop()

                # Convertir tipos de datos importantes
                try:
                    df_pred["HoraFecha"] = pd.to_datetime(df_pred["HoraFecha"], errors='coerce')
                    if df_pred["HoraFecha"].isnull().any():
                        st.warning(f"⚠️ Algunas fechas en 'HoraFecha' no pudieron ser convertidas.")
                    df_hist['tiempoestimada'] = pd.to_numeric(df_hist['tiempoestimada'], errors='coerce')
                    if df_hist['tiempoestimada'].isnull().any():
                        st.warning(f"⚠️ Algunos valores en 'tiempoestimada' (histórico) no son numéricos.")
                    df_pred['estimated_payment'] = pd.to_numeric(df_pred['estimated_payment'], errors='coerce')
                    if df_pred['estimated_payment'].isnull().any():
                         st.warning(f"⚠️ Algunos valores en 'estimated_payment' (predicciones) no son numéricos.")
                    st.write(f"✔️ Tipos de datos convertidos (Fecha, Tiempo Histórico, Pago).")
                except Exception as e:
                    st.error(f"Error Crítico durante la conversión de tipos de datos: {e}")
                    st.stop()
            st.success("Fase 1 completada.") # Mensaje al final del expander

        # --- Fase 2: Procesamiento de Datos Históricos ---
        with st.expander("⚙️ FASE 2: Procesamiento Histórico", expanded=False):
          with st.spinner('Calculando rutas y promedios históricos...'):
            try:
                # Asignación de H3 y cálculo de distancia con funciones vectorizadas
                df_hist['h3_origin'] = simulate_h3_vectorized(df_hist['latrecogida'], df_hist['lonrecogida'])
                df_hist['h3_destino'] = simulate_h3_vectorized(df_hist['latdestino'], df_hist['londestino'])
                df_hist['distance_km'] = haversine_vectorized(df_hist['latrecogida'], df_hist['lonrecogida'],
                                                            df_hist['latdestino'], df_hist['londestino'])
                st.write("✔️ H3 simulado y distancias calculadas para datos históricos.")

                # Filtrado de registros válidos
                valid_df = df_hist.dropna(subset=['tiempoestimada', 'distance_km'])

                # Cálculo de promedio de tiempo de viaje
                avg_times_df = valid_df.groupby(['h3_origin', 'h3_destino'], as_index=False)['tiempoestimada'].mean()
                avg_times_df.rename(columns={'tiempoestimada': 'avg_travel_time'}, inplace=True)

                # Cálculo de promedio de distancia
                avg_dist_df = valid_df.groupby(['h3_origin', 'h3_destino'], as_index=False)['distance_km'].mean()
                avg_dist_df.rename(columns={'distance_km': 'avg_distance_km'}, inplace=True)

                # Combinación en un solo DataFrame resumen
                summary_df = pd.merge(avg_times_df, avg_dist_df, on=['h3_origin', 'h3_destino'], how='outer')

                st.write(f"✔️ Promedios calculados: {len(summary_df)} rutas únicas.")
                if summary_df.empty:
                    st.warning("⚠️ No se pudieron calcular rutas promedio desde los datos históricos.")

                # Función para consultar tiempo promedio
                def get_average_time(h3_o, h3_d):
                    row = summary_df[(summary_df['h3_origin'] == h3_o) & (summary_df['h3_destino'] == h3_d)]
                    return row['avg_travel_time'].values[0] if not row.empty else np.nan

            except Exception as e:
                st.error(f"Error crítico durante el procesamiento de datos históricos: {e}")
                st.error(f"Traceback: {traceback.format_exc()}")
                st.stop()
            st.success("Fase 2 completada.")

            # --- Fase 3: Procesamiento de Predicciones ---
        with st.expander("📈 FASE 3: Procesamiento Predicciones", expanded=False):
            with st.spinner('Calculando rutas de predicciones y buscando tiempos promedio...'):
                try:
                    # Cálculo vectorizado de celdas H3
                    df_pred['h3_origin'] = simulate_h3_vectorized(df_pred['latrecogida'], df_pred['lonrecogida'])
                    df_pred['h3_destino'] = simulate_h3_vectorized(df_pred['latdestino'], df_pred['londestino'])
                    st.write("✔️ H3 simulado calculado para predicciones.")

                    # Crear copia para procesamiento
                    df = df_pred.copy()

                    # Calcular tiempo promedio individualmente
                    df['avg_travel_time'] = df.apply(
                        lambda row: get_average_time(row['h3_origin'], row['h3_destino']),
                        axis=1
                    )
                    st.write("✔️ Tiempo promedio calculado individualmente para cada predicción.")

                    # Calcular hora estimada de llegada
                    valid_time_mask = df['avg_travel_time'].notna() & df['HoraFecha'].notna()
                    df['estimated_arrival'] = pd.NaT
                    df.loc[valid_time_mask, 'estimated_arrival'] = df.loc[valid_time_mask, 'HoraFecha'] + pd.to_timedelta(df.loc[valid_time_mask, 'avg_travel_time'], unit='m')
                    st.write("✔️ Hora estimada de llegada calculada.")

                    # Asegurar valores nulos en estimated_payment
                    df['estimated_payment'].fillna(0, inplace=True)

                    # Ordenar por HoraFecha y payment antes de filtrar solapes
                    df_sorted = df.sort_values(by=["HoraFecha", "estimated_payment"], ascending=[True, False], na_position='last').reset_index(drop=True)

                    # Definir función para filtrar solapes
                    def filtrar_solapes(df_in):
                        if df_in is None or df_in.empty:
                            return pd.DataFrame()

                        df_filtrado = []
                        for movil_id, grupo in df_in.groupby('movil_id'):
                            grupo = grupo.sort_values('HoraFecha')
                            ultima_llegada = pd.Timestamp.min
                            for _, fila in grupo.iterrows():
                                if fila['HoraFecha'] >= ultima_llegada:
                                    df_filtrado.append(fila)
                                    ultima_llegada = fila['estimated_arrival']
                                else:
                                    print(f"❌ Reserva descartada por solaparse: {fila['reserva']} (inicio: {fila['HoraFecha']}, llegada anterior: {ultima_llegada})")
                        return pd.DataFrame(df_filtrado)

                    # Aplicar filtro de solapes
                    df_sin_solapes = filtrar_solapes(df_sorted)

                    # Actualizar df_sorted con datos filtrados
                    df_sorted = df_sin_solapes.reset_index(drop=True)

                    # Eliminar nulos en HoraFecha por seguridad
                    df_sorted.dropna(subset=['HoraFecha'], inplace=True)

                    st.write(f"✔️ Predicciones ordenadas y filtradas ({len(df_sorted)} válidas).")

                except Exception as e:
                    st.error(f"Error Crítico durante el procesamiento de predicciones: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()
            st.success("Fase 3 completada.")


        # --- Fase 4: Algoritmo de Asignación ---
        with st.expander("🚚 FASE 4: Asignación de Reservas", expanded=False):
            with st.spinner('Asignando reservas a móviles...'):
                try:
                    moviles = []
                    rutas_asignadas_list = []
                    reservas_no_asignadas_list = []
                    reservas_a_procesar = df_sorted.to_dict('records')
                    num_total_reservas = len(reservas_a_procesar)
                    st.write(f"Iniciando asignación para {num_total_reservas} reservas válidas...")

                    # --- Barra de Progreso ---
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    # -------------------------

                    for i, reserva in enumerate(reservas_a_procesar):
                        # --- Actualizar Progreso ---
                        progress_percentage = (i + 1) / num_total_reservas
                        progress_bar.progress(progress_percentage)
                        status_text.text(f"Procesando reserva {i+1}/{num_total_reservas}...")
                        # -------------------------

                        asignado = False
                        motivo_final_no_asignado = "No se encontró móvil compatible"

                        for idx, movil_actual in enumerate(moviles):
                            puede_agregar, tipo_relacion, intervalo, motivo = puede_agregarse_a_movil(movil_actual, reserva)
                            motivo_final_no_asignado = motivo
                            if puede_agregar:
                                movil_actual.append(reserva)
                                rutas_asignadas_list.append({
                                    "movil_id": idx + 1, **reserva,
                                    "tipo_relacion": tipo_relacion, "min_intervalo_aplicado": intervalo
                                })
                                asignado = True
                                break

                        if not asignado and len(moviles) < max_moviles_param:
                            moviles.append([reserva])
                            rutas_asignadas_list.append({
                                "movil_id": len(moviles), **reserva,
                                "tipo_relacion": "Inicio Ruta", "min_intervalo_aplicado": 0
                            })
                            asignado = True

                        if not asignado:
                            if motivo_final_no_asignado is None:
                                motivo_final_no_asignado = "Límite de móviles o restricción no evaluada"
                            reserva["motivo_no_asignado"] = motivo_final_no_asignado
                            reservas_no_asignadas_list.append(reserva)

                    # --- Limpiar Progreso ---
                    status_text.text(f"Asignación completada para {num_total_reservas} reservas.")
                    progress_bar.empty() # Opcional: ocultar la barra al final
                    # ------------------------

                    st.write(f"✔️ Asignación completada.")
                except Exception as e:
                    st.error(f"Error Crítico durante la asignación de reservas: {e}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    st.stop()
            st.success("Fase 4 completada.")


        # --- Fase 5: Resultados (Fuera de los expanders) ---
        st.subheader("🏁 Fase 5: Resultados Finales")
        try:
            st.success("✅ Proceso de asignación finalizado.")

            df_rutas = pd.DataFrame(rutas_asignadas_list) if rutas_asignadas_list else pd.DataFrame()
            df_no_asignadas = pd.DataFrame(reservas_no_asignadas_list) if reservas_no_asignadas_list else pd.DataFrame()

            # Mostrar Resumen
            st.subheader("📊 Resumen de la Asignación")
            num_asignadas = len(df_rutas)
            num_no_asignadas = len(df_no_asignadas)
            total_reservas_procesadas = num_asignadas + num_no_asignadas # Re-contar por si hubo filtrados
            num_moviles_usados = len(moviles)
            monto_total_asignado = df_rutas['estimated_payment'].sum() if not df_rutas.empty else 0

            perc_asignadas = (num_asignadas / total_reservas_procesadas * 100) if total_reservas_procesadas > 0 else 0
            perc_no_asignadas = (num_no_asignadas / total_reservas_procesadas * 100) if total_reservas_procesadas > 0 else 0

            col1, col2, col3 = st.columns(3)
            col1.metric("Reservas Procesadas", f"{total_reservas_procesadas}")
            col2.metric("Reservas Asignadas", f"{num_asignadas} ({perc_asignadas:.1f}%)")
            col3.metric("Reservas No Asignadas", f"{num_no_asignadas} ({perc_no_asignadas:.1f}%)")

            col1b, col2b, col3b = st.columns(3)
            col1b.metric("Móviles Utilizados", f"{num_moviles_usados} / {max_moviles_param}")
            col2b.metric("Monto Total Asignado", f"${monto_total_asignado:,.0f}")

            # Mostrar DataFrames y botones de descarga
            st.subheader(" Reservas Asignadas por Móvil")
            cols_mostrar_rutas = ['movil_id', 'reserva', 'HoraFecha', 'estimated_arrival', 'estimated_payment', 'Categoria_viaje', 'tipo_relacion', 'min_intervalo_aplicado', 'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'h3_origin', 'h3_destino', 'avg_travel_time', 'Convenio']
            if not df_rutas.empty:
                 cols_mostrar_rutas = [col for col in cols_mostrar_rutas if col in df_rutas.columns]
                 st.dataframe(df_rutas[cols_mostrar_rutas])
                 st.download_button(label="📥 Descargar rutas_asignadas.csv", data=df_rutas[cols_mostrar_rutas].to_csv(index=False, encoding='utf-8-sig'), file_name="rutas_asignadas.csv", mime="text/csv")
            else:
                 st.info("No se asignaron rutas.")


            st.subheader("🚨 Reservas No Asignadas")
            cols_mostrar_no_asignadas = ['reserva', 'HoraFecha', 'estimated_payment', 'Categoria_viaje', 'latrecogida', 'lonrecogida', 'latdestino', 'londestino', 'h3_origin', 'h3_destino', 'avg_travel_time', 'motivo_no_asignado']
            if not df_no_asignadas.empty:
                cols_mostrar_no_asignadas = [col for col in cols_mostrar_no_asignadas if col in df_no_asignadas.columns]
                st.dataframe(df_no_asignadas[cols_mostrar_no_asignadas])
                st.download_button(label="📥 Descargar reservas_no_asignadas.csv", data=df_no_asignadas[cols_mostrar_no_asignadas].to_csv(index=False, encoding='utf-8-sig'), file_name="reservas_no_asignadas.csv", mime="text/csv")
            else:
                 st.info("🎉 Todas las reservas válidas fueron asignadas o no hubo reservas para procesar.")

        except Exception as e:
            st.error(f"❌ Ocurrió un error inesperado durante la presentación de resultados:")
            st.error(e)
            st.error(f"Traceback: {traceback.format_exc()}")


    # Manejo de error general fuera del botón por si acaso
    # except Exception as e:
    #     st.error(f"❌ Ocurrió un error inesperado y fatal durante la ejecución:")
    #     st.error(e)
    #     st.error(f"Traceback: {traceback.format_exc()}")
    #     st.warning("El proceso se ha detenido. Revisa los mensajes de error anteriores para identificar el problema.")

else:
    st.info("Por favor, carga ambos archivos CSV (Históricos y Predicciones) para habilitar la ejecución.")
