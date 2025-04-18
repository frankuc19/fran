import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
import h3

# ⬇️ Constantes por defecto
RADIO_TIERRA_KM = 6371
PRECISION_H3 = 3

# ⬇️ Parámetros configurables a través de la interfaz de Streamlit
st.sidebar.header("Parámetros de Asignación")
MAX_MOVILES = st.sidebar.slider('Máximo de Móviles:', 1, 500, 100)
MAX_MONTO_POR_MOVIL = st.sidebar.slider('Monto Máximo por Móvil:', 100000, 1000000, 500000, step=50000)
MAX_RESERVAS_POR_MOVIL = st.sidebar.slider('Máximo de Reservas por Móvil:', 1, 10, 3)
MAX_HORAS_POR_MOVIL = st.sidebar.slider('Máximo de Horas por Móvil:', 1, 24, 10)

# ⬇️ Funciones principales (sin cambios significativos)
def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return RADIO_TIERRA_KM * c

def simulate_h3(lat, lon, precision=PRECISION_H3):
    return f"{round(lat, precision)}_{round(lon, precision)}"

def calcular_intervalo(ultima, nueva):
    if nueva["Categoria_viaje"] != ultima["Categoria_viaje"] or nueva["Categoria_viaje"] in ["Interregional", "Divisiones"]:
        return "Cambio/Interregional", 270
    if nueva["Categoria_viaje"] == "Urbano":
        return ("Urbano nocturno", 70) if 0 <= nueva["HoraFecha"].hour < 6 else ("Urbano diurno", 80)
    return "General", 80

def monto_total_movil(movil):
    return sum(r["estimated_payment"] for r in movil)

def puede_agregarse_a_movil(movil, nueva):
    if len(movil) >= MAX_RESERVAS_POR_MOVIL:
        return False, None, None, "Máximo de reservas alcanzado"
    ultima = movil[-1]
    tipo_int, intervalo_base = calcular_intervalo(ultima, nueva)
    intervalo = max(intervalo_base, int(nueva["avg_travel_time"] * 1.5)) if pd.notnull(nueva.get("avg_travel_time")) else intervalo_base
    if nueva["HoraFecha"] < ultima["HoraFecha"] + timedelta(minutes=intervalo):
        return False, None, None, f"No respeta intervalo ({intervalo} min)"
    if monto_total_movil(movil) + nueva["estimated_payment"] > MAX_MONTO_POR_MOVIL:
        return False, None, None, f"Excede monto máximo (${MAX_MONTO_POR_MOVIL:,.0f})"
    if (max(ultima["HoraFecha"], nueva["HoraFecha"]) - movil[0]["HoraFecha"]).total_seconds() / 3600 > MAX_HORAS_POR_MOVIL:
        return False, None, None, f"Excede {MAX_HORAS_POR_MOVIL} horas de ruta"
    divisiones = [r["Categoria_viaje"] for r in movil]
    num_interregional = divisiones.count("Interregional")
    num_urbano = divisiones.count("Urbano")
    divisiones_unicas_otras = set(d for d in divisiones if d not in ["Interregional", "Urbano"])
    nueva_division = nueva["Categoria_viaje"]
    nueva_es_interregional = nueva_division == "Interregional"
    nueva_es_urbano = nueva_division == "Urbano"
    nueva_es_otra = not (nueva_es_interregional or nueva_es_urbano)

    if nueva_es_interregional and num_interregional >= 2:
        return False, None, None, "Máximo 2 Interregionales"
    if nueva_es_otra and len(divisiones_unicas_otras) >= 2 and nueva_division not in divisiones_unicas_otras:
        return False, None, None, "Máximo 2 divisiones distintas"
    if num_urbano >= 1 or nueva_es_urbano:
        total_interregional = num_interregional + int(nueva_es_interregional)
        total_otras = len(divisiones_unicas_otras) + int(nueva_es_otra and nueva_division not in divisiones_unicas_otras)
        if total_interregional > 2:
            return False, None, None, "Con más de un Urbano se permiten hasta 2 Interregionales"
        if total_otras > 2:
            return False, None, None, "Con más de un Urbano se permiten hasta 2 divisiones distintas"
        if total_interregional > 1 and total_otras > 1:
            return False, None, None, "Con más de un Urbano: máximo 1 división distinta y 1 Interregional"
    return True, tipo_int, intervalo, None

# ⬇️ Interfaz para cargar archivos con Streamlit
st.header("Cargar Archivos CSV")
uploaded_file_hist = st.file_uploader("Subir archivo de Históricos (.csv)", type="csv")
uploaded_file_pred = st.file_uploader("Subir archivo de Predicciones (.csv)", type="csv")

if uploaded_file_hist is not None and uploaded_file_pred is not None:
    boton_ejecutar = st.button("Ejecutar Asignación")

    if boton_ejecutar:
        try:
            df_hist = pd.read_csv(uploaded_file_hist)
            df_pred = pd.read_csv(uploaded_file_pred)

            with st.spinner('Procesando archivos...'):
                df_hist['h3_origin'] = df_hist.apply(lambda row: simulate_h3(row['latrecogida'], row['lonrecogida']), axis=1)
                df_hist['h3_destino'] = df_hist.apply(lambda row: simulate_h3(row['latdestino'], row['londestino']), axis=1)
                df_hist['distance_km'] = df_hist.apply(lambda row: haversine(row['latrecogida'], row['lonrecogida'], row['latdestino'], row['londestino']), axis=1)
                avg_distances_df = df_hist.groupby(['h3_origin', 'h3_destino'])['distance_km'].mean().reset_index(name='avg_distance_km')
                avg_times_df = df_hist.groupby(['h3_origin', 'h3_destino'])['tiempoestimada'].mean().reset_index(name='avg_travel_time_min')
                summary_df = pd.merge(avg_distances_df, avg_times_df, on=['h3_origin', 'h3_destino'], how='outer')

                def get_average_time(o, d):
                    row = summary_df[(summary_df['h3_origin'] == o) & (summary_df['h3_destino'] == d)]
                    return row['avg_travel_time_min'].values[0] if not row.empty else np.nan

                df = df_pred.rename(columns={"pickup_datetime": "HoraFecha", "job_id": "reserva"})
                df["HoraFecha"] = pd.to_datetime(df["HoraFecha"])
                if "estimated_payment" not in df.columns:
                    st.error("Falta la columna 'estimated_payment' en el archivo de predicciones.")
                    raise ValueError("Falta la columna 'estimated_payment'")
                df['distance_km'] = df.apply(lambda row: haversine(row['latrecogida'], row['lonrecogida'], row['latdestino'], row['londestino']), axis=1)
                df['h3_origin'] = df.apply(lambda row: simulate_h3(row['latrecogida'], row['lonrecogida']), axis=1)
                df['h3_destino'] = df.apply(lambda row: simulate_h3(row['latdestino'], row['londestino']), axis=1)
                df['avg_travel_time'] = df.apply(lambda row: get_average_time(row['h3_origin'], row['h3_destino']), axis=1)
                df_sorted = df.sort_values(by=["HoraFecha", "estimated_payment"], ascending=[True, False]).reset_index(drop=True)

                moviles = []
                rutas_asignadas = []
                reservas_no_asignadas = []

                for _, reserva in df_sorted.iterrows():
                    reserva_data = reserva.to_dict()
                    asignado = False

                    if len(moviles) < MAX_MOVILES:
                        reserva_data['tiempo_estimado_min'] = get_average_time(reserva['h3_origin'], reserva['h3_destino'])
                        reserva_data['estimated_arrival'] = reserva['HoraFecha'] + timedelta(minutes=reserva_data['tiempo_estimado_min'] if pd.notnull(reserva_data['tiempo_estimado_min']) else 0)
                        moviles.append([reserva_data])
                        rutas_asignadas.append({"movil_id": len(moviles), **reserva_data, "tipo_relacion": "Inicio", "min_intervalo": 0})
                        continue

                    for movil_id, movil in enumerate(moviles, start=1):
                        puede_agregar, tipo_relacion, intervalo, motivo = puede_agregarse_a_movil(movil, reserva_data)
                        if puede_agregar:
                            reserva_data['tiempo_estimado_min'] = get_average_time(reserva['h3_origin'], reserva['h3_destino'])
                            reserva_data['estimated_arrival'] = reserva['HoraFecha'] + timedelta(minutes=reserva_data['tiempo_estimado_min'] if pd.notnull(reserva_data['tiempo_estimado_min']) else 0)
                            movil.append(reserva_data)
                            rutas_asignadas.append({"movil_id": movil_id, **reserva_data, "tipo_relacion": tipo_relacion, "min_intervalo": intervalo})
                            asignado = True
                            break

                    if not asignado:
                        reserva_data["motivo_no_asignado"] = motivo or "No cumple restricciones"
                        reservas_no_asignadas.append(reserva_data)

                df_rutas = pd.DataFrame(rutas_asignadas)
                df_no_asignadas = pd.DataFrame(reservas_no_asignadas)

                st.success("✅ Proceso finalizado.")

                st.subheader("Rutas Asignadas")
                st.dataframe(df_rutas)
                st.download_button(
                    label="Descargar rutas_asignadas.csv",
                    data=df_rutas.to_csv(index=False).encode('utf-8'),
                    file_name="rutas_asignadas.csv",
                    mime="text/csv",
                )

                st.subheader("Reservas No Asignadas")
                st.dataframe(df_no_asignadas)
                st.download_button(
                    label="Descargar reservas_no_asignadas.csv",
                    data=df_no_asignadas.to_csv(index=False).encode('utf-8'),
                    file_name="reservas_no_asignadas.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Ocurrió un error durante el procesamiento: {e}")
