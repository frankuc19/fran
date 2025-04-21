# pages/1_Tabla_MySQL.py
import streamlit as st
import pandas as pd
import mysql.connector
from mysql.connector import Error

# Establecer configuración de la página (opcional pero bueno para título e ícono)
st.set_page_config(
    page_title="Datos Preasignaciones",
    page_icon="📊",
)

st.title("Tabla de Datos")

# Función para conectar a MySQL y obtener datos
# Mantenemos la caché para mejorar el rendimiento si la consulta no cambia frecuentemente
@st.cache_data(ttl=600) # Cachear los datos por 10 minutos (600 segundos)
def fetch_mysql_data(query):
    """
    Se conecta a la base de datos MySQL usando st.secrets
    y ejecuta la consulta proporcionada.
    Devuelve un DataFrame de Pandas.
    """
    conn = None
    try:
        conn = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            port=st.secrets["mysql"]["port"],
            database=st.secrets["mysql"]["database"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"]
        )

        if conn.is_connected():
            # st.info("Conexión a MySQL exitosa.") # Puedes comentar esto si no quieres el mensaje siempre
            cursor = conn.cursor(dictionary=True) # dictionary=True devuelve dicts
            cursor.execute(query)
            data = cursor.fetchall() # Obtener todos los registros
            df = pd.DataFrame(data) # Convertir a DataFrame de Pandas
            return df

    except Error as e:
        st.error(f"Error al conectar o consultar MySQL: {e}")
        return pd.DataFrame() # Devuelve DataFrame vacío en caso de error
    except Exception as e:
        # Captura otros posibles errores (ej: KeyError si falta algo en secrets.toml)
        st.error(f"Ocurrió un error inesperado: {e}")
        return pd.DataFrame()

    finally:
        if conn and conn.is_connected():
            # Liberar recursos
            if 'cursor' in locals() and cursor:
                 cursor.close()
            conn.close()
            # st.info("Conexión a MySQL cerrada.") # Puedes comentar esto

# --- Interfaz de Usuario ---
st.markdown("### Vista de Servicios por Convenio")

# --- TU CONSULTA SQL ---
# Aquí pegas directamente tu consulta SQL usando triple comillas para múltiples líneas
sql_query = """
SELECT
    DATE(tb_jobs.job_pickup_datetime) as Fecha,
    HOUR(tb_jobs.job_pickup_datetime) as HoraFecha,
    tb_jobs.job_id as reserva,
    transvip_contract.name as convenio
FROM
    tb_jobs
LEFT JOIN transvip_regions ON region_id = tb_jobs.branch
LEFT JOIN tb_fleets ON tb_fleets.fleet_id = tb_jobs.fleet_id
LEFT JOIN transvip_contract ON contract_id = transvip_contract.id
LEFT JOIN transvip_car_details ON transvip_car_details.id = tb_jobs.transvip_car_details_id
WHERE TRUE
    AND tb_jobs.job_pickup_datetime BETWEEN '2025-04-22 12:00:00' AND '2025-04-24 15:59:59'
    AND job_status IN (6) -- Estado específico
    AND tb_jobs.type_of_trip IN ('R','P') -- Tipos de viaje específicos
    AND transvip_contract.name IN ('ALIANZA BCI','BANCO BCI PUNTO A PUNTO','SMU S.A'); -- Convenios específicos
"""
# --- FIN DE TU CONSULTA SQL ---


# Llama a la función para obtener los datos
# st.write("Ejecutando consulta...") # Mensaje opcional mientras carga
df_data = fetch_mysql_data(sql_query)
# st.write("Consulta finalizada.") # Mensaje opcional

# Muestra los datos si el DataFrame no está vacío
if not df_data.empty:
    st.dataframe(df_data) # Muestra la tabla interactiva
    st.success(f"Se cargaron {len(df_data)} registros que coinciden con los criterios.")
elif df_data is not None: # Si fetch_mysql_data devolvió un DF vacío (tabla vacía o sin coincidencias)
    st.warning("No se encontraron registros que coincidan con los criterios especificados en la consulta.")
# Si hubo un error de conexión/consulta, fetch_mysql_data ya mostró st.error

# Puedes añadir más análisis o visualizaciones aquí si lo deseas
# st.line_chart(df_data.set_index('Fecha')['reserva']) # Ejemplo si quisieras graficar algo
