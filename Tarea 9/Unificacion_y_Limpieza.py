import pandas as pd
import os

# Carpeta del script
base_dir = os.path.dirname(__file__)

# Función para cargar y limpiar
def limpiar_dataset(nombre_archivo, nombre_variable):
    ruta = os.path.join(base_dir, nombre_archivo + ".csv")
    df = pd.read_csv(ruta, encoding='latin1')
    df = df.drop(index=0)
    df.columns = ['fecha', nombre_variable]
    df['fecha'] = pd.to_datetime(df['fecha'], format='%b%y', errors='coerce')
    df[nombre_variable] = pd.to_numeric(df[nombre_variable], errors='coerce')
    df = df.drop_duplicates(subset='fecha')
    return df

# Cargar los datasets (ya con extensión .csv)
ipc = limpiar_dataset("ipc", "IPC")
pbi = limpiar_dataset("pbi", "PBI")
rin = limpiar_dataset("rin", "RIN")
tasa = limpiar_dataset("tasa_interbancaria", "Tasa_Interbancaria")
tipo = limpiar_dataset("tipo_cambio", "Tipo_Cambio")
bvl = limpiar_dataset("bvl", "BVL")

# Unificar
df_final = ipc.merge(pbi, on='fecha') \
              .merge(rin, on='fecha') \
              .merge(tasa, on='fecha') \
              .merge(tipo, on='fecha') \
              .merge(bvl, on='fecha')

# Guardar
df_final.to_csv(os.path.join(base_dir, "base_unificada.csv"), index=False)
print("✅ Archivo base_unificada.csv generado con éxito.")
