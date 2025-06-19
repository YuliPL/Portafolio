import pandas as pd
import os

# Ruta absoluta del dataset fuente
ruta_datos = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\datos_bcrp\base_unificada.csv"

# Leer datos
df = pd.read_csv(ruta_datos, parse_dates=["fecha"])
df_limpio = df.dropna()

# Estadísticas descriptivas
estadisticas = df_limpio.describe(percentiles=[.25, .5, .75]).T
estadisticas.columns = [
    "Cuenta", "Media", "Desviación Std", "Mínimo",
    "Percentil 25%", "Mediana", "Percentil 75%", "Máximo"
]
estadisticas = estadisticas.round(3)

# Obtener la ruta del script actual
ruta_script = os.path.dirname(os.path.abspath(__file__))

# Ruta completa donde se guardará el resultado
ruta_salida = os.path.join(ruta_script, "estadisticas_descriptivas.csv")

# Guardar archivo CSV
estadisticas.to_csv(ruta_salida)

# Confirmación
print(f"\n✅ Estadísticas descriptivas guardadas correctamente en:\n{ruta_salida}")
