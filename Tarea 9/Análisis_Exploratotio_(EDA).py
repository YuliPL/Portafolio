import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuración visual
sns.set(style="whitegrid")
plt.rcParams.update({'figure.max_open_warning': 0})

# Ruta a la base de datos y carpeta de salida
ruta_base = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\datos_bcrp\base_unificada.csv"
carpeta_resultados = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\resultados"

# Crear carpeta si no existe
os.makedirs(carpeta_resultados, exist_ok=True)

# Leer la base de datos
df_final = pd.read_csv(ruta_base, parse_dates=["fecha"])
df_final = df_final.dropna()

# --- Gráfico 1: Serie temporal múltiple ---
fig1, ax1 = plt.subplots(figsize=(12, 6))
for col in ['IPC', 'PBI', 'RIN', 'Tasa_Interbancaria', 'Tipo_Cambio', 'BVL']:
    ax1.plot(df_final['fecha'], df_final[col], label=col)
ax1.set_title('Evolución de Indicadores Macroeconómicos (2015–2024)')
ax1.set_xlabel('Fecha')
ax1.set_ylabel('Valor')
ax1.legend()
fig1.tight_layout()
ruta_img1 = os.path.join(carpeta_resultados, "serie_tiempo_macroeconomica.png")
fig1.savefig(ruta_img1)
print(f"✅ Gráfico de serie temporal guardado en: {ruta_img1}")

# --- Gráfico 2: Mapa de calor de correlación ---
corr_matrix = df_final.drop(columns='fecha').corr()
fig2, ax2 = plt.subplots(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax2)
ax2.set_title('Matriz de Correlación de Variables Macroeconómicas')
fig2.tight_layout()
ruta_img2 = os.path.join(carpeta_resultados, "correlacion_macroeconomica.png")
fig2.savefig(ruta_img2)
print(f"✅ Mapa de correlación guardado en: {ruta_img2}")

# --- Gráfico 3: Histogramas por variable ---
fig3, axs3 = plt.subplots(2, 3, figsize=(15, 8))
axs3 = axs3.flatten()
variables = ['IPC', 'PBI', 'RIN', 'Tasa_Interbancaria', 'Tipo_Cambio', 'BVL']
for i, col in enumerate(variables):
    sns.histplot(df_final[col], kde=True, ax=axs3[i])
    axs3[i].set_title(f'Distribución: {col}')
fig3.tight_layout()
ruta_img3 = os.path.join(carpeta_resultados, "histogramas_macroeconomicos.png")
fig3.savefig(ruta_img3)
print(f"✅ Histogramas guardados en: {ruta_img3}")
