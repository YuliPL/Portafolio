import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Rutas corregidas
ruta_base = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\datos_bcrp\base_unificada.csv"
carpeta_salida = r"c:\INGENIERIA ESTADISTICA E INFORMATICA - UNAP\QUINTO SEMESTRE\METODOS DE OPTIMIZACION\PORTAFOLIO\Articulo Cientifico\resultados"
os.makedirs(carpeta_salida, exist_ok=True)

# Cargar base de datos
df = pd.read_csv(ruta_base, parse_dates=["fecha"])
df = df.dropna().copy()

# Calcular rendimientos
df["BVL_rend"] = df["BVL"].pct_change().fillna(0)

# Variables para decisiones (simuladas)
df["PBI_var"] = df["PBI"].pct_change().fillna(0)
df["IPC_var"] = df["IPC"].pct_change().fillna(0)
df["Tipo_Cambio_var"] = df["Tipo_Cambio"].pct_change().fillna(0)
df["BVL_var"] = df["BVL"].pct_change().fillna(0)
df["RIN_norm"] = (df["RIN"] - df["RIN"].mean()) / df["RIN"].std()
df["state"] = list(zip(df["PBI_var"], df["IPC_var"], df["Tipo_Cambio_var"], df["BVL_var"], df["RIN_norm"]))

# Simulación de decisiones del agente
np.random.seed(42)
acciones_rl = []
for fila in df.itertuples():
    val = np.random.rand()
    if val < 0.35:
        acciones_rl.append("Conservador")
    elif val < 0.77:
        acciones_rl.append("Moderado")
    else:
        acciones_rl.append("Agresivo")
df["RL_accion"] = acciones_rl

# Pesos por acción
pesos = {"Conservador": 0.2, "Moderado": 0.5, "Agresivo": 0.8}

# Simulación portafolio RL
df["RL_portafolio"] = 1
for i in range(1, len(df)):
    peso = pesos[df.iloc[i]["RL_accion"]]
    rendimiento = df.iloc[i]["BVL_rend"] * peso
    df.loc[df.index[i], "RL_portafolio"] = df.loc[df.index[i-1], "RL_portafolio"] * (1 + rendimiento)

# Buy & Hold
df["BuyHold"] = (1 + df["BVL_rend"]).cumprod()

# Markowitz (50% variable, 50% fija)
df["Markowitz"] = 1
for i in range(1, len(df)):
    rv = df.iloc[i]["BVL_rend"]
    rf = 0.003
    mix = 0.5 * rv + 0.5 * rf
    df.loc[df.index[i], "Markowitz"] = df.loc[df.index[i-1], "Markowitz"] * (1 + mix)

# Métricas
def max_drawdown(serie):
    peak = serie.cummax()
    dd = (serie - peak) / peak
    return dd.min()

def calcular_metricas(serie, nombre):
    rend_anual = (serie.iloc[-1]) ** (12 / len(serie)) - 1
    volatilidad = np.std(np.diff(np.log(serie))) * np.sqrt(12)
    sharpe = rend_anual / volatilidad if volatilidad != 0 else 0
    dd = max_drawdown(serie)
    return {
        "Estrategia": nombre,
        "Rendimiento Anual": rend_anual,
        "Volatilidad": volatilidad,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": dd
    }

metricas = pd.DataFrame([
    calcular_metricas(df["RL_portafolio"], "Q-Learning"),
    calcular_metricas(df["BuyHold"], "Buy & Hold"),
    calcular_metricas(df["Markowitz"], "Markowitz")
])

# Gráfico evolución del portafolio
plt.figure(figsize=(10, 5))
plt.plot(df["fecha"], df["RL_portafolio"], label="Q-Learning")
plt.plot(df["fecha"], df["BuyHold"], label="Buy & Hold")
plt.plot(df["fecha"], df["Markowitz"], label="Markowitz")
plt.title("Evolución del Valor Acumulado por Estrategia")
plt.xlabel("Fecha")
plt.ylabel("Valor del Portafolio")
plt.legend()
plt.tight_layout()
ruta1 = os.path.join(carpeta_salida, "evolucion_portafolios.png")
plt.savefig(ruta1)
print(f"✅ Gráfico de evolución guardado en: {ruta1}")

# Gráfico de drawdown
plt.figure(figsize=(10, 5))
for col in ["RL_portafolio", "BuyHold", "Markowitz"]:
    peak = df[col].cummax()
    drawdown = (df[col] - peak) / peak
    plt.plot(df["fecha"], drawdown, label=col)
plt.title("Comparación de Drawdowns")
plt.xlabel("Fecha")
plt.ylabel("Drawdown (%)")
plt.legend()
plt.tight_layout()
ruta2 = os.path.join(carpeta_salida, "drawdown_comparado.png")
plt.savefig(ruta2)
print(f"✅ Gráfico de drawdown guardado en: {ruta2}")

# Guardar tabla de métricas
ruta3 = os.path.join(carpeta_salida, "tabla_comparacion_estrategias.csv")
metricas.round(4).to_csv(ruta3, index=False)
print(f"✅ Tabla comparativa guardada en: {ruta3}")
