#!/usr/bin/env python3

import pandas as pd
import numpy as np
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def cargar_datos():
    """Carga datos reales del MEF o crea datos de ejemplo"""
    try:
        # Intenta cargar datos reales del MEF
        print("Buscando archivo: 2024-Gasto-Diario.csv")
        df = pd.read_csv("datos/2024-Gasto-Diario.csv")
        print(f"Datos reales cargados: {df.shape[0]:,} filas")
        return df
    except:
        # Si no encuentra el archivo, crea datos de ejemplo
        print("Archivo no encontrado. Creando datos de ejemplo...")
        print("Para datos reales: descarga de datosabiertos.gob.pe")
        
        # Crear datos sintéticos similares al MEF
        np.random.seed(42)
        n = 3000  # 3000 registros
        
        departamentos = ['Lima', 'Arequipa', 'La Libertad', 'Piura', 'Cusco', 'Junín']
        sectores = ['Educación', 'Salud', 'Transporte', 'Agricultura', 'Interior']
        
        # Generar datos base
        df = pd.DataFrame({
            'año': np.random.choice([2022, 2023, 2024], n),
            'departamento': np.random.choice(departamentos, n),
            'sector': np.random.choice(sectores, n),
            'pim': np.random.lognormal(15, 1.5, n),  # Presupuesto asignado
        })
        
        # Variables correlacionadas (como en la realidad)
        df['compromiso'] = df['pim'] * np.random.uniform(0.8, 0.95, n)
        df['devengado'] = df['compromiso'] * np.random.uniform(0.85, 0.98, n)
        df['girado'] = df['devengado'] * np.random.uniform(0.9, 0.99, n)  # Target
        
        print(f"Datos de ejemplo creados: {df.shape[0]:,} filas")
        return df

def preparar_datos(df):
    """Limpia y prepara los datos para el modelo"""
    print("\n🔧 Preparando datos...")
    
    # Limpiar datos
    cols_numericas = ['pim', 'compromiso', 'devengado', 'girado']
    df = df.dropna(subset=cols_numericas)
    df = df[(df[cols_numericas] > 0).all(axis=1)]
    
    # Crear variables útiles
    df['pct_ejecucion'] = (df['compromiso'] / df['pim']) * 100
    df['log_pim'] = np.log1p(df['pim'])
    df['log_compromiso'] = np.log1p(df['compromiso'])
    
    # Codificar variables categóricas
    df_encoded = pd.get_dummies(df, columns=['departamento', 'sector'])
    
    # Separar X (features) y y (target)
    target = 'girado'
    features = [col for col in df_encoded.columns if col != target]
    
    X = df_encoded[features]
    y = df_encoded[target]
    
    print(f"Datos preparados: {X.shape[1]} variables, {X.shape[0]:,} muestras")
    return X, y

def objetivo_optuna(trial, X, y):
    """Función que Optuna va a optimizar"""
    
    # Optuna elige el algoritmo
    algoritmo = trial.suggest_categorical('algoritmo', ['RandomForest', 'Ridge', 'Lasso'])
    
    # Configurar modelo según lo que eligió Optuna
    if algoritmo == 'RandomForest':
        modelo = RandomForestRegressor(
            n_estimators=trial.suggest_int('n_estimators', 50, 300),
            max_depth=trial.suggest_int('max_depth', 5, 20),
            random_state=42
        )
    elif algoritmo == 'Ridge':
        modelo = Ridge(alpha=trial.suggest_float('alpha', 0.1, 100, log=True))
    else:  # Lasso
        modelo = Lasso(alpha=trial.suggest_float('alpha', 0.1, 100, log=True))
    
    # Validación cruzada (prueba el modelo 5 veces)
    scores = cross_val_score(modelo, X, y, cv=5, scoring='neg_mean_squared_error')
    return scores.mean()

def optimizar_modelo(X, y):
    """Usa Optuna para encontrar el mejor modelo"""
    print("Optimizando con Optuna...")
    print("Esto puede tomar 2-3 minutos...")
    
    # Crear estudio de Optuna
    study = optuna.create_study(direction='maximize')  # Maximizar porque usamos neg_MSE
    
    # Optimizar (probar 50 combinaciones diferentes)
    study.optimize(lambda trial: objetivo_optuna(trial, X, y), n_trials=50)
    
    # Mostrar mejor resultado
    print(f"Mejor score: {study.best_value:.4f}")
    print(f"Mejor algoritmo: {study.best_params['algoritmo']}")
    
    return study

def entrenar_mejor_modelo(study, X, y):
    """Entrena el mejor modelo que encontró Optuna"""
    print("Entrenando mejor modelo...")
    
    # Dividir datos en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Construir el mejor modelo
    params = study.best_params
    if params['algoritmo'] == 'RandomForest':
        modelo = RandomForestRegressor(
            n_estimators=params['n_estimators'],
            max_depth=params['max_depth'],
            random_state=42
        )
    elif params['algoritmo'] == 'Ridge':
        modelo = Ridge(alpha=params['alpha'])
    else:  # Lasso
        modelo = Lasso(alpha=params['alpha'])
    
    # Entrenar
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones
    y_pred = modelo.predict(X_test)
    
    # Calcular métricas
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"R² (qué tan bien predice): {r2:.3f}")
    print(f"Error promedio: {mae/1_000_000:.2f} millones de soles")
    
    return modelo, {'r2': r2, 'rmse': rmse, 'mae': mae, 'y_test': y_test, 'y_pred': y_pred}

def graficar_resultados(metricas, study):
    """Hace gráficos para ver qué tal funcionó"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Predicciones vs Real
    y_test = metricas['y_test'] / 1_000_000  # Convertir a millones
    y_pred = metricas['y_pred'] / 1_000_000
    
    axes[0,0].scatter(y_test, y_pred, alpha=0.6)
    axes[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    axes[0,0].set_xlabel('Real (Millones)')
    axes[0,0].set_ylabel('Predicción (Millones)')
    axes[0,0].set_title(f'Predicciones vs Real\nR² = {metricas["r2"]:.3f}')
    
    # 2. Errores
    errores = y_test - y_pred
    axes[0,1].scatter(y_pred, errores, alpha=0.6, color='green')
    axes[0,1].axhline(0, color='red', linestyle='--')
    axes[0,1].set_xlabel('Predicciones (Millones)')
    axes[0,1].set_ylabel('Errores (Millones)')
    axes[0,1].set_title('Análisis de Errores')
    
    # 3. Distribución de errores
    axes[1,0].hist(errores, bins=30, alpha=0.7, color='orange')
    axes[1,0].set_xlabel('Errores (Millones)')
    axes[1,0].set_title('Distribución de Errores')
    
    # 4. Evolución de Optuna
    trials = [trial.value for trial in study.trials if trial.value is not None]
    axes[1,1].plot(trials)
    axes[1,1].set_xlabel('Intento')
    axes[1,1].set_ylabel('Score')
    axes[1,1].set_title('Mejora con Optuna')
    
    plt.tight_layout()
    plt.show()
    
    # Interpretación simple
    print(f"INTERPRETACIÓN:")
    if metricas['r2'] > 0.8:
        print("🟢 ¡Excelente! El modelo predice muy bien")
    elif metricas['r2'] > 0.6:
        print("🟡 Bueno: El modelo predice decentemente")
    else:
        print("🔴 Regular: El modelo necesita mejoras")

def main():
    """Ejecuta todo el proceso paso a paso"""
    
    print("="*50)
    print("🇵🇪 REGRESIÓN MEF CON OPTUNA")
    print("Predice cuánto dinero se girará realmente")
    print("="*50)
    
    # PASO 1: Cargar datos
    df = cargar_datos()
    print(f"Columnas: {list(df.columns)}")
    
    # PASO 2: Preparar datos
    X, y = preparar_datos(df)
    
    # PASO 3: Optimizar con Optuna
    study = optimizar_modelo(X, y)
    
    # PASO 4: Entrenar mejor modelo
    modelo, metricas = entrenar_mejor_modelo(study, X, y)
    
    # PASO 5: Ver resultados
    graficar_resultados(metricas, study)
    
    print(f"\n🎉 ¡TERMINADO!")
    print(f"Tu modelo puede predecir montos girados con R² = {metricas['r2']:.3f}")
    print(f"Error promedio: {metricas['mae']/1_000_000:.2f} millones de soles")
    
    return modelo, metricas

if __name__ == "__main__":
    modelo_final, resultados = main()
