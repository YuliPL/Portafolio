import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import folium
from pulp import *
import networkx as nx
from math import radians, cos, sin, asin, sqrt
import random
import time

class GLPOptimizationPuno:
    """
    Modelo de optimización para el problema de enrutamiento e inventario (IRP)
    para la distribución de GLP en Puno utilizando programación lineal entera mixta.
    """
    
    def __init__(self, n_clientes, n_periodos, n_vehiculos, semilla=42):
        """
        Inicializa el modelo con los parámetros básicos
        
        Args:
            n_clientes: Número de clientes a atender
            n_periodos: Horizonte de planificación en días
            n_vehiculos: Número de vehículos disponibles
            semilla: Semilla para reproducibilidad
        """
        # Configurar semilla para reproducibilidad
        random.seed(semilla)
        np.random.seed(semilla)
        
        # Parámetros básicos
        self.n_clientes = n_clientes
        self.n_periodos = n_periodos
        self.n_vehiculos = n_vehiculos
        
        # Conjuntos
        self.O = [0]  # Depósito central
        self.V1 = list(range(1, n_clientes + 1))  # Clientes
        self.V = self.O + self.V1  # Todos los nodos
        self.T = list(range(1, n_periodos + 1))  # Períodos
        self.M = [0] + self.T  # Tiempo incluyendo período inicial
        self.K = list(range(1, n_vehiculos + 1))  # Vehículos
        
        # Resultados
        self.solution = None
        self.modelo = None
        self.tiempo_ejecucion = None
        self.costo_total = None
        self.rutas = None
        
        # Datos generados automáticamente
        self._generar_datos_puno()
    
    def _generar_datos_puno(self):
        """
        Genera datos simulados para Puno basados en el artículo
        """
        # Coordenadas del depósito (Parque Industrial Taparachi, Puno)
        deposito_lat = -15.8422
        deposito_lng = -70.0199
        
        # Generar coordenadas aleatorias para clientes dentro de Puno (radio de ~5km)
        self.coordenadas = {0: (deposito_lat, deposito_lng)}
        
        # Generar clientes en diferentes zonas de Puno
        for i in self.V1:
            # Variación aleatoria alrededor del centro de Puno
            lat_variation = random.uniform(-0.05, 0.05)
            lng_variation = random.uniform(-0.05, 0.05)
            
            self.coordenadas[i] = (deposito_lat + lat_variation, deposito_lng + lng_variation)
        
        # Calcular matriz de distancias (en km) usando Haversine
        self.distancias = {}
        for i in self.V:
            for j in self.V:
                if i != j:
                    self.distancias[(i, j)] = self._haversine(self.coordenadas[i][1], 
                                                             self.coordenadas[i][0],
                                                             self.coordenadas[j][1], 
                                                             self.coordenadas[j][0])
        
        # Calcular costos de transporte (S/ por km)
        # Ajustados por altitud según el artículo (~15% menos eficientes)
        costo_por_km = 1.15 * 2.5  # S/ por km ajustado por altitud
        self.costos_transporte = {(i, j): self.distancias[(i, j)] * costo_por_km 
                                for i in self.V for j in self.V if i != j}
        
        # Costos de almacenamiento (S/ por balón por día)
        self.costos_almacenamiento = {i: 0.5 for i in self.V}
        self.costos_almacenamiento[0] = 0.3  # Menor costo en depósito
        
        # Demanda (balones por día)
        # Más demanda en invierno según el artículo
        self.demanda = {}
        for i in self.V1:
            # Base de demanda diaria para cada cliente
            base_demanda = random.randint(1, 5)
            for t in self.T:
                # Variación estacional (más en invierno, período 2-3)
                factor_estacional = 1.3 if 2 <= t <= 3 else 1.0
                self.demanda[(i, t)] = max(1, int(base_demanda * factor_estacional * random.uniform(0.8, 1.2)))
        
        # Disponibilidad en depósito (suficiente para cubrir demanda)
        self.disponibilidad_deposito = {}
        for t in self.T:
            self.disponibilidad_deposito[t] = sum(self.demanda[(i, t)] for i in self.V1) * 1.2
        
        # Capacidad de almacenamiento (balones)
        self.capacidad_almacenamiento = {0: float('inf')}  # Depósito central
        for i in self.V1:
            # Capacidad basada en tamaño de cliente
            self.capacidad_almacenamiento[i] = max(10, int(sum(self.demanda[(i, t)] for t in self.T) / 
                                                 len(self.T) * random.uniform(2.5, 3.5)))
        
        # Inventario inicial (50-80% de capacidad)
        self.inventario_inicial = {0: sum(self.disponibilidad_deposito[t] for t in self.T) * 0.3}
        for i in self.V1:
            self.inventario_inicial[i] = int(self.capacidad_almacenamiento[i] * random.uniform(0.5, 0.8))
        
        # Capacidad de vehículos (balones)
        self.capacidad_vehiculos = {}
        for k in self.K:
            if k <= self.n_vehiculos // 3:  # Camionetas
                self.capacidad_vehiculos[k] = 30
            else:  # Motocicletas adaptadas
                self.capacidad_vehiculos[k] = 3
        
        # Costos fijos de operación por vehículo (S/ por día)
        self.costos_fijos = {}
        for k in self.K:
            if k <= self.n_vehiculos // 3:  # Camionetas
                self.costos_fijos[k] = 120
            else:  # Motocicletas adaptadas
                self.costos_fijos[k] = 50
    
    def _haversine(self, lon1, lat1, lon2, lat2):
        """
        Calcula la distancia entre dos puntos geográficos en kilómetros
        usando la fórmula de Haversine.
        """
        # Convertir a radianes
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
        
        # Fórmula de Haversine
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        # Radio de la Tierra en kilómetros
        r = 6371
        return c * r
    
    def construir_modelo_flujos(self):
        """
        Construye el modelo de programación lineal entera mixta con el método
        de flujos para eliminar sub-tours.
        
        Este modelo implementa la formulación descrita en el artículo.
        """
        # Crear modelo
        modelo = LpProblem("GLP_IRP_Puno", LpMinimize)
        
        # Variables de decisión
        
        # X[i,j,k,t] = 1 si el vehículo k viaja del nodo i al j en periodo t
        X = LpVariable.dicts("X", 
                            [(i, j, k, t) for i in self.V for j in self.V 
                             for k in self.K for t in self.T if i != j],
                            cat=LpBinary)
        
        # Y[i,k,t] = 1 si el nodo i es visitado por vehículo k en periodo t
        Y = LpVariable.dicts("Y", 
                            [(i, k, t) for i in self.V for k in self.K for t in self.T],
                            cat=LpBinary)
        
        # I[i,t] = nivel de inventario en nodo i al final del periodo t
        I = LpVariable.dicts("I", 
                            [(i, t) for i in self.V for t in self.M],
                            lowBound=0)
        
        # q[i,k,t] = cantidad entregada al nodo i por vehículo k en periodo t
        q = LpVariable.dicts("q", 
                            [(i, k, t) for i in self.V1 for k in self.K for t in self.T],
                            lowBound=0)
        
        # f[i,j,k,t] = variable para flujo para eliminación de sub-tours
        f = LpVariable.dicts("f", 
                           [(i, j, k, t) for i in self.V for j in self.V 
                            for k in self.K for t in self.T if i != j],
                           lowBound=0)
        
        # Establecer valores iniciales de inventario
        for i in self.V:
            I[(i, 0)] = self.inventario_inicial[i]
        
        # Función objetivo: minimizar costos de almacenamiento + transporte + fijos
        modelo += (
            lpSum([self.costos_almacenamiento[i] * I[(i, t)] for i in self.V for t in self.T]) +
            lpSum([self.costos_transporte[(i, j)] * X[(i, j, k, t)] 
                   for i in self.V for j in self.V for k in self.K for t in self.T if i != j]) +
            lpSum([self.costos_fijos[k] * Y[(0, k, t)] for k in self.K for t in self.T])
        )
        
        # Restricciones
        
        # 1. Conservación de inventario en depósito
        for t in self.T:
            modelo += I[(0, t)] == I[(0, t-1)] + self.disponibilidad_deposito[t] - \
                     lpSum(q[(i, k, t)] for i in self.V1 for k in self.K)
        
        # 2. Conservación de inventario en clientes
        for i in self.V1:
            for t in self.T:
                modelo += I[(i, t)] == I[(i, t-1)] + \
                         lpSum(q[(i, k, t)] for k in self.K) - self.demanda[(i, t)]
        
        # 3. Límites de capacidad de almacenamiento
        for i in self.V1:
            for t in self.T:
                modelo += I[(i, t)] <= self.capacidad_almacenamiento[i]
        
        # 4. Política Maximum Level (ML)
        for i in self.V1:
            for t in self.T:
                modelo += lpSum(q[(i, k, t)] for k in self.K) <= self.capacidad_almacenamiento[i] - I[(i, t-1)]
                for k in self.K:
                    modelo += q[(i, k, t)] <= self.capacidad_almacenamiento[i] * Y[(i, k, t)]
        
        # 5. Capacidad de vehículos
        for k in self.K:
            for t in self.T:
                modelo += lpSum(q[(i, k, t)] for i in self.V1) <= self.capacidad_vehiculos[k] * Y[(0, k, t)]
        
        # 6. Rutas y flujo
        for k in self.K:
            for t in self.T:
                # Cada cliente visitado debe tener una salida
                for i in self.V1:
                    modelo += lpSum(X[(i, j, k, t)] for j in self.V if j != i) == Y[(i, k, t)]
                
                # Cada cliente visitado debe tener una entrada
                for j in self.V1:
                    modelo += lpSum(X[(i, j, k, t)] for i in self.V if i != j) == Y[(j, k, t)]
                
                # Conservación de flujo en cada nodo
                for i in self.V:
                    modelo += lpSum(X[(i, j, k, t)] for j in self.V if j != i) - \
                             lpSum(X[(j, i, k, t)] for j in self.V if j != i) == 0
                
                # Salida del depósito si el vehículo es usado
                modelo += lpSum(X[(0, j, k, t)] for j in self.V1) == Y[(0, k, t)]
        
        # 7. Eliminación de sub-tours mediante flujos
        for k in self.K:
            for t in self.T:
                # Flujo saliente del depósito igual al número de clientes visitados
                modelo += lpSum(f[(0, j, k, t)] for j in self.V1) == lpSum(Y[(j, k, t)] for j in self.V1)
                
                # Conservación de flujo en cada cliente
                for i in self.V1:
                    modelo += lpSum(f[(j, i, k, t)] for j in self.V if j != i) - \
                             lpSum(f[(i, j, k, t)] for j in self.V if j != i) == Y[(i, k, t)]
                
                # Limitar flujo a arcos existentes
                for i in self.V:
                    for j in self.V:
                        if i != j:
                            modelo += f[(i, j, k, t)] <= self.n_clientes * X[(i, j, k, t)]
        
        # Guardar modelo
        self.modelo = modelo
        return modelo
    
    def resolver(self, tiempo_limite=3600, gap=0.05):
        """
        Resuelve el modelo y obtiene la solución
        
        Args:
            tiempo_limite: Tiempo máximo en segundos para la resolución
            gap: GAP de tolerancia para finalizar la optimización
        
        Returns:
            LpStatus: Estado de la solución
        """
        inicio = time.time()
        
        # Asegurarse de que el modelo esté construido
        if self.modelo is None:
            self.construir_modelo_flujos()
        
        # Configurar el solver con límite de tiempo y gap
        solver = PULP_CBC_CMD(timeLimit=tiempo_limite, gapRel=gap)
        
        # Resolver modelo
        status = self.modelo.solve(solver)
        
        # Registrar tiempo de ejecución
        self.tiempo_ejecucion = time.time() - inicio
        
        # Obtener valor de la función objetivo
        if status == LpStatusOptimal or status == LpStatusNotSolved:
            self.costo_total = value(self.modelo.objective)
            self.procesar_solucion()
        
        return LpStatus[status]
    
    def procesar_solucion(self):
        """
        Procesa los resultados del modelo y extrae información útil
        como rutas, entregas, e inventarios
        """
        if self.modelo is None or LpStatus[self.modelo.status] not in ["Optimal", "Not Solved"]:
            print("No hay solución óptima disponible")
            return
        
        self.solution = {
            "X": {},  # Rutas
            "Y": {},  # Visitas
            "q": {},  # Entregas
            "I": {}   # Inventarios
        }
        
        # Extraer valores de variables
        for v in self.modelo.variables():
            name = v.name.split("_")
            var_type = name[0]
            
            if var_type == "X" and v.value() > 0.5:
                # X_i_j_k_t
                i, j, k, t = map(int, name[1:5])
                self.solution["X"][(i, j, k, t)] = 1
            
            elif var_type == "Y" and v.value() > 0.5:
                # Y_i_k_t
                i, k, t = map(int, name[1:4])
                self.solution["Y"][(i, k, t)] = 1
            
            elif var_type == "q" and v.value() > 0:
                # q_i_k_t
                i, k, t = map(int, name[1:4])
                self.solution["q"][(i, k, t)] = v.value()
            
            elif var_type == "I":
                # I_i_t
                i, t = map(int, name[1:3])
                self.solution["I"][(i, t)] = v.value()
        
        # Extraer rutas por vehículo y período
        self.rutas = {}
        for t in self.T:
            self.rutas[t] = {}
            for k in self.K:
                # Verificar si el vehículo k es usado en período t
                if any(self.solution["Y"].get((i, k, t), 0) for i in self.V):
                    ruta = []
                    nodo_actual = 0  # Empezar en depósito
                    
                    # Construir la ruta siguiendo los arcos X
                    while True:
                        # Buscar el siguiente nodo en la ruta
                        siguiente_nodo = None
                        for j in self.V:
                            if j != nodo_actual and self.solution["X"].get((nodo_actual, j, k, t), 0) > 0.5:
                                siguiente_nodo = j
                                break
                        
                        if siguiente_nodo is None or siguiente_nodo == 0 or siguiente_nodo in ruta:
                            # Si volvimos al depósito o no hay más nodos, terminamos
                            break
                        
                        ruta.append(siguiente_nodo)
                        nodo_actual = siguiente_nodo
                    
                    if ruta:  # Solo guardar rutas no vacías
                        self.rutas[t][k] = [0] + ruta + [0]  # Añadir depósito al inicio y final

    def visualizar_rutas(self, periodo=1):
        """
        Visualiza las rutas en un mapa usando Folium para un período específico
        
        Args:
            periodo: El período de tiempo a visualizar
        
        Returns:
            Objeto mapa de Folium
        """
        if self.rutas is None or periodo not in self.rutas:
            print(f"No hay rutas disponibles para el período {periodo}")
            return None
        
        # Crear mapa centrado en Puno
        mapa = folium.Map(
            location=[-15.8422, -70.0199],
            zoom_start=13,
            tiles='OpenStreetMap'
        )
        
        # Color para el depósito
        folium.Marker(
            location=[self.coordenadas[0][0], self.coordenadas[0][1]],
            popup='Depósito',
            icon=folium.Icon(color='red', icon='star')
        ).add_to(mapa)
        
        # Colores para diferentes vehículos
        colores = ['blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 
                  'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'lightblue']
        
        # Para cada vehículo en el período
        for k, ruta in self.rutas[periodo].items():
            color_vehiculo = colores[k % len(colores)]
            
            # Crear marcadores para cada cliente en la ruta
            for i, nodo in enumerate(ruta):
                if nodo != 0:  # No repetir depósito
                    # Calcular entrega si existe
                    entrega = self.solution["q"].get((nodo, k, periodo), 0)
                    
                    # Añadir marcador
                    folium.Marker(
                        location=[self.coordenadas[nodo][0], self.coordenadas[nodo][1]],
                        popup=f'Cliente {nodo}<br>Entrega: {entrega:.1f} balones',
                        icon=folium.Icon(color=color_vehiculo)
                    ).add_to(mapa)
            
            # Dibujar la ruta con líneas
            puntos_ruta = [(self.coordenadas[nodo][0], self.coordenadas[nodo][1]) for nodo in ruta]
            
            # Estilo de línea según tipo de vehículo
            estilo_linea = {'weight': 3, 'color': color_vehiculo}
            if k <= self.n_vehiculos // 3:  # Camionetas
                estilo_linea['opacity'] = 1.0
                tipo_vehiculo = "Camioneta"
            else:  # Motocicletas
                estilo_linea['opacity'] = 0.7
                estilo_linea['dashArray'] = '5, 10'
                tipo_vehiculo = "Motocicleta"
            
            # Añadir línea al mapa
            folium.PolyLine(
                puntos_ruta,
                tooltip=f'{tipo_vehiculo} {k}',
                **estilo_linea
            ).add_to(mapa)
        
        return mapa
    
    def generar_reporte(self):
        """
        Genera un reporte completo de la solución
        
        Returns:
            dict: Reporte con análisis detallado
        """
        if self.solution is None:
            print("No hay solución disponible para generar reporte")
            return None
        
        reporte = {
            "estado": LpStatus[self.modelo.status],
            "costo_total": self.costo_total,
            "tiempo_ejecucion": self.tiempo_ejecucion,
            "desglose_costos": {},
            "estadisticas": {}
        }
        
        # Calcular desglose de costos
        costo_almacenamiento = sum(self.costos_almacenamiento[i] * self.solution["I"].get((i, t), 0) 
                                 for i in self.V for t in self.T)
        
        costo_transporte = sum(self.costos_transporte[(i, j)] 
                             for i, j, k, t in self.solution["X"])
        
        costo_fijo = sum(self.costos_fijos[k] 
                        for i, k, t in self.solution["Y"] if i == 0)
        
        reporte["desglose_costos"] = {
            "almacenamiento": costo_almacenamiento,
            "transporte": costo_transporte,
            "fijo_vehiculos": costo_fijo
        }
        
        # Estadísticas por período
        reporte["estadisticas"]["por_periodo"] = {}
        
        for t in self.T:
            clientes_visitados = sum(1 for i, k, t_val in self.solution["Y"] 
                                   if t_val == t and i != 0)
            
            vehiculos_usados = len(set(k for i, k, t_val in self.solution["Y"] 
                                     if t_val == t and i == 0))
            
            total_entregado = sum(self.solution["q"].get((i, k, t), 0) 
                                for i in self.V1 for k in self.K)
            
            reporte["estadisticas"]["por_periodo"][t] = {
                "clientes_visitados": clientes_visitados,
                "vehiculos_usados": vehiculos_usados,
                "total_entregado": total_entregado
            }
        
        # Estadísticas globales
        reporte["estadisticas"]["global"] = {
            "promedio_clientes_por_periodo": sum(datos["clientes_visitados"] 
                                              for t, datos in reporte["estadisticas"]["por_periodo"].items()) / len(self.T),
            "promedio_vehiculos_por_periodo": sum(datos["vehiculos_usados"] 
                                               for t, datos in reporte["estadisticas"]["por_periodo"].items()) / len(self.T),
            "total_entregado": sum(datos["total_entregado"] 
                                for t, datos in reporte["estadisticas"]["por_periodo"].items())
        }
        
        return reporte
    
    def graficar_inventarios(self):
        """
        Genera un gráfico de los niveles de inventario a lo largo del tiempo para
        una muestra de clientes
        
        Returns:
            Objeto matplotlib.figure.Figure
        """
        if self.solution is None:
            print("No hay solución disponible para graficar inventarios")
            return None
        
        # Seleccionar algunos clientes para la visualización (máximo 5)
        clientes_muestra = self.V1[:min(5, len(self.V1))]
        
        # Crear la figura
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Para cada cliente en la muestra
        for i in clientes_muestra:
            inventarios = [self.solution["I"].get((i, t), 0) for t in range(self.n_periodos + 1)]
            ax.plot(range(self.n_periodos + 1), inventarios, marker='o', linewidth=2, label=f'Cliente {i}')
            
            # Añadir capacidad máxima como línea punteada
            ax.axhline(y=self.capacidad_almacenamiento[i], linestyle='--', alpha=0.3, 
                     color=ax.lines[-1].get_color())
        
        # Añadir inventario del depósito
        inventarios_deposito = [self.solution["I"].get((0, t), 0) for t in range(self.n_periodos + 1)]
        ax.plot(range(self.n_periodos + 1), inventarios_deposito, marker='s', linewidth=2, 
               color='black', label='Depósito')
        
        # Configuraciones del gráfico
        ax.set_xlabel('Período')
        ax.set_ylabel('Nivel de Inventario (balones)')
        ax.set_title('Evolución de Inventarios')
        ax.set_xticks(range(self.n_periodos + 1))
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def exportar_resultados(self, directorio="."):
        """
        Exporta los resultados del modelo a archivos CSV
        
        Args:
            directorio: Ruta donde guardar los archivos
        """
        if self.solution is None:
            print("No hay solución disponible para exportar")
            return
        
        # Exportar rutas
        rutas_data = []
        for t in self.T:
            if t in self.rutas:
                for k, ruta in self.rutas[t].items():
                    for idx in range(len(ruta)-1):
                        i, j = ruta[idx], ruta[idx+1]
                        rutas_data.append({
                            'periodo': t,
                            'vehiculo': k,
                            'origen': i,
                            'destino': j,
                            'tipo_vehiculo': 'Camioneta' if k <= self.n_vehiculos // 3 else 'Motocicleta',
                            'distancia': self.distancias.get((i, j), 0)
                        })
        
        # Exportar entregas
        entregas_data = []
        for (i, k, t), cantidad in self.solution["q"].items():
            if cantidad > 0:
                entregas_data.append({
                    'periodo': t,
                    'cliente': i,
                    'vehiculo': k,
                    'cantidad': cantidad
                })
        
        # Exportar inventarios
        inventarios_data = []
        for (i, t), nivel in self.solution["I"].items():
            inventarios_data.append({
                'periodo': t,
                'nodo': i,
                'tipo': 'Depósito' if i == 0 else 'Cliente',
                'nivel': nivel,
                'capacidad': self.capacidad_almacenamiento.get(i, float('inf'))
            })
        
        # Guardar a CSV
        pd.DataFrame(rutas_data).to_csv(f"{directorio}/rutas.csv", index=False)
        pd.DataFrame(entregas_data).to_csv(f"{directorio}/entregas.csv", index=False)
        pd.DataFrame(inventarios_data).to_csv(f"{directorio}/inventarios.csv", index=False)


# Ejemplo de uso
def ejecutar_optimizacion():
    """
    Función principal que demuestra el uso del modelo
    """
    print("Modelo de Optimización de Distribución de GLP en Puno\n")
    
    # Instancia pequeña para demostración
    print("Creando instancia con 5 clientes, 3 períodos y 2 vehículos...")
    modelo = GLPOptimizationPuno(n_clientes=5, n_periodos=3, n_vehiculos=2)
    
    # Construir y resolver modelo
    print("Construyendo modelo de flujos...")
    modelo.construir_modelo_flujos()
    
    print("Resolviendo modelo...")
    inicio = time.time()
    estado = modelo.resolver(tiempo_limite=600)
    tiempo = time.time() - inicio
    
    print(f"Estado: {estado}")
    print(f"Tiempo: {tiempo:.2f} segundos")
    print(f"Costo total: S/ {modelo.costo_total:.2f}")
    
    # Generar reporte
    print("\nGenerando reporte detallado:")
    reporte = modelo.generar_reporte()
    
    # Mostrar desglose de costos
    print("\nDesglose de costos:")
    for categoria, valor in reporte["desglose_costos"].items():
        print(f"  - {categoria.capitalize()}: S/ {valor:.2f} ({valor/modelo.costo_total*100:.1f}%)")
    
    # Mostrar estadísticas por período
    print("\nEstadísticas por período:")
    for t, datos in reporte["estadisticas"]["por_periodo"].items():
        print(f"  Período {t}:")
        print(f"    - Clientes visitados: {datos['clientes_visitados']}")
        print(f"    - Vehículos utilizados: {datos['vehiculos_usados']}")
        print(f"    - Total entregado: {datos['total_entregado']:.1f} balones")
    
    # Mostrar rutas
    print("\nRutas generadas:")
    for t in modelo.T:
        print(f"  Período {t}:")
        if t in modelo.rutas:
            for k, ruta in modelo.rutas[t].items():
                tipo = "Camioneta" if k <= modelo.n_vehiculos // 3 else "Motocicleta"
                print(f"    - {tipo} {k}: {' -> '.join(map(str, ruta))}")
        else:
            print("    No hay rutas en este período")
    
    print("\nOptimización completada. Se pueden exportar los resultados o visualizar las rutas.")
    
    # Exportar resultados
    modelo.exportar_resultados()
    
    return modelo

# Si se ejecuta como script principal
if __name__ == "__main__":
    modelo = ejecutar_optimizacion()
