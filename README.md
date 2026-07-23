# 📦 Optimización de Cadena de Suministro: Demand Forecasting & Inventory Management

Este proyecto implementa una solución de **Machine Learning y Optimización de Inventarios (Newsvendor / Vendedor de Periódicos)** para predecir la demanda semanal por combinación de `Tienda x Producto` y generar recomendaciones de reabastecimiento que minimicen los costos combinados de **Stockout (faltante)** y **Overstock (exceso/almacenamiento)**.

---

## 🏗️ Estructura del Proyecto

```text
supply-chain-optimization/
│
├── data/                           # Archivos de datos de entrada y salida
│   ├── ventas.csv                  # Histórico de ventas semanales
│   ├── productos.csv               # Precios, costos unitarios y costos de almacenamiento
│   ├── inventario.csv              # Niveles actuales de stock por tienda/producto
│   └── plan_pedidos_recomendados.csv # Output generado con el plan de pedidos
│
├── models/                         # Binarios de los modelos entrenados por cuantil (.cbm)
│
├── notebooks/                      # Notebooks de exploración y experimentación
│   ├── 01_eda_and_features.ipynb
│   └── 02_experimentacion.ipynb
│
├── src/                            # Módulos Python empaquetados
│   ├── __init__.py
│   ├── data_loader.py              # Ingesta y validación de calidad de datos
│   ├── feature_engineering.py      # Generación de panel completo, lags y rolling stats
│   ├── model.py                    # Regresión cuantílica con CatBoost
│   └── optimization.py             # Lógica de Newsvendor y Cuantil Crítico
│
├── main.py                         # Script orquestador del pipeline end-to-end
├── requirements.txt                # Dependencias del proyecto
└── README.md                       # Documentación técnica
```

---

## ⚙️ Arquitectura de la Solución

El pipeline consta de 5 etapas principales integradas en un flujo libre de **Data Leakage**:

1. **Garantía de Continuidad del Panel (`feature_engineering.py`):**
   Crea la grilla completa de $Semanas 	imes Tiendas 	imes Productos$. Las combinaciones sin ventas registradas se rellenan explícitamente con 0 para reflejar periodos de nula demanda.

2. **Feature Engineering Anti-Leakage:**
   - **Lags:** $t-1, t-2, t-4$.
   - **Rolling Stats (Ventanas Móviles):** Promedio, desviación estándar y máximo sobre ventanas de 2, 4 y 8 semanas.
   - **Regla Estricta:** Aplicación de `.shift(1)` antes del cálculo de agregaciones móviles para evitar fuga de información futura.

3. **Regresión Cuantílica (`model.py`):**
   Uso de **CatBoostRegressor** con pérdida cuantílica para predecir la distribución empírica de la demanda en 3 cuantiles clave:
   - $Q_{0.10}$: Escenario pesimista.
   - $Q_{0.50}$: Estimación mediana (pronóstico base).
   - $Q_{0.90}$: Escenario optimista.

4. **Optimización Económica de Pedidos - Newsvendor (`optimization.py`):**
   Conexión de la incertidumbre estadística con la estructura financiera de cada SKU mediante el **Cuantil Crítico ($CR$)**:

   $$CR = \frac{C_u}{C_u + C_o}$$

   - $C_u$ (Costo de Stockout) $= \text{Precio de Venta} - \text{Costo Unitario}$
   - $C_o$ (Costo de Overstock) $= \text{Costo de Almacenamiento Semanal}$

   **Regla de Decisión:**
   - Si un producto tiene un alto margen y bajo costo de bodega ($CR 	o 1$), se prioriza evitar el faltante alineándose con cuantiles altos (ej. $Q_{0.90}$).
   - Si un producto tiene bajo margen y alto costo de bodega ($CR 	o 0$), la decisión adopta postura conservadora ($Q_{0.10}$).

5. **Cálculo Final de Reabastecimiento ($Q_{pedido}$):**

   $$\text{Demanda Objetivo} = \text{Interpolación}(CR, [Q_{0.10}, Q_{0.50}, Q_{0.90}])$$

   $$Q_{pedido} = \max(0, \lceil \text{Demanda Objetivo} - \text{Stock Actual} \rceil)$$

---

## 🚀 Guía de Instalación y Ejecución

### 1. Clonar el repositorio y preparar el entorno virtual

```bash
# Clonar el proyecto
git clone https://github.com/usuario/supply-chain-optimization.git
cd supply-chain-optimization

# Crear y activar entorno virtual
python -m venv venv

# En Linux/macOS:
source venv/bin/activate
# En Windows:
# venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Ejecutar el Pipeline Completo

Para correr la validación de datos, entrenamiento del modelo, evaluación de métricas (Pinball Loss & MAE) y exportación del plan de reabastecimiento:

```bash
python main.py
```

---

## 📊 Métricas de Evaluación

Para evaluar modelos predictivos con salida probabilística/cuantílica, se utiliza el **Pinball Loss (Quantile Loss)**:

$$L_q(y, \hat{y}_q) = \max(q(y - \hat{y}_q), (q - 1)(y - \hat{y}_q))$$

- **Pinball Loss $Q_{0.10}$**: Evalúa la precisión del límite inferior.
- **Pinball Loss $Q_{0.50}$ / MAE**: Evalúa la precisión central.
- **Pinball Loss $Q_{0.90}$**: Evalúa la cobertura del riesgo de desabastecimiento.

---

## ✒️ Autor
Proyecto desarrollado para el módulo de Optimización de Cadena de Suministro y Machine Learning Aplicado.
