# Optimización de Abastecimiento con Pronóstico Probabilístico

Solución de ciencia de datos para pronosticar la demanda de la siguiente semana por combinación **SKU-Tienda** y recomendar cantidades de pedido que minimicen el costo esperado de **stockout** y **overstock**.

El proyecto combina un modelo global de regresión cuantílica con CatBoost y una política de inventario tipo **Newsvendor**. El modelo genera los cuantiles P10, P50 y P90, y la optimización selecciona el nivel de demanda objetivo según el margen unitario, el costo semanal de almacenamiento y el inventario actual.

## Objetivos

1. Pronosticar la demanda semanal por `id_tienda` e `id_producto`.
2. Representar la incertidumbre mediante cuantiles predictivos P10, P50 y P90.
3. Transformar el pronóstico en una recomendación de pedido mediante la razón crítica de Newsvendor.
4. Comparar modelos y baselines mediante backtesting temporal.
5. Generar un archivo reproducible con los pedidos recomendados.

## Datos de entrada

El pipeline espera los siguientes archivos en `data/raw/`:

- `ventas_historicas.csv`: `fecha`, `id_tienda`, `id_producto`, `unidades_vendidas`.
- `inventario_actual.csv`: `id_tienda`, `id_producto`, `stock_actual`.
- `catalogo_productos.csv`: `id_producto`, `nombre`, `costo_unitario`, `precio_venta`, `costo_almacenamiento_semanal`.
- `maestro_tiendas.csv`: `id_tienda`, `ciudad`, `tamaño_m2`.

El histórico disponible comprende el periodo entre el **1 de enero y el 31 de marzo de 2024**, equivalente a 13 semanas. El pronóstico final corresponde a la semana que inicia el **1 de abril de 2024**.

> **Privacidad de datos:** antes de publicar el repositorio, verifica que los archivos originales puedan compartirse. Si son confidenciales, no los incluyas en GitHub y conserva únicamente la estructura de carpetas y las instrucciones para ubicarlos localmente.

## Arquitectura de la solución

```text
Datos crudos
    ↓
Validación de esquema y calidad
    ↓
Panel continuo SKU-Tienda-Semana
    ↓
Lags y estadísticas móviles sin fuga de información
    ↓
Backtesting temporal y comparación de modelos
    ↓
Modelos CatBoost cuantílicos P10, P50 y P90
    ↓
Pronóstico probabilístico de la siguiente semana
    ↓
Razón crítica Newsvendor por SKU-Tienda
    ↓
Pedido recomendado
```

### 1. Preparación de datos

- Validación de columnas, duplicados, fechas y valores negativos.
- Agregación semanal por tienda y producto.
- Construcción de un panel continuo para evitar que un rezago represente la última semana con registro en lugar de la semana inmediatamente anterior.
- Imputación de cero ventas en semanas ausentes, bajo el supuesto de que la combinación SKU-Tienda estaba activa.
- Exclusión del inventario actual como predictor histórico. El inventario se incorpora únicamente en la etapa de optimización.

### 2. Ingeniería de variables

Debido a que solo existen 13 semanas de historia, se priorizaron variables de recencia:

- Rezagos de 1, 2, 3, 4 y 6 semanas para la configuración probabilística seleccionada.
- Promedios, desviaciones y otras estadísticas móviles de 2 y 4 semanas.
- Proporción reciente de semanas con ventas iguales a cero.
- Atributos de producto y tienda.
- Aplicación de `shift(1)` antes de las ventanas móviles para evitar fuga de información.

### 3. Experimentación

Se utilizó backtesting temporal con origen expansivo sobre las últimas cuatro semanas. Se compararon:

- Naive de una semana.
- Media móvil de dos semanas.
- Media móvil de cuatro semanas.
- CatBoost con dos rezagos.
- CatBoost con rezagos de 1 a 4 semanas.
- CatBoost con rezagos de 1, 2, 3, 4 y 6 semanas.

La media móvil de cuatro semanas obtuvo el mejor desempeño puntual. CatBoost con rezagos de 1, 2, 3, 4 y 6 semanas fue seleccionado como modelo probabilístico porque mantuvo un desempeño cercano al mejor baseline y permite estimar los cuantiles requeridos por la optimización.

### 4. Pronóstico probabilístico

Se entrena un modelo CatBoost independiente para cada cuantil:

- **P10:** escenario de demanda baja.
- **P50:** mediana de la demanda.
- **P90:** escenario de demanda alta.

Los modelos finales se almacenan en formato nativo de CatBoost:

```text
models/
├── modelo_demanda_p10.cbm
├── modelo_demanda_p50.cbm
├── modelo_demanda_p90.cbm
└── model_config.json
```

`model_config.json` conserva los rezagos, ventanas, cuantiles, variables categóricas, variables numéricas y el orden de las columnas utilizado durante el entrenamiento.

### 5. Optimización Newsvendor

Para cada SKU-Tienda se define:

- Costo de faltante:

```text
Cu = precio_venta - costo_unitario
```

- Costo de exceso:

```text
Co = costo_almacenamiento_semanal
```

- Razón crítica:

```text
CR = Cu / (Cu + Co)
```

La demanda objetivo corresponde al cuantil de la distribución asociado a `CR`. Como el modelo estima P10, P50 y P90, el valor se obtiene mediante interpolación entre los cuantiles disponibles.

El pedido final es:

```text
pedido_recomendado = max(demanda_objetivo - stock_actual, 0)
```

Los productos con mayor margen y menor costo de almacenamiento reciben una política más agresiva. Los productos con menor margen o mayor costo de almacenamiento reciben una política más conservadora.

## Resultados de experimentación

Resultados promedio del backtesting para el pronóstico central:

| Modelo | WAPE promedio | Bias promedio |
|---|---:|---:|
| Media móvil 4 semanas | 11,64% | -0,13% |
| CatBoost, rezagos 1, 2, 3, 4 y 6 | 11,94% | -0,14% |
| Media móvil 2 semanas | 12,08% | -0,13% |
| CatBoost, rezagos 1 a 4 | 12,14% | -0,28% |
| CatBoost, rezagos 1 y 2 | 12,55% | -0,67% |
| Naive, rezago 1 | 13,14% | -0,11% |

Las métricas completas se encuentran en:

```text
outputs/metrics/
├── comparacion_cuantiles.csv
├── comparacion_modelos_p50.csv
└── metricas_experimentos.csv
```

También se reportan Pinball Loss y cobertura empírica para evaluar la calidad de los cuantiles. Debido al horizonte histórico limitado, la calibración probabilística debe interpretarse como evidencia preliminar.

## Estructura del repositorio

```text
supply-chain-optimization/
├── data/
│   ├── raw/
│   │   ├── ventas_historicas.csv
│   │   ├── catalogo_productos.csv
│   │   ├── inventario_actual.csv
│   │   └── maestro_tiendas.csv
│   └── processed/
│       └── panel_semanal.parquet
├── models/
│   ├── modelo_demanda_p10.cbm
│   ├── modelo_demanda_p50.cbm
│   ├── modelo_demanda_p90.cbm
│   └── model_config.json
├── notebooks/
│   ├── 01_EDA.ipynb
│   └── 02_Experimentacion.ipynb
├── outputs/
│   ├── metrics/
│   ├── predictions/
│   └── pedidos_recomendados.csv
├── presentation/
│   ├── presentacion_ejecutiva.pptx
│   └── presentacion_ejecutiva.pdf
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── model.py
│   └── optimization.py
├── main.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Instalación

### Requisitos

- Python 3.10 o superior recomendado.
- Git.

### 1. Clonar el repositorio

```bash
git clone https://github.com/TU_USUARIO/supply-chain-optimization.git
cd supply-chain-optimization
```

Reemplaza `TU_USUARIO` por el usuario real de GitHub antes de publicar la entrega.

### 2. Crear y activar un entorno virtual

Linux o macOS:

```bash
python -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Instalar dependencias

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Ejecución

### Pipeline end-to-end

Desde la raíz del repositorio:

```bash
python main.py
```

El script principal:

1. Carga y valida los datos.
2. Carga `model_config.json`.
3. Carga los modelos P10, P50 y P90 ya entrenados.
4. Reconstruye las variables requeridas para inferencia.
5. Pronostica la semana siguiente.
6. Corrige posibles cruces entre cuantiles.
7. Calcula la razón crítica de Newsvendor.
8. Genera el pedido recomendado.

Salidas principales:

```text
outputs/predictions/pronostico_semana_2024-04-01.csv
outputs/pedidos_recomendados.csv
```

### Notebooks

```bash
jupyter lab
```

Orden recomendado:

1. `notebooks/01_EDA.ipynb`
2. `notebooks/02_Experimentacion.ipynb`

Los notebooks documentan el análisis y la selección del modelo. La ejecución productiva se realiza mediante `main.py` y los módulos de `src/`.

## Métricas de evaluación

### Pronóstico puntual

- **MAE:** error absoluto promedio.
- **WAPE:** error absoluto ponderado por el volumen real.
- **Bias:** identifica sobreestimación o subestimación sistemática.

### Pronóstico probabilístico

- **Pinball Loss:** evalúa cada cuantil de acuerdo con su nivel objetivo.
- **Cobertura:** proporción de observaciones reales por debajo del cuantil estimado.
- **Cruce de cuantiles:** validación de que P10 sea menor o igual que P50, y P50 menor o igual que P90.

### Negocio

- Costo de stockout.
- Costo de overstock.
- Costo total esperado.
- Unidades faltantes y sobrantes.
- Pedido recomendado por SKU-Tienda.

## Limitaciones

- Solo se dispone de 13 semanas de información, por lo que no puede estimarse de forma robusta la estacionalidad anual.
- No existe inventario histórico ni una señal histórica de stockout. Las ventas observadas se utilizan como aproximación de la demanda.
- Completar semanas ausentes con cero supone que el producto estaba activo y disponible.
- No se incluyen promociones, festivos, lead times, inventario en tránsito, mínimos de compra ni múltiplos de empaque.
- El costo de stockout se aproxima mediante el margen unitario perdido.
- El costo de overstock se aproxima mediante el costo semanal de almacenamiento.
- La evaluación económica histórica es una simulación de políticas de stock objetivo, no una reconstrucción completa de decisiones históricas.
- El uso de P10, P50 y P90 limita la interpolación de la razón crítica a los cuantiles estimados.

## Próximos pasos

- Incorporar inventario histórico y eventos de stockout.
- Agregar promociones, festivos y variables externas.
- Modelar lead time e inventario en tránsito.
- Incorporar mínimos de compra, múltiplos de empaque, presupuesto y capacidad.
- Ampliar el histórico para capturar estacionalidad.
- Monitorear drift, cobertura de cuantiles y desempeño económico.
- Automatizar reentrenamiento y versionamiento de artefactos.

## Reproducibilidad

Antes de publicar o entregar el repositorio se recomienda comprobar:

```bash
python main.py
```

El pipeline debe ejecutarse desde la raíz sin rutas absolutas y debe generar las salidas documentadas. Las versiones utilizadas se especifican en `requirements.txt`.

## Autor

**David Castiblanco**  
Proyecto desarrollado como caso técnico de pronóstico de demanda y optimización de abastecimiento.
