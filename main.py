import os
import sys
import pandas as pd

# Importar el paquete modular src
from src.data_loader import cargar_datos, validar_calidad_datos
from src.feature_engineering import completar_panel_semanal, crear_variables_modelo
from src.model import ModeloPronostico
from src.optimization import optimizar_pedidos


def main():
  print('==================================================')
  print('   PIPELINE DE OPTIMIZACIÓN DE CADENA DE SUMINISTRO')
  print('==================================================\n')

  # 1. Definición de rutas
  DIR_BASE = os.path.dirname(os.path.abspath(__file__))
  RUTA_VENTAS = os.path.join(DIR_BASE, 'data', 'ventas.csv')
  RUTA_PRODUCTOS = os.path.join(DIR_BASE, 'data', 'productos.csv')
  RUTA_INVENTARIO = os.path.join(DIR_BASE, 'data', 'inventario.csv')
  DIR_MODELOS = os.path.join(DIR_BASE, 'models')
  RUTA_SALIDA_PEDIDOS = os.path.join(
      DIR_BASE, 'data', 'plan_pedidos_recomendados.csv'
  )

  # 2. Carga y Validación de Calidad de Datos
  print('[1/6] Cargando y validando datos de entrada...')
  df_ventas, df_productos, df_inventario = cargar_datos(
      RUTA_VENTAS, RUTA_PRODUCTOS, RUTA_INVENTARIO
  )
  reporte_calidad = validar_calidad_datos(
      df_ventas, df_productos, df_inventario
  )

  print(
      f" - Registros de ventas cargados: {reporte_calidad['total_registros_ventas']}"
  )
  print(
      f" - Tiendas únicas: {reporte_calidad['tiendas_unicas']} | Productos"
      f" únicos: {reporte_calidad['productos_unicos']}"
  )
  if reporte_calidad['inconsistencias_precios'] > 0:
    print(
        ' - [ADVERTENCIA]: Se detectaron'
        f" {reporte_calidad['inconsistencias_precios']} productos con precio de"
        ' venta menor al costo.'
    )

  # 3. Construcción del Panel Continuo y Feature Engineering
  print('\n[2/6] Generando panel semanal y variables temporales (Lags &' ' Rollings)...')
  df_panel = completar_panel_semanal(df_ventas)
  df_features = crear_variables_modelo(
      df_panel, lags=[1, 2, 4], ventanas_rolling=[2, 4, 8]
  )

  # Eliminar filas iniciales producidas por los retardos (NaNs)
  df_model = df_features.dropna().copy()

  # 4. División de Datos en Entrenamiento y Prueba (Temporal Train/Test Split)
  print('\n[3/6] Realizando división temporal Train/Test...')
  semanas = sorted(df_model['semana_anio'].unique())
  # Tomamos las últimas 4 semanas para prueba (evaluación fuera de muestra)
  corte_semana = semanas[-4]

  train_data = df_model[df_model['semana_anio'] < corte_semana]
  test_data = df_model[df_model['semana_anio'] >= corte_semana]

  cols_excluir = ['semana_anio', 'unidades_vendidas']
  features = [c for c in df_model.columns if c not in cols_excluir]
  cat_features = ['tienda_id', 'producto_id']

  # Convertir variables categóricas a string para CatBoost
  for cat in cat_features:
    train_data[cat] = train_data[cat].astype(str)
    test_data[cat] = test_data[cat].astype(str)
    df_model[cat] = df_model[cat].astype(str)

  X_train, y_train = train_data[features], train_data['unidades_vendidas']
  X_test, y_test = test_data[features], test_data['unidades_vendidas']

  # 5. Entrenamiento del Modelo Predictor y Evaluación
  print('\n[4/6] Entrenando modelo de Regresión Cuantílica (CatBoost)...')
  modelo = ModeloPronostico(
      cuantiles=[0.10, 0.50, 0.90],
      depth=6,
      learning_rate=0.05,
      iterations=500,
  )
  modelo.entrenar(
      X_train, y_train, cat_features=cat_features, verbose=0
  )

  metricas = modelo.evaluar(X_test, y_test)
  print(' - Desempeño en Test Set (Pinball Loss & MAE):')
  for k, v in metricas.items():
    print(f'   * {k}: {v:.4f}')

  # Guardar los modelos entrenados
  modelo.guardar_modelos(DIR_MODELOS)
  print(f' - Modelos guardados en: {DIR_MODELOS}')

  # 6. Generar Predicciones para la Última Semana Conocida (Ventana de Abastecimiento)
  print('\n[5/6] Generando pronóstico de demanda futura por cuantil...')
  semana_reciente = semanas[-1]
  df_futuro = df_model[df_model['semana_anio'] == semana_reciente].copy()

  predicciones = modelo.predecir(df_futuro[features])
  df_pred_unidas = pd.concat(
      [df_futuro[['tienda_id', 'producto_id']].reset_index(drop=True),
       predicciones.reset_index(drop=True)],
      axis=1,
  )

  # 7. Optimización de Inventarios (Newsvendor / Cuantil Crítico)
  print('\n[6/6] Calculando cantidades óptimas a pedir (Newsvendor)...')
  plan_pedidos = optimizar_pedidos(
      df_pred_unidas, df_productos, df_inventario
  )

  # Guardar resultados finales
  plan_pedidos.to_csv(RUTA_SALIDA_PEDIDOS, index=False)
  print('==================================================')
  print('   ¡PROCESO FINALIZADO CON ÉXITO!')
  print(f'   Plan de reabastecimiento exportado a:\n   {RUTA_SALIDA_PEDIDOS}')
  print('==================================================\n')

  # Mostrar muestra del plan generado
  print('Resumen de los primeros 5 pedidos recomendados:')
  print(plan_pedidos.head())


if __name__ == '__main__':
  main()