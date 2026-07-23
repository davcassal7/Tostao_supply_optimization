from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.data_loader import cargar_datos, validar_calidad_datos
from src.feature_engineering import completar_panel_semanal, crear_variables_modelo
from src.optimization import optimizar_pedidos


CUANTILES_A_COLUMNAS = {
    0.10: "demanda_p10",
    0.50: "demanda_p50",
    0.90: "demanda_p90",
}


def cargar_configuracion(ruta_config: Path) -> dict[str, Any]:
    """Carga y valida la configuracion utilizada en el entrenamiento final."""
    if not ruta_config.exists():
        raise FileNotFoundError(
            f"No se encontro la configuracion del modelo: {ruta_config}"
        )

    with ruta_config.open("r", encoding="utf-8") as archivo:
        config = json.load(archivo)

    requeridas = {"lags", "ventanas", "cuantiles"}
    faltantes = requeridas.difference(config)
    if faltantes:
        raise ValueError(
            "model_config.json no contiene las claves requeridas: "
            f"{sorted(faltantes)}"
        )

    return config


def cargar_modelos_catboost(
    directorio_modelos: Path,
    cuantiles: list[float] | tuple[float, ...],
) -> dict[float, CatBoostRegressor]:
    """Carga un modelo CatBoost nativo por cada cuantil configurado."""
    modelos: dict[float, CatBoostRegressor] = {}

    for cuantil_original in cuantiles:
        cuantil = float(cuantil_original)
        ruta_modelo = (
            directorio_modelos
            / f"modelo_demanda_p{int(cuantil * 100):02d}.cbm"
        )

        if not ruta_modelo.exists():
            raise FileNotFoundError(f"No se encontro el modelo: {ruta_modelo}")

        modelo = CatBoostRegressor()
        modelo.load_model(str(ruta_modelo))
        modelos[cuantil] = modelo

    return modelos


def agregar_semana_futura(
    panel: pd.DataFrame,
    columna_semana: str = "semana_anio",
    columna_objetivo: str = "unidades_vendidas",
) -> tuple[pd.DataFrame, Any]:
    """Agrega una fila futura por tienda-producto sin demanda observada."""
    columnas_clave = ["tienda_id", "producto_id"]
    faltantes = [
        columna for columna in columnas_clave + [columna_semana, columna_objetivo]
        if columna not in panel.columns
    ]
    if faltantes:
        raise ValueError(
            "El panel no contiene las columnas necesarias para inferencia: "
            f"{faltantes}"
        )

    panel = panel.sort_values(columnas_clave + [columna_semana]).copy()
    ultima_semana = panel[columna_semana].max()

    if pd.api.types.is_datetime64_any_dtype(panel[columna_semana]):
        semana_futura = pd.Timestamp(ultima_semana) + pd.Timedelta(weeks=1)
    else:
        semana_futura = ultima_semana + 1

    filas_futuras = (
        panel.groupby(columnas_clave, observed=True, as_index=False)
        .tail(1)
        .copy()
    )
    filas_futuras[columna_semana] = semana_futura
    filas_futuras[columna_objetivo] = np.nan

    panel_ampliado = pd.concat(
        [panel, filas_futuras],
        ignore_index=True,
    )

    return panel_ampliado, semana_futura


def obtener_orden_variables(config: dict[str, Any]) -> list[str]:
    """Obtiene el mismo orden de variables usado durante entrenamiento."""
    if config.get("orden_variables"):
        return list(config["orden_variables"])

    numericas = list(config.get("variables_numericas", []))
    categoricas = list(config.get("variables_categoricas", []))
    variables = numericas + categoricas

    if not variables:
        raise ValueError(
            "La configuracion debe incluir 'orden_variables' o las listas "
            "'variables_numericas' y 'variables_categoricas'."
        )

    return variables


def preparar_matriz_inferencia(
    datos: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    """Prepara la matriz futura respetando esquema, tipos y orden de columnas."""
    variables = obtener_orden_variables(config)
    categoricas = list(config.get("variables_categoricas", []))
    numericas = [columna for columna in variables if columna not in categoricas]

    faltantes = [columna for columna in variables if columna not in datos.columns]
    if faltantes:
        raise ValueError(
            "Las variables futuras no coinciden con el entrenamiento. "
            f"Faltan: {faltantes}"
        )

    matriz = datos[variables].copy()

    for columna in numericas:
        matriz[columna] = pd.to_numeric(matriz[columna], errors="coerce")
        matriz[columna] = matriz[columna].replace([np.inf, -np.inf], np.nan)

    for columna in categoricas:
        matriz[columna] = matriz[columna].fillna("DESCONOCIDO").astype(str)

    return matriz


def corregir_cruce_cuantiles(pronostico: pd.DataFrame) -> pd.DataFrame:
    """Garantiza que P10 sea menor o igual que P50 y P90."""
    columnas = ["demanda_p10", "demanda_p50", "demanda_p90"]
    presentes = [columna for columna in columnas if columna in pronostico.columns]

    if len(presentes) == 3:
        pronostico[presentes] = np.sort(
            pronostico[presentes].to_numpy(dtype=float),
            axis=1,
        )

    return pronostico


def main() -> None:
    """Ejecuta inferencia y optimizacion usando modelos ya entrenados."""
    print("\nPIPELINE DE OPTIMIZACION DE CADENA DE SUMINISTRO\n")

    directorio_base = Path(__file__).resolve().parent
    directorio_datos = directorio_base / "data"
    directorio_modelos = directorio_base / "models"
    directorio_salidas = directorio_base / "outputs"
    directorio_salidas.mkdir(parents=True, exist_ok=True)

    ruta_ventas = directorio_datos / "ventas.csv"
    ruta_productos = directorio_datos / "productos.csv"
    ruta_inventario = directorio_datos / "inventario.csv"
    ruta_config = directorio_modelos / "model_config.json"
    ruta_pronostico = directorio_salidas / "pronostico_demanda.csv"
    ruta_pedidos = directorio_salidas / "plan_pedidos_recomendados.csv"

    print("[1/6] Cargando y validando datos...")
    df_ventas, df_productos, df_inventario = cargar_datos(
        ruta_ventas,
        ruta_productos,
        ruta_inventario,
    )
    reporte = validar_calidad_datos(df_ventas, df_productos, df_inventario)
    print(f" - Registros de ventas: {reporte['total_registros_ventas']}")
    print(
        f" - Tiendas: {reporte['tiendas_unicas']} | "
        f"Productos: {reporte['productos_unicos']}"
    )

    if reporte.get("inconsistencias_precios", 0) > 0:
        print(
            " - ADVERTENCIA: se detectaron "
            f"{reporte['inconsistencias_precios']} inconsistencias de precios."
        )

    print("\n[2/6] Cargando configuracion y modelos finales...")
    config = cargar_configuracion(ruta_config)
    cuantiles = tuple(float(valor) for valor in config["cuantiles"])
    modelos = cargar_modelos_catboost(directorio_modelos, cuantiles)
    print(f" - Lags: {config['lags']}")
    print(f" - Ventanas: {config['ventanas']}")
    print(f" - Modelos cargados: {sorted(modelos)}")

    print("\n[3/6] Construyendo semana futura y variables...")
    panel = completar_panel_semanal(df_ventas)
    panel_futuro, semana_pronostico = agregar_semana_futura(panel)

    features = crear_variables_modelo(
        panel_futuro,
        lags=tuple(int(valor) for valor in config["lags"]),
        ventanas_rolling=tuple(int(valor) for valor in config["ventanas"]),
    )

    datos_futuros = features[
        features["semana_anio"] == semana_pronostico
    ].copy()

    if datos_futuros.empty:
        raise ValueError("No se generaron registros para la semana futura.")

    matriz_futura = preparar_matriz_inferencia(datos_futuros, config)
    print(f" - Semana pronosticada: {semana_pronostico}")
    print(f" - Combinaciones a pronosticar: {len(datos_futuros)}")

    print("\n[4/6] Generando pronosticos P10, P50 y P90...")
    pronostico = datos_futuros[
        ["tienda_id", "producto_id"]
    ].reset_index(drop=True)

    for cuantil, modelo in modelos.items():
        columna = CUANTILES_A_COLUMNAS.get(
            cuantil,
            f"demanda_p{int(cuantil * 100):02d}",
        )
        pronostico[columna] = np.clip(
            modelo.predict(matriz_futura),
            a_min=0,
            a_max=None,
        )

    pronostico = corregir_cruce_cuantiles(pronostico)
    pronostico.insert(0, "semana_pronostico", semana_pronostico)
    pronostico.to_csv(ruta_pronostico, index=False)
    print(f" - Pronostico guardado en: {ruta_pronostico}")

    print("\n[5/6] Optimizando pedidos con Newsvendor...")
    plan_pedidos = optimizar_pedidos(
        pronostico,
        df_productos,
        df_inventario,
    )

    print("\n[6/6] Guardando plan recomendado...")
    plan_pedidos.to_csv(ruta_pedidos, index=False)
    print(f" - Plan guardado en: {ruta_pedidos}")
    print("\nPROCESO FINALIZADO CON EXITO\n")
    print(plan_pedidos.head())


if __name__ == "__main__":
    main()
