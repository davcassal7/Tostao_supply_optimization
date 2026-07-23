import pandas as pd
import numpy as np


def completar_panel_semanal(df_ventas: pd.DataFrame) -> pd.DataFrame:
    """Garantiza la continuidad del panel creando la grilla completa de

    Semanas x Tienda x Producto, rellenando con 0 la demanda no observada.
    """
    # Crear la columna año-semana si no existe
    if "semana_anio" not in df_ventas.columns and "fecha" in df_ventas.columns:
        df_ventas["semana_anio"] = (
            df_ventas["fecha"].dt.isocalendar().year.astype(str)
            + "-W"
            + df_ventas["fecha"].dt.isocalendar().week.astype(str).str.zfill(2)
        )

    tiendas = df_ventas["tienda_id"].unique()
    productos = df_ventas["producto_id"].unique()
    semanas = df_ventas["semana_anio"].unique()

    # Producto cartesiano completo
    index_multi = pd.MultiIndex.from_product(
        [semanas, tiendas, productos], names=["semana_anio", "tienda_id", "producto_id"]
    )
    panel = pd.DataFrame(index=index_multi).reset_index()

    # Agrupar ventas observadas
    ventas_agrupadas = (
        df_ventas.groupby(["semana_anio", "tienda_id", "producto_id"])["unidades_vendidas"]
        .sum()
        .reset_index()
    )

    # Unir panel con ventas y rellenar ceros
    df_panel = pd.merge(
        panel, ventas_agrupadas, on=["semana_anio", "tienda_id", "producto_id"], how="left"
    )
    df_panel["unidades_vendidas"] = df_panel["unidades_vendidas"].fillna(0)

    # Ordenar cronológicamente para el cálculo de variables temporales
    df_panel = df_panel.sort_values(by=["tienda_id", "producto_id", "semana_anio"]).reset_index(
        drop=True
    )

    return df_panel


def crear_variables_modelo(
    df_panel: pd.DataFrame, lags: list = [1, 2, 4], ventanas_rolling: list = [2, 4, 8]
) -> pd.DataFrame:
    """Genera características temporales (lags, rolling stats) evitando Data Leakage

    mediante el uso de shift(1) previo al cálculo de ventanas móviles.
    """
    df = df_panel.copy()

    # Agrupación base para transformar por serie
    grupo = df.groupby(["tienda_id", "producto_id"])["unidades_vendidas"]

    # 1. Variables Rezagadas (Lags)
    for lag in lags:
        df[f"demanda_lag_{lag}"] = grupo.shift(lag)

    # 2. Estadísticas Móviles (Rolling Stats con shift(1) estricto)
    for w in ventanas_rolling:
        df[f"demanda_rolling_mean_{w}"] = grupo.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
        )
        df[f"demanda_rolling_std_{w}"] = grupo.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).std()
        ).fillna(0)
        df[f"demanda_rolling_max_{w}"] = grupo.transform(
            lambda x: x.shift(1).rolling(window=w, min_periods=1).max()
        )

    # 3. Extraer semana numérica para componentes estacionales simples
    df["num_semana"] = df["semana_anio"].apply(lambda x: int(x.split("-W")[-1]))

    return df