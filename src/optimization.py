import pandas as pd
import numpy as np


def calcular_cuantil_critico(
    precio_venta: float, costo_unitario: float, costo_almacenamiento_semanal: float
) -> float:
    """Calcula el Cuantil Crítico (Critical Ratio) de la teoría de Newsvendor.

    CR = Cu / (Cu + Co)
    Cu: Costo de Stockout (Margen no ganado = Precio - Costo)
    Co: Costo de Overstock (Costo de mantener inventario semanal)
    """
    costo_faltante_cu = max(0.0, precio_venta - costo_unitario)
    costo_exceso_co = max(0.01, costo_almacenamiento_semanal)  # Evitar div por 0

    cr = costo_faltante_cu / (costo_faltante_cu + costo_exceso_co)
    return float(np.clip(cr, 0.01, 0.99))


def optimizar_pedidos(
    df_predicciones: pd.DataFrame,
    df_productos: pd.DataFrame,
    df_inventario: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula la cantidad recomendada a pedir (Q_pedido) integrando las

    predicciones cuantílicas con las variables económicas de cada producto.
    """
    # Unir información financiera de producto e inventario actual
    df_opt = df_predicciones.copy()
    df_opt = df_opt.merge(df_productos, on="producto_id", how="left")
    df_opt = df_opt.merge(
        df_inventario[["tienda_id", "producto_id", "stock_actual"]],
        on=["tienda_id", "producto_id"],
        how="left",
    )
    df_opt["stock_actual"] = df_opt["stock_actual"].fillna(0)

    # 1. Calcular Cuantil Crítico por registro
    df_opt["cuantil_critico"] = df_opt.apply(
        lambda row: calcular_cuantil_critico(
            row["precio_venta"],
            row["costo_unitario"],
            row["costo_almacenamiento_semanal"],
        ),
        axis=1,
    )

    # 2. Interpolación lineal de la demanda objetivo según el cuantil crítico
    # Interpolamos entre los cuantiles calculados (0.10, 0.50, 0.90)
    def interpolar_demanda_objetivo(row):
        cr = row["cuantil_critico"]
        q10 = row.get("pred_q_0.1", row.get("pred_q_0.5", 0))
        q50 = row.get("pred_q_0.5", 0)
        q90 = row.get("pred_q_0.9", row.get("pred_q_0.5", 0))

        if cr <= 0.50:
            return q10 + (cr - 0.10) * (q50 - q10) / (0.50 - 0.10)
        else:
            return q50 + (cr - 0.50) * (q90 - q50) / (0.90 - 0.50)

    df_opt["demanda_objetivo"] = df_opt.apply(interpolar_demanda_objetivo, axis=1)

    # 3. Regla de decisión de inventario: Q_pedido = Max(0, Demanda_Objetivo - Stock_Actual)
    df_opt["cantidad_a_pedir"] = np.maximum(
        0, np.ceil(df_opt["demanda_objetivo"] - df_opt["stock_actual"])
    )

    return df_opt[
        [
            "tienda_id",
            "producto_id",
            "stock_actual",
            "cuantil_critico",
            "demanda_objetivo",
            "cantidad_a_pedir",
        ]
    ]