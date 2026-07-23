import pandas as pd
from typing import Tuple, Dict, Any


def cargar_datos(
    ruta_ventas: str, ruta_productos: str, ruta_inventario: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los conjuntos de datos principales desde archivos CSV.

    Args:
        ruta_ventas (str): Ruta al CSV de ventas históricas.
        ruta_productos (str): Ruta al CSV de catálogo de productos.
        ruta_inventario (str): Ruta al CSV de estado de inventarios.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: DataFrames de ventas,
        productos e inventario.
    """
    df_ventas = pd.read_csv(ruta_ventas)
    df_productos = pd.read_csv(ruta_productos)
    df_inventario = pd.read_csv(ruta_inventario)

    # Conversión de tipos de datos esenciales
    if "fecha" in df_ventas.columns:
        df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])

    return df_ventas, df_productos, df_inventario


def validar_calidad_datos(
    df_ventas: pd.DataFrame, df_productos: pd.DataFrame, df_inventario: pd.DataFrame
) -> Dict[str, Any]:
    """Realiza chequeos de calidad e integridad sobre los conjuntos de datos.

    Args:
        df_ventas (pd.DataFrame): DataFrame de ventas.
        df_productos (pd.DataFrame): DataFrame de productos.
        df_inventario (pd.DataFrame): DataFrame de inventario.

    Returns:
        Dict[str, Any]: Diccionario con el resumen de validaciones.
    """
    reporte = {
        "nulos_ventas": df_ventas.isnull().sum().to_dict(),
        "duplicados_ventas": int(df_ventas.duplicated().sum()),
        "total_registros_ventas": len(df_ventas),
        "tiendas_unicas": (
            df_ventas["tienda_id"].nunique()
            if "tienda_id" in df_ventas.columns
            else 0
        ),
        "productos_unicos": (
            df_ventas["producto_id"].nunique()
            if "producto_id" in df_ventas.columns
            else 0
        ),
        "inconsistencias_precios": int(
            (df_productos["precio_venta"] < df_productos["costo_unitario"]).sum()
        )
        if "precio_venta" in df_productos.columns
        else 0,
    }
    return reporte