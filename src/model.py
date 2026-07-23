import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from typing import List, Dict


class ModeloPronostico:
    """Clase para envolver el modelo de regresión cuantílica con CatBoost."""

    def __init__(self, cuantiles: List[float] = [0.10, 0.50, 0.90]):
        self.cuantiles = cuantiles
        self.modelos: Dict[float, CatBoostRegressor] = {}

    def entrenar(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cat_features: List[str] = None,
        iterations: int = 500,
        verbose: int = 0,
    ):
        """Entrena un modelo CatBoost dedicado para cada cuantil especificado."""
        for q in self.cuantiles:
            modelo = CatBoostRegressor(
                iterations=iterations,
                loss_function=f"Quantile:alpha={q}",
                eval_metric=f"Quantile:alpha={q}",
                random_seed=42,
                verbose=verbose,
            )
            modelo.fit(X_train, y_train, cat_features=cat_features)
            self.modelos[q] = modelo

    def predecir(self, X: pd.DataFrame) -> pd.DataFrame:
        """Genera predicciones para todos los cuantiles entrenados.

        Returns:
            pd.DataFrame: DataFrame con columnas 'pred_q_0.1', 'pred_q_0.5',
            etc.
        """
        predicciones = pd.DataFrame(index=X.index)
        for q, modelo in self.modelos.items():
            # Las predicciones de demanda no pueden ser negativas
            predicciones[f"pred_q_{q}"] = np.maximum(0, modelo.predict(X))

        return predicciones