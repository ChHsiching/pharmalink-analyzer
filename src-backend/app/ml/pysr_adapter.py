"""PySRResultAdapter — isolates PySR DataFrame schema from service code."""
from __future__ import annotations

from typing import Any


class PySRResultAdapter:
    """Wraps a fitted PySR model to provide a stable equation-extraction interface.

    Centralises all knowledge about the PySR ``equations_`` DataFrame schema
    (e.g. the ``pick`` column removed in newer PySR versions) so downstream
    code never touches the DataFrame directly.
    """

    def __init__(self, model: Any) -> None:
        self._model = model
        self._eq = model.equations_

    def get_best_equation(self) -> dict:
        """Return the single best equation selected by the model."""
        row = self._best_row()
        return {
            "sympy_expr": self._model.sympy(),
            "latex": self._model.latex(),
            "complexity": int(row["complexity"]),
            "loss": float(row["loss"]),
        }

    def get_pareto_equations(self) -> list[dict]:
        """Return every equation on the Pareto front."""
        results: list[dict] = []
        for idx in range(len(self._eq)):
            row = self._eq.iloc[idx]
            results.append({
                "index": idx,
                "latex": self._model.latex(index=idx),
                "complexity": int(row["complexity"]),
                "loss": float(row["loss"]),
                "sympy_expr": self._model.sympy(index=idx),
            })
        return results

    def _best_row(self):
        """Locate the DataFrame row for the model's chosen best equation.

        * Old PySR (``pick`` column present): use the marked row.
        * New PySR (no ``pick`` column): filter to equations within 1.5x of
          minimum loss, then pick the highest ``score`` — matching PySR's
          ``model_selection="best"`` implementation.
        """
        if "pick" in self._eq.columns:
            idx = self._eq.query("pick == True").index[0]
        else:
            threshold = 1.5 * self._eq["loss"].min()
            filtered = self._eq[self._eq["loss"] <= threshold]
            idx = filtered["score"].idxmax()
        return self._eq.iloc[idx]
