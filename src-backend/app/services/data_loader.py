from pathlib import Path
import pandas as pd

from app.config import DATA_DIR
from app.models.dataset import ColumnStats, DatasetMeta, DatasetDetail, DatasetStatsResponse


class DataLoader:
    def __init__(self, data_dir: Path | None = None):
        self._data_dir = data_dir or DATA_DIR
        self._datasets: dict[str, tuple[DatasetMeta, pd.DataFrame]] = {}
        self._checkpoint_resolver = None

    def set_checkpoint_resolver(self, resolver):
        """Inject a checkpoint resolver. Pass None to remove the guard."""
        self._checkpoint_resolver = resolver

    def load_preset_datasets(self) -> list[DatasetMeta]:
        if not self._data_dir.exists():
            return []
        metas = []
        for f in sorted(self._data_dir.glob("*.csv")):
            meta, df = self._load_csv(f, is_preset=True)
            self._datasets[meta.id] = (meta, df)
            metas.append(meta)
        return metas

    def _load_csv(
        self, path: Path, is_preset: bool = False, name: str | None = None,
        target_column: str | None = None,
    ) -> tuple[DatasetMeta, pd.DataFrame]:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        target_col = target_column if target_column else df.columns[-1]
        if target_col not in df.columns:
            raise ValueError(f"Column '{target_col}' not found in CSV")
        feature_names = [c for c in df.columns if c != target_col]
        plant_part = self._detect_plant_part(path.name)

        meta = DatasetMeta(
            id=path.stem,
            name=name or path.stem,
            filename=path.name,
            plant_part=plant_part,
            target=target_col,
            n_samples=len(df),
            n_features=len(feature_names),
            feature_names=feature_names,
            columns=list(df.columns),
            is_preset=is_preset,
        )
        return meta, df

    @staticmethod
    def _detect_plant_part(filename: str) -> str:
        if "leaf" in filename.lower():
            return "Leaf"
        return "Fruit"

    def list_datasets(self) -> list[DatasetMeta]:
        return [meta for meta, _ in self._datasets.values()]

    def get_dataset(self, dataset_id: str) -> DatasetDetail | None:
        if dataset_id not in self._datasets:
            return None
        meta, df = self._datasets[dataset_id]
        preview = df.head(10).round(4).to_dict(orient="records")
        return DatasetDetail(**meta.model_dump(), preview=preview)

    def get_stats(self, dataset_id: str) -> DatasetStatsResponse | None:
        if dataset_id not in self._datasets:
            return None
        _, df = self._datasets[dataset_id]
        stats = []
        for col in df.columns:
            series = df[col]
            stats.append(
                ColumnStats(
                    column=col,
                    mean=round(float(series.mean()), 4),
                    std=round(float(series.std(ddof=1)), 4),
                    min=round(float(series.min()), 4),
                    max=round(float(series.max()), 4),
                )
            )
        return DatasetStatsResponse(dataset_id=dataset_id, stats=stats)

    def get_dataframe(self, dataset_id: str) -> pd.DataFrame | None:
        if dataset_id not in self._datasets:
            return None
        return self._datasets[dataset_id][1].copy()

    def add_dataset(
        self, filename: str, content: bytes, name: str | None = None,
        target_column: str | None = None,
    ) -> DatasetMeta:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            f.write(content)
            temp_path = Path(f.name)
        try:
            meta, df = self._load_csv(temp_path, is_preset=False, name=name, target_column=target_column)
            stem = Path(filename).stem
            base_id = stem
            meta = meta.model_copy(update={"id": stem, "filename": filename})
            counter = 1
            while meta.id in self._datasets:
                meta = meta.model_copy(update={"id": f"{base_id}_{counter}"})
                counter += 1
            self._datasets[meta.id] = (meta, df)
            return meta
        finally:
            temp_path.unlink(missing_ok=True)

    def update_target(self, dataset_id: str, target_column: str) -> DatasetMeta:
        if dataset_id not in self._datasets:
            raise ValueError(f"Dataset '{dataset_id}' not found")
        if self._checkpoint_resolver is not None:
            checkpoints = self._checkpoint_resolver.list_checkpoints()
            if any(cp.dataset_id == dataset_id for cp in checkpoints):
                raise ValueError(
                    f"Cannot change target: dataset '{dataset_id}' has existing checkpoints"
                )
        meta, df = self._datasets[dataset_id]
        if target_column not in df.columns:
            raise ValueError(f"Column '{target_column}' not found in dataset")
        feature_names = [c for c in df.columns if c != target_column]
        updated_meta = meta.model_copy(update={
            "target": target_column,
            "feature_names": feature_names,
            "n_features": len(feature_names),
        })
        self._datasets[dataset_id] = (updated_meta, df)
        return updated_meta

    def delete_dataset(self, dataset_id: str) -> str:
        if dataset_id not in self._datasets:
            return "not_found"
        if self._datasets[dataset_id][0].is_preset:
            return "is_preset"
        del self._datasets[dataset_id]
        return "deleted"


data_loader = DataLoader()
