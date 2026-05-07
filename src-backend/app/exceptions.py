class DomainError(Exception):
    pass


class CheckpointNotFoundError(DomainError):
    def __init__(self, model_id: str):
        super().__init__(f"Checkpoint not found: {model_id}")
        self.model_id = model_id


class DatasetNotFoundError(DomainError):
    def __init__(self, dataset_id: str):
        super().__init__(f"Dataset not found: {dataset_id}")
        self.dataset_id = dataset_id


class ExpressionNotFoundError(DomainError):
    def __init__(self, expr_id: str):
        super().__init__(f"Expression not found: {expr_id}")
        self.expr_id = expr_id


class UndoLimitError(DomainError):
    def __init__(self):
        super().__init__("No more history to undo")
