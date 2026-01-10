import pytest

from app.exceptions import (
    CheckpointNotFoundError,
    DatasetNotFoundError,
    DomainError,
    ExpressionNotFoundError,
    UndoLimitError,
)


class TestDomainError:
    def test_is_base_exception(self):
        assert issubclass(DomainError, Exception)

    def test_carries_message(self):
        err = DomainError("test message")
        assert str(err) == "test message"


class TestCheckpointNotFoundError:
    def test_is_domain_error(self):
        assert issubclass(CheckpointNotFoundError, DomainError)

    def test_carries_model_id(self):
        err = CheckpointNotFoundError("model-123")
        assert err.model_id == "model-123"
        assert "model-123" in str(err)

    def test_raises(self):
        with pytest.raises(CheckpointNotFoundError):
            raise CheckpointNotFoundError("abc")


class TestDatasetNotFoundError:
    def test_is_domain_error(self):
        assert issubclass(DatasetNotFoundError, DomainError)

    def test_carries_dataset_id(self):
        err = DatasetNotFoundError("ds-1")
        assert err.dataset_id == "ds-1"
        assert "ds-1" in str(err)


class TestExpressionNotFoundError:
    def test_is_domain_error(self):
        assert issubclass(ExpressionNotFoundError, DomainError)

    def test_carries_expr_id(self):
        err = ExpressionNotFoundError("expr_abc")
        assert err.expr_id == "expr_abc"
        assert "expr_abc" in str(err)


class TestUndoLimitError:
    def test_is_domain_error(self):
        assert issubclass(UndoLimitError, DomainError)

    def test_default_message(self):
        err = UndoLimitError()
        assert "undo" in str(err).lower() or "history" in str(err).lower()
