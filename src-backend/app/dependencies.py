from fastapi import Depends

from app.services.analysis import AnalysisService
from app.services.evaluation import EvaluationService
from app.services.expression import ExpressionService
from app.services.training import TrainingService
from app.services.checkpoint_resolver import CheckpointResolver
from app.config import CHECKPOINT_DIR
from app.services.data_loader import data_loader as _default_data_loader


def get_checkpoint_resolver() -> CheckpointResolver:
    return CheckpointResolver(CHECKPOINT_DIR, _default_data_loader)


def get_analysis_service() -> AnalysisService:
    return AnalysisService(resolver=CheckpointResolver(CHECKPOINT_DIR, _default_data_loader))


_evaluation_service = EvaluationService(resolver=CheckpointResolver(CHECKPOINT_DIR, _default_data_loader))


def get_evaluation_service() -> EvaluationService:
    return _evaluation_service


_expression_service = ExpressionService(resolver=CheckpointResolver(CHECKPOINT_DIR, _default_data_loader))


def get_expression_service() -> ExpressionService:
    return _expression_service


_training_service = TrainingService()


def get_training_service() -> TrainingService:
    return _training_service


from app.services.formulation import FormulationService

_formulation_service = FormulationService(
    state_manager=_expression_service._state,
    resolver=_expression_service._resolver,
)


def get_formulation_service() -> FormulationService:
    return _formulation_service
