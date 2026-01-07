from app.models.evaluation import (
    FoldMetrics,
    AggregateMetrics,
    EvaluationMetricsResponse,
    PredictionPoint,
    PredictionsResponse,
    ResidualBin,
    ResidualsResponse,
    LossCurvePoint,
    FoldLossCurve,
    LossCurveResponse,
)


def test_fold_metrics():
    m = FoldMetrics(fold=1, r2=0.85, mse=0.12, mae=0.28)
    assert m.fold == 1
    assert m.r2 == 0.85


def test_aggregate_metrics():
    a = AggregateMetrics(
        r2_mean=0.82, r2_std=0.05,
        mse_mean=0.15, mse_std=0.03,
        mae_mean=0.30, mae_std=0.04,
    )
    assert a.r2_mean == 0.82


def test_evaluation_metrics_response():
    r = EvaluationMetricsResponse(
        model_id="test_model",
        folds=[FoldMetrics(fold=1, r2=0.9, mse=0.1, mae=0.2)],
        aggregate=AggregateMetrics(
            r2_mean=0.9, r2_std=0.0,
            mse_mean=0.1, mse_std=0.0,
            mae_mean=0.2, mae_std=0.0,
        ),
    )
    assert len(r.folds) == 1
    assert r.aggregate.r2_mean == 0.9


def test_predictions_response():
    r = PredictionsResponse(
        model_id="m1",
        predictions=[
            PredictionPoint(actual=1.0, predicted=1.1),
            PredictionPoint(actual=2.0, predicted=1.9),
        ],
    )
    assert len(r.predictions) == 2
    assert r.predictions[0].actual == 1.0


def test_residuals_response():
    r = ResidualsResponse(
        model_id="m1",
        residuals=[0.1, -0.1, 0.05],
        bins=[ResidualBin(range_start=-0.1, range_end=0.0, count=1),
              ResidualBin(range_start=0.0, range_end=0.1, count=2)],
        mean=0.017,
        std=0.094,
    )
    assert len(r.residuals) == 3
    assert r.bins[0].count == 1


def test_loss_curve_response():
    r = LossCurveResponse(
        model_id="m1",
        folds=[
            FoldLossCurve(fold=1, points=[
                LossCurvePoint(epoch=1, train_loss=0.5, val_loss=0.6),
                LossCurvePoint(epoch=2, train_loss=0.3, val_loss=0.4),
            ]),
        ],
    )
    assert len(r.folds) == 1
    assert r.folds[0].points[0].train_loss == 0.5
