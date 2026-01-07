from app.models.expression import (
    ExpressionNode,
    ExpressionResponse,
    ExpressionHistoryEntry,
    ExpressionHistoryResponse,
)


def test_expression_node():
    node = ExpressionNode(type="operator", value="+", children=[
        ExpressionNode(type="variable", value="x0", children=[]),
        ExpressionNode(type="constant", value="1.5", children=[]),
    ])
    assert node.type == "operator"
    assert len(node.children) == 2


def test_expression_response():
    resp = ExpressionResponse(
        expr_id="expr1",
        model_id="model1",
        latex="x_{0} + 1.5",
        complexity=3,
        r2_score=0.95,
        tree=ExpressionNode(type="variable", value="x0", children=[]),
    )
    assert resp.expr_id == "expr1"


def test_expression_history_entry():
    entry = ExpressionHistoryEntry(operation="simplify", latex="x_{0}", complexity=1, r2_score=0.94)
    assert entry.operation == "simplify"


def test_expression_history_response():
    resp = ExpressionHistoryResponse(
        expr_id="expr1",
        history=[ExpressionHistoryEntry(operation="generate", latex="x_{0}", complexity=1, r2_score=0.95)],
        current_index=0,
    )
    assert len(resp.history) == 1
