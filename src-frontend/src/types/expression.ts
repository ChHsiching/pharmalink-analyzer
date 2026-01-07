export interface ExpressionNode {
  type: "operator" | "function" | "variable" | "constant";
  value: string;
  children: ExpressionNode[];
}

export interface ExpressionResponse {
  expr_id: string;
  model_id: string;
  latex: string;
  complexity: number;
  r2_score: number;
  tree: ExpressionNode;
}

export interface ExpressionHistoryEntry {
  operation: string;
  latex: string;
  complexity: number;
  r2_score: number;
}

export interface ExpressionHistoryResponse {
  expr_id: string;
  history: ExpressionHistoryEntry[];
  current_index: number;
}
