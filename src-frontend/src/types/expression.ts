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
  pareto_count: number;
  pareto_index: number;
  variable_impact: Record<string, number>;
  indicators: Record<string, number>;
  target_name: string;
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

export interface TaskStatusResponse {
  task_id: string;
  status: "pending" | "running" | "completed" | "failed";
  result: ExpressionResponse | null;
  error: string | null;
}
