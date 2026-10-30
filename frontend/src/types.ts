export type RiskTier = 'low' | 'medium' | 'high'

export interface FeatureSpec {
  name: string
  kind: 'numeric' | 'categorical'
  label: string
  group: string
  description: string
  allowed_values: string[]
  minimum: number | null
  maximum: number | null
  unit: string | null
  nullable: boolean
  integer: boolean
}

export interface ExcludedFeature {
  name: string
  reason: string
}

export interface TierCutoffs {
  medium: number
  high: number
  medium_recall_target: number
}

export interface TierContext {
  tier: RiskTier
  customers: number
  share_of_customers: number
  observed_churn_rate: number
  share_of_churners: number
}

export interface RiskFactor {
  feature: string
  label: string
  customer_value: string
  reference_value: string
  probability_if_reference: number
  effect: number
  attribution_score?: number | null
  direction?: 'risk_increasing' | 'risk_reducing' | null
}

export interface LocalAttribution {
  top_risk_increasing: RiskFactor[]
  top_risk_reducing: RiskFactor[]
}

export interface PredictionResult {
  customer_id: string | null
  churn_probability: number
  expected_revenue_at_risk: number
  risk_tier: RiskTier
  risk_tier_label: string
  flagged: boolean
  decision_threshold: number
  tier_cutoffs: TierCutoffs
  tier_context: TierContext | null
  factors: RiskFactor[]
  local_attribution: LocalAttribution
  imputed_fields: string[]
  warnings: string[]
  limitations: string[]
  model_version: string
}

export interface BatchRow {
  row: number
  customer_id: string | null
  status: 'scored' | 'invalid'
  churn_probability: number | null
  expected_revenue_at_risk?: number | null
  risk_tier: RiskTier | null
  flagged: boolean
  issues: string
  imputed_fields: string[]
}

export interface HistogramBin {
  lower: number
  upper: number
  count: number
}

export interface BatchSummary {
  input_rows: number
  scored_rows: number
  invalid_rows: number
  flagged_rows: number
  mean_probability: number | null
  predicted_churn_rate: number
  baseline_churn_rate: number
  expected_churners: number
  total_arr_at_risk: number
  tier_counts: Record<RiskTier, number>
  histogram: HistogramBin[]
}

export interface BatchPredictionResult {
  model_version: string
  decision_threshold: number
  tier_cutoffs: TierCutoffs
  summary: BatchSummary
  rows: BatchRow[]
}

export interface ModelHeadlineMetrics {
  roc_auc: number
  pr_auc: number
  brier: number
  precision: number
  recall: number
  f1: number
  decision_threshold: number
  evaluated_on: string
}

export interface ModelInfo {
  model_version: string
  created_at: string
  model: {
    name: string
    label: string
    role: string
    complexity: number
    eligible_for_selection: boolean
    description: string
  }
  selection: {
    metric: string
    best_candidate: string
    best_cv_mean: number
    best_cv_standard_error: number
    acceptance_floor: number
    candidates_within_one_se: string[]
    selected: string
    explanation: string
  }
  features: FeatureSpec[]
  excluded_features: ExcludedFeature[]
  training_config: {
    seed: number
    test_size: number
    cv_folds: number
    medium_recall_target: number
    selection_metric: string
    selection_rule: string
  }
  dataset: {
    provenance: {
      name: string
      source_url: string | null
      repository: string | null
      license: string | null
      sha256: string
      rows: number
      retrieved_at: string
      synthetic: boolean
      notes: string
    }
    quality: {
      input_rows: number
      invalid_rows_dropped: number
      duplicate_ids_dropped: number
      modelled_rows: number
      churned: number
      retained: number
      churn_rate: number
      missing_values_imputed: Record<string, number>
      validation_issue_examples: Array<{ row: number; column: string; message: string }>
    }
    split: {
      strategy: string
      train_rows: number
      test_rows: number
      train_churn_rate: number
      test_churn_rate: number
    }
  }
  risk_tiers: {
    cutoffs: TierCutoffs
    derivation: string
    holdout_summary: TierContext[]
  }
  headline_metrics: ModelHeadlineMetrics
  reference_profile: Record<string, string | number>
  environment: Record<string, string>
  warnings: string[]
  limitations: string[]
}

export interface BaselineComparisonEntry {
  name: string
  label: string
  roc_auc: number
  pr_auc: number
  f1: number
}

export interface BaselineComparison {
  baseline: BaselineComparisonEntry
  production: BaselineComparisonEntry
  lift: {
    roc_auc_delta: number
    pr_auc_delta: number
    f1_delta: number
  }
}

export interface ModelEvaluation {
  model_version: string
  synthetic_data: boolean
  candidates: Array<{
    name: string
    label: string
    role: string
    complexity: number
    eligible_for_selection: boolean
    cv: Record<string, { mean: number; std: number; folds: number[] }>
    test: Record<string, number>
    description: string
  }>
  baseline_comparison?: BaselineComparison
  selection: {
    metric: string
    best_candidate: string
    best_cv_mean: number
    best_cv_standard_error: number
    acceptance_floor: number
    candidates_within_one_se: string[]
    selected: string
    explanation: string
  }
  holdout: {
    rows: number
    positives: number
    ranking: {
      roc_auc: number
      pr_auc: number
      f1?: number
      brier: number
      log_loss: number
      base_rate: number
    }
    at_decision_threshold: {
      threshold: number
      precision: number
      recall: number
      f1: number
      specificity: number
      accuracy: number
      flagged_rate: number
      confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
    }
    at_default_threshold: {
      threshold: number
      precision: number
      recall: number
      f1: number
      specificity: number
      accuracy: number
      flagged_rate: number
      confusion_matrix: { tn: number; fp: number; fn: number; tp: number }
    }
    confidence_intervals: Record<string, { lower: number; upper: number; samples: number }>
    curves: {
      roc: Array<{ fpr: number; tpr: number }>
      precision_recall: Array<{ recall: number; precision: number }>
      threshold_sweep: Array<{
        threshold: number
        precision: number
        recall: number
        f1: number
        flagged_rate: number
      }>
      calibration: Array<{ predicted: number; observed: number; count: number }>
      cumulative_gains: Array<{
        population_share: number
        churners_captured: number
        lift: number
      }>
    }
    score_histogram: Array<{ lower: number; upper: number; retained: number; churned: number }>
    tiers: TierContext[]
  }
  permutation_importance: {
    scoring: string
    repeats: number
    evaluated_on: string
    features: Array<{
      feature: string
      label: string
      importance_mean: number
      importance_std: number
    }>
  }
}

export interface HealthStatus {
  status: 'ok' | 'degraded'
  model_loaded: boolean
  model_version: string | null
  reason?: string | null
  detail?: string | null
  warnings: string[]
}
