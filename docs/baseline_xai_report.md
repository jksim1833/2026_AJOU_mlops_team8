# Baseline and SHAP XAI Report


This document is the report target for the modeling task: baseline comparison
and SHAP-based XAI for the diecasting normal/defect binary classifier.

## Current Repository Baseline

The committed RandomForest serving artifact reports:

| Split | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.781 | 0.427 | 0.712 | 0.534 | 0.828 |
| Test | 0.796 | 0.444 | 0.649 | 0.527 | 0.839 |

Accuracy is not the primary metric because normal samples are the majority
class. F1, recall, precision, and ROC-AUC are reported together to evaluate
defect detection quality.

## Reproducible Baseline/XAI Task Command

After the raw CSV is available at
`data/raw/DieCasting_Quality_Raw_Data_product1.csv`, run:

```bash
python -m src.data.prepare_data
python -m src.models.compare_baselines_xai
```

The comparison script trains Logistic Regression, Decision Tree,
RandomForest, XGBoost, and LightGBM baselines when the optional packages are
installed. It ranks models by validation F1 and uses validation ROC-AUC as the
tie-breaker.

## Expected Deliverables

- `artifacts/reports/baseline_metric_table.md`
- `artifacts/reports/baseline_comparison.csv`
- `artifacts/reports/candidate_handoff_note.md`
- `artifacts/reports/xai_feature_interpretation.md`
- `artifacts/reports/shap_local_explanation.md`
- `artifacts/plots/baseline_validation_metrics.png`
- `artifacts/plots/baseline_confusion_matrices.png`
- `artifacts/plots/shap_summary_bar.png`
- `artifacts/plots/shap_beeswarm.png`
- `artifacts/plots/shap_waterfall_defect_sample.png`
- `artifacts/models/baseline_candidate.joblib`
- `artifacts/models/baseline_candidate_metadata.json`

## Presentation Talking Points

- The label is binary: normal is 0 and defect is 1.
- Defect columns are removed from features to prevent label leakage.
- Baselines are compared on the same stratified train/validation/test split.
- Positive SHAP values push the prediction toward defect; negative values push
  it toward normal.
- The selected baseline candidate should be handed off for threshold or
  hyperparameter tuning before final serving.
