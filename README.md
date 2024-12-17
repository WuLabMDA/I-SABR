# I-SABR-SELECT: A Computational Tool for Personalized Treatment Decisions in SABR and I-SABR

**I-SABR-SELECT** is a computational framework that estimates the individualized treatment effect (ITE) for adding immunotherapy to stereotactic ablative radiotherapy (SABR) in patients with early-stage inoperable non-small cell lung cancer (NSCLC). By leveraging radiomic features extracted from contrast-enhanced CT imaging and clinical predictors, I-SABR-SELECT models heterogeneous treatment effects between I-SABR and SABR alone using counterfactual reasoning and machine learning techniques. The framework identifies patients who are most likely to benefit from immunotherapy while minimizing unnecessary treatment exposure for others.

## Key Features
- **Radiomics and Clinical Integration**: Combines 43 radiomic and 8 clinical predictors for precise patient stratification.
- **Counterfactual Reasoning**: Models treatment effects between SABR and I-SABR using individualized treatment effect (ITE) scores.
- **Robust Feature Selection**: Implements swarm intelligence (grey wolf optimizer) and cross-validation to identify predictive features.
- **External Validation**: Validated using data from the I-SABR randomized trial and STARS trial for generalizability.
- **Interpretability**: SHAP-based analysis quantifies the influence of clinical and radiomic predictors.

This repository holds the code for the I-SABR-SELECT framework, as described in [XXXXX]. 

### 1. Preprocessing
Scripts for:
- CT image harmonization (voxel resampling, normalization)
- Region of Interest (ROI) segmentation for tumors, peritumoral regions, lung parenchyma, and blood vessels
- Radiomic feature extraction and qualification using PyRadiomics and in-house MATLAB codes

### 2. Modeling
Codes for:
- Swarm intelligence-based feature selection
- Counterfactual reasoning and individualized treatment effect (ITE) estimation
- Model training and bootstrapping-based cross-validation

### 3. Validation
Scripts for:
- Internal validation on the I-SABR trial cohort
- External validation using the STARS trial cohort
- Kaplan-Meier survival analysis, hazard ratios, and event-free survival (EFS) metrics

### 4. Interpretation
- SHAP-based feature importance analysis for model explainability
- Investigation of individual predictor contributions to treatment recommendations

## Results
- **Treatment Recommendations**: I-SABR-SELECT identified a significant subgroup of patients benefiting from adding immunotherapy.
- **Improved Outcomes**: Patients treated following model recommendations demonstrated superior event-free survival (EFS) compared to random treatment assignment.

## Citation
If you use this framework, please cite our work:

```bibtex
@article{ISABRSelect,
  title={Artificial Intelligence-Based Clinical and Radiomic Analysis to Optimize Patient Selection for Combined Immunotherapy and SABR in Early-Stage NSCLC – Secondary Analysis of the I-SABR Randomized Controlled Trial},
  author={},
  journal={},
  year={Year},
  volume={Volume},
  pages={Pages},
  doi={DOI}
}
```

For questions, contributions, or issues, please contact us or create a new issue in this repository.
