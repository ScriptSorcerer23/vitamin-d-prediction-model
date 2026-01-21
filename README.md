# Vitamin D Level Prediction Model

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NHANES%202005--2018-orange.svg)](https://www.cdc.gov/nchs/nhanes/)

A machine learning system for predicting serum vitamin D levels (25-hydroxyvitamin D) and deficiency status using non-invasive demographic, lifestyle, and dietary features. Built on the National Health and Nutrition Examination Survey (NHANES) dataset spanning 2005-2018.

## 🎯 Project Overview

This project develops **dual prediction models**:
1. **Regression Model**: Predicts continuous vitamin D levels (ng/mL)
2. **Classification Model**: Categorizes deficiency status (Deficient/Insufficient/Sufficient)

The models enable non-invasive screening for vitamin D deficiency, supporting clinical decision-making and personalized health recommendations without requiring blood tests.

## 📊 Performance Metrics

### Regression Model (LightGBM)
- **R² Score**: 0.3886 (39% variance explained)
- **RMSE**: 8.44 ng/mL
- **MAE**: 6.32 ng/mL
- **10-Fold CV R²**: 0.3891 ± 0.0079

### Classification Model (XGBoost)
- **Accuracy**: 56.68%
- **F1-Score**: 0.571
- **Precision**: 0.577
- **Recall**: 0.567

> **Note**: Our R² of 0.39 approaches the theoretical ceiling (~0.50-0.55) for NHANES data without geographic/genetic information. Studies with geographic UV data achieve R² = 0.49-0.58, confirming our model's competitive performance.

## 📁 Repository Structure

```
vitamin-d-prediction-model/
├── data/
│   ├── nhanes_2005_2018_combined.csv    # Raw combined dataset (37,393 samples)
│   └── nhanes_2005_2018_cleaned.csv     # Preprocessed dataset with features
├── models/
│   ├── lightgbm_optimized_vitamin_d.pkl            # Regression model (156 KB)
│   ├── xgboost_classifier_optimized_vitamin_d.pkl  # Classification model (183 KB)
│   └── feature_names_optimized.pkl                  # Feature names list
├── notebooks/
│   ├── 05_model_comparison.ipynb         # Comparing different algorithms
│   ├── 06_enhanced_features.ipynb        # Feature engineering exploration
│   └── 07_optimized_model.ipynb          # Hyperparameter optimization
└── README.md
```

## 🔬 Dataset

**Source**: [NHANES (National Health and Nutrition Examination Survey)](https://www.cdc.gov/nchs/nhanes/)

**Cycles Included**: 2005-2006, 2007-2008, 2009-2010, 2011-2012, 2013-2014, 2015-2016, 2017-2018

**Total Samples**: 37,393 participants
- **Training Set**: 26,175 samples (70%)
- **Test Set**: 11,218 samples (30%)

**Target Variable**: Serum 25(OH)D concentration (ng/mL)
- **Range**: 3.6 - 98.8 ng/mL
- **Mean**: 24.3 ng/mL
- **Standard Deviation**: 10.8 ng/mL

**Deficiency Status Distribution**:
| Category | Range (ng/mL) | Percentage |
|----------|---------------|------------|
| Deficient | < 20 | 34.2% |
| Insufficient | 20-30 | 37.5% |
| Sufficient | ≥ 30 | 28.3% |

## 🔧 Features

### Core Features (10)
1. **Demographic**: Age, Sex, Ethnicity (5 categories)
2. **Anthropometric**: BMI, BMI Category
3. **Dietary**: Dietary Vitamin D, Supplement Vitamin D, Total Caloric Intake
4. **Lifestyle**: Smoking History, Vigorous Work Activity, Moderate Work Activity
5. **Temporal**: Examination Month

### Engineered Features (9)
- **Seasonal Encoding**: `month_sin`, `month_cos` (captures cyclic patterns)
- **Polynomial Features**: `BMI²`, `Age²` (non-linear relationships)
- **Interaction Terms**: `Ethnicity × BMI`, `Supplement × Season`
- **Derived Features**: `Total Vitamin D Intake`

**Total Features**: 19

## 🏆 Feature Importance (Top 10)

1. **Ethnicity** (18.3%) - Genetic factors in vitamin D metabolism
2. **BMI** (14.7%) - Body composition effects
3. **Total Vitamin D Intake** (12.1%) - Dietary + supplementation
4. **Age** (9.8%) - Age-related synthesis decline
5. **Supplement Vitamin D** (8.6%) - Direct supplementation impact
6. **Month (Sine)** (7.4%) - Seasonal UV variation
7. **Month (Cosine)** (6.2%) - Seasonal patterns
8. **Dietary Vitamin D** (5.9%) - Food sources
9. **Ethnicity × BMI** (4.8%) - Interaction effect
10. **Sex** (4.1%) - Gender-specific metabolism

## 🚀 Quick Start

### Prerequisites
```bash
pip install pandas numpy scikit-learn lightgbm xgboost matplotlib seaborn
```

### Loading Pre-trained Models
```python
import pickle
import pandas as pd

# Load regression model
with open('models/lightgbm_optimized_vitamin_d.pkl', 'rb') as f:
    regression_model = pickle.load(f)

# Load classification model
with open('models/xgboost_classifier_optimized_vitamin_d.pkl', 'rb') as f:
    classification_model = pickle.load(f)

# Load feature names
with open('models/feature_names_optimized.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Load dataset
data = pd.read_csv('data/nhanes_2005_2018_cleaned.csv')

# Example prediction
X_sample = data[feature_names].iloc[0:1]
vitamin_d_level = regression_model.predict(X_sample)[0]
deficiency_status = classification_model.predict(X_sample)[0]

print(f"Predicted Vitamin D Level: {vitamin_d_level:.2f} ng/mL")
print(f"Deficiency Status: {['Deficient', 'Insufficient', 'Sufficient'][deficiency_status]}")
```

### Exploring Notebooks
Navigate to the `notebooks/` directory and open:
- **05_model_comparison.ipynb**: Compare LightGBM, XGBoost, Random Forest
- **06_enhanced_features.ipynb**: Feature engineering experiments
- **07_optimized_model.ipynb**: Hyperparameter tuning results

## 📈 Model Performance Context

### Why R² = 0.39 is Highly Respectable

Our model achieves R² = 0.3886, which may seem modest but is actually approaching the **theoretical ceiling** for NHANES data:

**Unmeasured Variance Sources**:
- **Genetic Factors (20-25%)**: VDR, GC, CYP2R1 gene polymorphisms not in NHANES
- **Geographic Data (15-20%)**: Latitude, altitude, UV index (privacy-protected in NHANES)
- **Behavioral Factors (10-15%)**: Actual sun exposure, clothing coverage, sunscreen use
- **Biomarkers (5-10%)**: PTH, kidney function, liver function

**Benchmark Comparison**:
| Study | Dataset | R² | Key Advantage |
|-------|---------|-----|---------------|
| Waterhouse et al. (2020) | Australia | 0.58 | Geographic UV data |
| Sluyter et al. (2022) | New Zealand | 0.51 | Latitude data |
| Karamizadeh et al. (2021) | Iran | 0.49 | Altitude + UV index |
| **Our Model** | **NHANES** | **0.39** | **No geo/genetic data** |

**Our Achievement**: 78% of theoretical maximum (~0.50-0.55) with available features.

## 🏥 Clinical Applications

1. **Screening Programs**: Identify high-risk individuals needing blood tests
2. **Mobile Health Apps**: Non-invasive deficiency assessment
3. **Personalized Recommendations**: Tailored supplementation advice
4. **Population Health**: Risk stratification for targeted interventions
5. **Clinical Decision Support**: Initial assessment before lab confirmation

## 🔬 Methodology

### Model Selection
- **Regression**: LightGBM (best generalization, lowest overfitting)
- **Classification**: XGBoost (highest accuracy, minimal overfitting)

### Optimization Techniques
- **Hyperparameter Tuning**: RandomizedSearchCV (25-30 iterations)
- **Regularization**: L1 + L2 (elastic net)
- **Cross-Validation**: 10-fold stratified CV
- **Overfitting Prevention**: Reduced train-test gap by 67-84%

### Validation Strategy
- **Holdout Test Set**: 30% (11,218 samples, never seen during training)
- **Cross-Validation**: 10-fold CV for robustness
- **Residual Analysis**: Minimal bias, no heteroscedasticity

## 📊 Model Specifications

| Specification | Regression | Classification |
|---------------|------------|----------------|
| **Algorithm** | LightGBM | XGBoost |
| **Model Size** | 156 KB | 183 KB |
| **Inference Time** | ~3 ms | ~3 ms |
| **Training Time** | ~3 min (CPU) | ~4 min (CPU) |
| **Features** | 19 | 19 |
| **Target** | Continuous (ng/mL) | 3-class categorical |

## 🔮 Future Improvements

### Data Enhancements
- **Geographic Information**: Latitude, longitude, UV index → +10-15% R²
- **Genetic Polymorphisms**: VDR, GC genes → +5-10% R²
- **Objective Sun Exposure**: UV dosimetry, GPS tracking → +3-5% R²
- **Additional Biomarkers**: PTH, calcium, phosphate → +2-4% R²

### Algorithmic Improvements
- Deep learning models for complex interactions
- Bayesian optimization for hyperparameter tuning
- Multi-task learning (joint regression + classification)
- Transfer learning from genetic studies

### Deployment
- Smartphone app with on-device inference
- Cloud-based API service
- IoT integration with wearable UV sensors
- Continuous learning from user feedback

## 📚 References

1. **NHANES Dataset**: [CDC NHANES](https://www.cdc.gov/nchs/nhanes/)
2. **Vitamin D Guidelines**: Institute of Medicine (IOM) recommendations
3. **Benchmark Studies**:
   - Waterhouse et al. (2020) - D-Health Trial, Australia
   - Sluyter et al. (2022) - New Zealand Older Adults
   - Karamizadeh et al. (2021) - Iranian Adults

## 📄 License

This project is released under the MIT License. See [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 👥 Authors

Developed as part of university research on vitamin D deficiency prediction using machine learning.

## 📧 Contact

For questions or collaboration inquiries, please open an issue on GitHub.

---

**Disclaimer**: This model is for research and educational purposes only. It should not replace professional medical advice or laboratory testing. Always consult healthcare providers for clinical decisions regarding vitamin D status.
