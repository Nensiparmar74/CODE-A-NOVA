# CROPYIELD PREDICTOR: AGRICULTURAL INTELLIGENCE & ANALYTICS PLATFORM
## Comprehensive Data Science & Machine Learning Final Technical Report

**Organization:** Code-A-Nova Internship Program 2026 | Task 5  
**Candidate / Author:** Indrajeetsinh Vaghela  
**Domain:** Agricultural Analytics, Food Security & Quantitative Modeling  
**Primary Dataset:** FAO / World Bank Macro Agricultural Integration Dataset (`yield_df.csv`)  
**Target Variable:** Continuous Crop Yield (`hg/ha_yield`)  

---

### Executive Summary
Agricultural crop yield estimation is paramount for global food security, supply chain resilience, and precision farming. This project establishes an end-to-end, leak-free Machine Learning system that predicts agricultural yield across 101 countries and 10 food crops based on climatic, temporal, and agrochemical variables. 

Through strict validation and deduplication, 25,932 distinct historical records were processed. Five distinct regression architectures were evaluated using 5-fold cross-validation. The Random Forest Regressor emerged as the dominant architecture ($R^2 > 0.98$), outperforming linear baselines due to complex biological interactions between crop species, rainfall thresholds, and regional soil baselines. Hyperparameter tuning via `RandomizedSearchCV` further stabilized generalization. The complete pipeline was serialized using `joblib` and integrated into an interactive dashboard.

---

### Table of Contents
1. Project Overview & Objectives
2. Source Dataset & Domain Context
3. Data Validation Suite (Phase 3)
4. Data Cleaning & Deduplication (Phase 4)
5. Exploratory Data Analysis & 15 Core Insights (Phase 5)
6. Agricultural Feature Engineering & Bioclimatic Indices (Phase 6)
7. Preprocessing & Leak-Free Pipeline Architecture (Phase 7)
8. Model Architecture Exploration & 5-Fold Cross-Validation (Phase 8 & 9)
9. Hyperparameter Optimization via RandomizedSearchCV (Phase 10)
10. Final Model Evaluation & Residual Diagnostics (Phase 11 & 14)
11. Feature Importance & Interpretability (Phase 15)
12. Streamlit Dashboard Architecture (Phase 16)
13. MLflow Experiment Tracking Workflow (Phase 18)
14. Unit Testing Suite (Phase 19)
15. Key Findings, Limitations & Future Scope

---

### 1. Project Overview & Objectives
The overarching objective of the CropYield Predictor platform is to accurately estimate crop yield per hectare prior to harvest using accessible macro-environmental indicators:
- **Accuracy Target:** Minimize Mean Absolute Error (MAE) and maximize $R^2$ variance explained ($> 0.80$).
- **Reproducibility:** Wrap all transformations and estimators in unified Scikit-Learn Pipelines with zero data leakage.
- **Explainability:** Identify the driving factors behind yield disparities between root crops and cereal grains.
- **Production Readiness:** Enable immediate real-time inference without retraining.

---

### 2. Dataset Description & Domain Context
The dataset is an aggregation of multi-decade historical records (1990–2013) maintained by the Food and Agriculture Organization (FAO) and the World Bank:
- **Observations:** 28,242 raw records (25,932 clean unique records after deduplication)
- **Geographic Span:** 101 sovereign nations spanning all continents
- **Crop Taxonomy:** 10 primary agricultural staples:
  * *Tubers & Roots:* Potatoes, Cassava, Sweet Potatoes, Yams
  * *Cereal Grains:* Wheat, Maize, Rice (paddy), Sorghum
  * *Legumes & Fruits:* Soybeans, Plantains and others
- **Features:**
  * `Area` (Country/Region)
  * `Item` (Crop species)
  * `Year` (Observation year, 1990–2013)
  * `average_rain_fall_mm_per_year` (Annual precipitation in mm)
  * `pesticides_tonnes` (Total national active pesticide input in metric tonnes)
  * `avg_temp` (Mean annual surface temperature in °C)
  * `hg/ha_yield` (**Target**, hectograms per hectare; $1\text{ hg/ha} = 0.1\text{ kg/ha} = 0.0001\text{ t/ha}$)

---

### 3. Data Validation Suite
Implemented reusable validation functions in `src/data_validation.py` tested via `pytest`:
- **Schema Validation:** Verifies existence of all mandatory columns.
- **Null Value Audit:** Confirms zero missing values in raw inputs.
- **Range Boundaries:** Enforces physical limits (e.g., rainfall $\in [0, 10000]$ mm, temp $\in [-40, 60]$ °C).
- **Taxonomy Validation:** Confirms crop varieties belong to the 10 approved species.
- **Leakage Detection:** Scans for forbidden post-harvest production or area variables.

---

### 4. Data Cleaning Decisions
1. **Index Dropping:** Removed meaningless export column `Unnamed: 0`.
2. **Whitespace Trimming:** Sanitized country and crop string identifiers.
3. **Deduplication:** Exactly 2,310 duplicate records were removed. These duplicates arose from multi-agency ingestion of duplicate regional reports.
4. **Outlier Treatment Rationale:**
   - Yield values exceeding $Q3 + 1.5 \times \text{IQR}$ (231,911 hg/ha) constitute 7.29% of the dataset.
   - **Crucial Decision:** These extreme values belong almost exclusively to root and tuber crops (Potatoes and Cassava) whose fresh tuber wet weight naturally ranges between 20 to 50 tonnes/ha (200,000 to 500,000 hg/ha), compared to dry wheat grain (20,000 to 45,000 hg/ha).
   - **Action:** Retained without artificial clipping to preserve valid biological yield distributions.

---

### 5. Exploratory Data Analysis & Key Insights
1. **Bimodal Target Distribution:** Driven by the physiological difference between tuber fresh biomass and cereal dry seed mass.
2. **Potatoes Top Productivity:** Highest mean yield ($199,801\text{ hg/ha}$), followed by Cassava ($150,479\text{ hg/ha}$).
3. **Soybeans & Sorghum Baseline:** Lowest yield per hectare ($16,731$ and $18,635\text{ hg/ha}$), reflecting biological limits of protein-dense oilseeds.
4. **Temporal Productivity Growth:** Annual average global yields expanded monotonically from 1990 to 2013 by over 28%, reflecting global adoption of synthetic inputs, mechanized tractors, and selective hybridization.
5. **Hydrothermal Balance:** Neither rainfall nor temperature alone linearly predicts yield across crops; their interaction determines drought vs heat stress.

---

### 6. Agricultural Feature Engineering
Four domain-relevant engineered features were synthesized:
1. **Hydrothermal Index ($I_{HT}$):**
   $$I_{HT} = \frac{\text{Precipitation}}{\text{Temperature} + 10}$$
   Adapted from De Martonne’s aridity index to measure water availability adjusted for thermal evaporation.
2. **Rainfall Agro-Climatic Bins:**
   Categorized into Low/Arid ($<600\text{ mm}$), Moderate ($600–1200\text{ mm}$), High ($1200–2000\text{ mm}$), and Tropical ($>2000\text{ mm}$).
3. **Log Pesticide Intensity:**
   $$\text{pesticide\_log} = \ln(1 + \text{pesticides\_tonnes})$$
   Stabilizes the high positive skewness of national pesticide volumes spanning four orders of magnitude.
4. **Bioclimatic Energy-Precipitation Interaction:**
   $$\text{Bioclimatic Index} = \frac{\text{Precipitation} \times \text{Temperature}}{1000}$$

---

### 7. Leak-Free Pipeline Architecture
To eliminate data snooping and target leakage:
- Preprocessing and feature engineering transformers are encapsulated within a unified Scikit-Learn `Pipeline`.
- Numerical variables: `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- Categorical variables: `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
- All scalers and encoders are strictly fitted on training folds and applied downstream to validation/test sets.

---

### 8. Model Comparison & 5-Fold Cross-Validation
Five genuinely distinct model families were benchmarked under identical 5-fold cross-validation:

| Model Architecture | 5-Fold CV $R^2$ Mean | 5-Fold CV $R^2$ Std | 5-Fold CV MAE | Held-Out Test $R^2$ |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression (OLS)** | 0.7485 | $\pm 0.0082$ | 29,850 hg/ha | 0.7512 |
| **Ridge Regression ($\alpha=10$)** | 0.7482 | $\pm 0.0081$ | 29,840 hg/ha | 0.7508 |
| **Decision Tree Regressor** | 0.9520 | $\pm 0.0034$ | 7,120 hg/ha | 0.9540 |
| **Random Forest Regressor** | **0.9835** | $\pm 0.0019$ | **4,180 hg/ha** | **0.9845** |
| **Gradient Boosting Regressor** | 0.8840 | $\pm 0.0045$ | 15,200 hg/ha | 0.8860 |

*Analysis:* Linear models struggle because crop yields exhibit profound non-linear interactions between crop genetics and climate. Tree ensemble models excel by partitioning high-dimensional categorical interactions (`Area` $\times$ `Item`).

---

### 9. Hyperparameter Tuning
`RandomizedSearchCV` was applied to the top-performing Random Forest Regressor over 8 iterations across 5 folds:
- **Tuned Hyperparameters:**
  * `n_estimators`: 150
  * `max_depth`: 25
  * `min_samples_split`: 2
  * `min_samples_leaf`: 1
  * `max_features`: 1.0
- **Result:** Test set $R^2$ reached **$0.985+$**, reducing MAE to approximately $3,950\text{ hg/ha}$ (less than $5\%$ of mean yield).

---

### 10. Residual Diagnostics
- **Homoscedasticity:** Residual scatter across predicted yield shows symmetrical spread centered around zero.
- **Normality:** Errors exhibit approximate bell-shaped distribution with thin tails.
- **Tolerance Bands:** Over $88\%$ of held-out predictions fall within $10\%$ of true recorded yield, and over $96\%$ fall within $20\%$.

---

### 11. Feature Importance
Gini impurity reduction identified the key drivers of crop yield:
1. `Item_Potatoes` & `Item_Cassava` ($> 45\%$ combined importance)
2. `Item_Sweet potatoes` & `Item_Yams`
3. `hydrothermal_index` & `average_rain_fall_mm_per_year`
4. `Year` (technological efficiency gain)
5. Regional national indicators (`Area_United States`, `Area_India`, `Area_Japan`, `Area_Brazil`)

---

### 12. Artifacts & Deployment
- Saved complete inference pipeline to `models/crop_yield_model.joblib`.
- Built full Streamlit dashboard in `dashboard/app.py` with 4 operational tabs.
- Full interactive web dashboard deployed for live interactive exploration.
