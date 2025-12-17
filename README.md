### 🎯 [**LIVE DEMO**](https://observatorio-salud-mental-bogota-2ynkrapsjostnmrxfcxbxz.streamlit.app/) | 📊 [**OPEN DATA**](https://herramientas.datos.gov.co/usos/observatorio-de-salud-mental-escolar-bogota)

*AI-powered system for predictive analysis of mental health in school-age population*

**Team SENSORY** | 🏆 **5th Place National - Data to Ecosystem 2025**

</div>

---

## 📋 Table of Contents

- [Executive Summary](#-executive-summary)
- [System Architecture](#-system-architecture)
- [Technology Stack](#-technology-stack)
- [Datasets and Sources](#-datasets-and-sources)
- [Machine Learning Models](#-machine-learning-models)
- [Feature Engineering](#-feature-engineering)
- [Dashboard Modules](#-dashboard-modules)
- [Installation and Deployment](#-installation-and-deployment)
- [Results and Metrics](#-results-and-metrics)
- [Recognition: Data to Ecosystem 2025](#-recognition-data-to-ecosystem-2025)
- [Team](#-team)
- [License](#-license)

---

## 🎯 Executive Summary

The **Bogotá School Mental Health Observatory** is an artificial intelligence platform that integrates public health open data to provide predictive analysis and risk classification in mental health for school-age population (6-17 years). The system reduces public policy report generation time from 21 days to 5 minutes through automation of complex analysis and predictive modeling through 2030.

### Problem Statement

- **44.7%** of children and adolescents in Colombia show signs of mental health impairment (UNICEF 2024)
- **230 minor suicides** in 2023, **140** in Q1 2024 (Legal Medicine Institute)
- Current ratio: **1 counselor per 500 students** (insufficient for effective care)
- Average manual analysis time: **21 days** per report
- Scattered data across multiple sources without integration

### Technical Solution

End-to-end system that:
1. Automatically integrates 4 open data sources
2. Applies 3 ML/DL models for classification and prediction
3. Generates interactive visualizations in real-time
4. Provides projections through 2030 with confidence intervals
5. Classifies localities by risk level with 87% accuracy

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    DATA LAYER                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Morbidity   │  │  Enrollment  │  │Gender Parity │      │
│  │  59,657 rec  │  │ 4M students  │  │    Index     │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┴──────────────────┘              │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PROCESSING LAYER                               │
│  ┌────────────────────────────────────────────────┐         │
│  │  ETL Pipeline (Pandas)                         │         │
│  │  • Data cleaning and validation                │         │
│  │  • Code normalization                          │         │
│  │  • Multi-source integration                    │         │
│  │  • Feature engineering (70 variables)          │         │
│  └────────────────────┬───────────────────────────┘         │
└────────────────────────┼────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 MODELING LAYER                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Random Forest │  │Deep Neural   │  │   K-Means    │      │
│  │Classification│  │  Network     │  │  Clustering  │      │
│  │  87% acc     │  │  RMSE: 156   │  │  3 clusters  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┴──────────────────┘              │
└───────────────────────────┼──────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              PRESENTATION LAYER                             │
│  ┌────────────────────────────────────────────────┐         │
│  │  Streamlit Dashboard (8 modules)              │         │
│  │  • Plotly visualizations                       │         │
│  │  • Real-time interactivity                     │         │
│  │  • Report export                               │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

### Languages and Frameworks

```python
Python 3.10.12
├── Data Processing
│   ├── pandas==2.1.4
│   ├── numpy==1.24.3
│   └── openpyxl==3.1.2
│
├── Machine Learning
│   ├── scikit-learn==1.3.2
│   ├── tensorflow==2.15.0
│   └── keras==2.15.0
│
├── Visualization
│   ├── plotly==5.18.0
│   ├── matplotlib==3.8.2
│   └── seaborn==0.13.0
│
├── Dashboard
│   ├── streamlit==1.30.0
│   └── streamlit-folium==0.15.1
│
└── Utilities
    ├── json==built-in
    └── datetime==built-in
```

### Infrastructure

- **Hosting**: Streamlit Community Cloud
- **Version Control**: GitHub
- **CI/CD**: Automatic deployment on push to main
- **Storage**: CSV files (optimized for speed)

---

## 📊 Datasets and Sources

### 1. Mental Health Morbidity

**Source**: District Health Secretariat of Bogotá  
**Portal**: [Colombia Open Data](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Morbilidad-en-Salud-Mental/iib8-v6ks)

**Characteristics**:
- **Records**: 59,657 medical visits
- **Period**: 2019-2024
- **Population**: 6-17 years (school age)
- **Key variables**: 
  - Year of care
  - Provider locality
  - ICD-10 diagnosis
  - Patient gender
  - Average age
  - Type of care

**Processing**:
```python
# Target population filtering
df = df[(df['edad_min'] >= 6) & (df['edad_max'] <= 17)]

# Gender normalization
df['genero'] = df['sexo_gen'].map({
    'M': 'Masculino', 'H': 'Masculino',
    'F': 'Femenino', 'MUJER': 'Femenino'
})

# Disorder categorization
df['categoria_trastorno'] = df['dxprincipal_agrupacion1_nombre'].apply(
    categorizar_trastorno
)
```

### 2. Official Enrollment

**Source**: Ministry of National Education  
**Portal**: [Colombia Open Data](https://www.datos.gov.co/Educaci-n/MEN_MATRICULA_POR_GRADO/nudc-7mev)

**Characteristics**:
- **Records**: 4,479,813 students
- **Period**: 2019-2024
- **Disaggregation**: By locality and gender
- **Key variables**:
  - Total enrollment
  - Male enrollment
  - Female enrollment
  - Distribution by educational level

### 3. Gender Parity Index

**Source**: Ministry of National Education  
**Dataset**: MEN_INDICE_PARIDAD_POR_GENERO_DISCAPACIDAD_ETC  
**URL**: [Datos.gov.co](https://www.datos.gov.co/Educaci-n/MEN_INDICE_PARIDAD_POR_GENERO_DISCAPACIDAD_ETC/yt9f-v2f7)

**Characteristics**:
- Gender Parity Indices (GPI)
- Educational equity indicators
- Distribution by disability type

**Usage in project**:
- Gender gap analysis in mental health care
- Vulnerable population identification
- Equity indicators calculation for service access

### 4. ECAS 2016 + Updated Data

**Sources**:
- ECAS 2016 (Bogotá Education Secretariat)
- UNICEF Colombia - 2024 Report
- Legal Medicine Institute - Statistics 2023-2024
- National Study on Substance Use 2022

**Variables**: 10 risk factors with time series 2016-2024

---

## 🤖 Machine Learning Models

### 1. Random Forest Classifier

**Objective**: Classify localities by risk level

**Architecture**:
```python
RandomForestClassifier(
    n_estimators=100,        # 100 decision trees
    max_depth=10,            # Maximum depth
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples in leaf
    max_features='sqrt',     # Features per tree
    random_state=42,
    class_weight='balanced'  # Handle imbalanced classes
)
```

**Input Features (15 variables)**:
1. `total_atenciones` - Cumulative total
2. `matricula` - Locality enrollment
3. `tasa_por_500` - (visits/enrollment) × 500
4. `porcentaje_masculino` - % male visits
5. `porcentaje_femenino` - % female visits
6. `brecha_genero` - M/F ratio
7. `porcentaje_primaria` - % primary level (6-10 years)
8. `porcentaje_secundaria` - % secondary level (11-14 years)
9. `porcentaje_media` - % high school level (15-17 years)
10. `top_trastorno_1_dummy` - Primary disorder binary
11. `top_trastorno_2_dummy` - Secondary disorder binary
12. `top_trastorno_3_dummy` - Tertiary disorder binary
13. `indice_paridad` - Locality GPI
14. `tendencia_crecimiento` - % growth 2019-2024
15. `año` - Temporal variable

**Target**: `riesgo` (High / Medium / Low)

**Classification Criteria**:
```python
if tasa_por_500 >= 12.5:
    riesgo = "Alto"  # High
elif tasa_por_500 >= 7.5:
    riesgo = "Medio"  # Medium
else:
    riesgo = "Bajo"  # Low
```

**Performance Metrics**:
- **Accuracy**: 87%
- **Precision (High)**: 92%
- **Recall (High)**: 85%
- **F1-Score**: 0.88
- **ROC-AUC**: 0.91

**Validation**:
- Train-test split: 80-20
- 5-fold cross-validation
- Stratified sampling

**Feature Importance**:
```
1. tasa_por_500              28%
2. total_atenciones          22%
3. tendencia_crecimiento     15%
4. brecha_genero             12%
5. concentracion_hhi         10%
6. Other variables           13%
```

### 2. Deep Neural Network

**Objective**: Predict visits 2025-2030

**Architecture**:
```python
Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)  # Output: continuous prediction
])
```

**Hyperparameters**:
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Epochs**: 100 (with early stopping)
- **Batch size**: 16
- **Validation split**: 20%

**Callbacks**:
```python
[
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5
    )
]
```

**Input Features**:
- Base variables + lag features (previous year, 2 years back)
- 3-year moving average
- Year-over-year change rate
- Features scaled with StandardScaler

**Performance Metrics**:
- **RMSE**: 156 visits
- **MAE**: 124 visits
- **R² Score**: 0.94
- **MAPE**: 3.2%

**Interpretation**: Error of ±156 cases represents less than 3% on scale of 5,000-10,000 annual visits.

### 3. K-Means Clustering

**Objective**: Group localities by similarity

**Configuration**:
```python
KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
```

**Input Features (6 variables)**:
1. Rate per 500 students
2. Year-over-year growth (%)
3. Case volatility
4. Gender gap
5. Disorder concentration (HHI)
6. Parity index

**Normalization**: StandardScaler (critical for K-Means)

**K-determination Method**:
- Elbow Method
- Silhouette Score: 0.68

**Identified Clusters**:
- **Cluster 0 (High Risk)**: 6 localities
- **Cluster 1 (Medium Risk)**: 8 localities
- **Cluster 2 (Low Risk)**: 6 localities

---

## 🔧 Feature Engineering

### Created Variables (50 of 70 total)

#### 1. Temporal Variables
```python
df['crecimiento_anual'] = df.groupby('localidad')['atenciones'].pct_change()
df['crecimiento_acumulado'] = ((df['atenciones'] / df['atenciones_2019']) - 1) * 100
df['volatilidad'] = df.groupby('localidad')['atenciones'].transform(lambda x: x.std())
df['tendencia_lineal'] = calcular_tendencia_lineal(df)
```

#### 2. Rate Variables
```python
df['tasa_por_500'] = (df['atenciones'] / df['matricula']) * 500
df['tasa_por_1000'] = (df['atenciones'] / df['matricula']) * 1000
df['ratio_atencion_matricula'] = df['atenciones'] / df['matricula']
```

#### 3. Gender Variables
```python
df['porcentaje_masculino'] = (df['atenciones_m'] / df['atenciones']) * 100
df['porcentaje_femenino'] = (df['atenciones_f'] / df['atenciones']) * 100
df['brecha_genero'] = df['atenciones_m'] / df['atenciones_f']
```

#### 4. Concentration Variables
```python
# Herfindahl-Hirschman Index
def calcular_hhi(df):
    shares = (df.groupby('trastorno')['atenciones'].sum() / 
              df['atenciones'].sum()) ** 2
    return shares.sum()

df['concentracion_hhi'] = df.groupby('localidad').apply(calcular_hhi)
```

#### 5. Categorical Variables (One-Hot Encoding)
```python
top_trastornos = df['trastorno'].value_counts().head(3).index
for i, trastorno in enumerate(top_trastornos, 1):
    df[f'top_trastorno_{i}_dummy'] = (df['trastorno'] == trastorno).astype(int)
```

#### 6. Age Variables
```python
def asignar_nivel_educativo(edad_promedio):
    if 6 <= edad_promedio <= 10:
        return 'Primary (6-10)'
    elif 11 <= edad_promedio <= 14:
        return 'Secondary (11-14)'
    elif 15 <= edad_promedio <= 17:
        return 'High School (15-17)'
    else:
        return 'Other'

df['nivel_educativo'] = df['edad_promedio'].apply(asignar_nivel_educativo)
```

#### 7. Time Series Variables (for Neural Network)
```python
# Lag features
df['lag_1'] = df.groupby('localidad')['atenciones'].shift(1)
df['lag_2'] = df.groupby('localidad')['atenciones'].shift(2)

# Moving average
df['media_movil_3'] = df.groupby('localidad')['atenciones'].rolling(3).mean().reset_index(0, drop=True)

# Change rate
df['tasa_cambio'] = df.groupby('localidad')['atenciones'].diff()
```

---

## 📱 Dashboard Modules

### Module 1: Home

**Components**:
- Key KPIs (population, visits, rate)
- Risk traffic light with interpretation
- Executive summary

**Technologies**: `streamlit.metric()`, `plotly.graph_objects`

### Module 2: Key Indicators

**Visualizations**:
- Time series 2019-2024 (line chart)
- Gender distribution (pie chart)
- Calculated gender gap
- Counselors needed vs available

**Key code**:
```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['año'], 
    y=df['atenciones'],
    mode='lines+markers',
    name='Visits'
))
st.plotly_chart(fig, use_container_width=True)
```

### Module 3: Risk Map

**Analysis**:
- ML classification by locality
- K-Means clustering
- Top 10 critical localities
- Confusion matrix

**Outputs**:
- Interactive table with classification
- Bar chart with risk levels
- Model confidence metrics

### Module 4: Temporal Analysis and Predictions

**Integrated Models**:
- Historical 2019-2024
- ML/DL predictions 2025
- 95% confidence intervals
- Volatility analysis

**Predictive Visualization**:
```python
# Historical line
fig.add_trace(go.Scatter(
    x=df_historico['año'],
    y=df_historico['atenciones'],
    mode='lines+markers',
    name='Historical',
    line=dict(color='blue', width=3)
))

# Prediction line
fig.add_trace(go.Scatter(
    x=df_pred['año'],
    y=df_pred['atenciones_pred'],
    mode='lines+markers',
    name='Prediction',
    line=dict(color='red', width=3, dash='dash')
))
```

### Module 5: Risk Factors

**ECAS Analysis + Projections**:
- 10 risk factors (2016-2030)
- Substance use with trends
- School violence and suicidal ideation
- Projections with polynomial regression

**Analyzed Factors**:
1. General mental health (44.7%)
2. Anxiety (15.2%)
3. Depression (15.7%)
4. ADHD (3.1%)
5. Alcohol consumption (50.8%)
6. Tobacco consumption (12.9%)
7. Marijuana consumption (12.8%)
8. Bullying (28.6%)
9. Suicidal ideation (7.1%)
10. Problematic substance use (5.8/100k)

### Module 6: Gender Analysis

**Components**:
- Distribution by gender and disorder
- Temporal evolution of gap
- Disorders with greatest difference
- Gender predictions through 2030

### Module 7: Locality Search

**Functionality**:
- Interactive locality selector
- Complete profile with all indicators
- Comparison with Bogotá average
- Specific locality charts

### Module 8: Download Reports

**Available Formats**:
- CSV (all datasets)
- JSON (KPIs and alerts)
- Custom reports by dimension

---

## 🚀 Installation and Deployment

### Prerequisites

```bash
Python 3.10 or higher
pip 23.0 or higher
Git
```

### Local Installation

```bash
# 1. Clone repository
git clone https://github.com/paulabadt/observatorio-salud-mental-bogota.git
cd observatorio-salud-mental-bogota

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app_dashboard.py
```

### Deployment on Streamlit Cloud

1. Fork repository on GitHub
2. Access [share.streamlit.io](https://share.streamlit.io)
3. Connect with GitHub
4. Select repository and `main` branch
5. Specify `app_dashboard.py` as main file
6. Click "Deploy"

**Deployment time**: ~5-10 minutes

### Environment Variables

No environment variables required. All data is in public CSV files.

### File Structure

```
observatorio-salud-mental-bogota/
├── app_dashboard.py                      # Main dashboard
├── requirements.txt                      # Dependencies
├── README.md                             # Documentation
│
├── data/                                 # Processed data
│   ├── morbilidad_salud_mental_limpio.csv
│   ├── dataset_integrado_completo.csv
│   ├── clasificacion_riesgo_localidades.csv
│   ├── clustering_localidades.csv
│   ├── kpis_y_alertas.json
│   ├── predicciones_totales_2030.csv
│   ├── predicciones_genero_2030.csv
│   ├── predicciones_localidad_2030.csv
│   └── coordenadas_localidades_bogota.csv
│
└── .streamlit/
    └── config.toml                       # Streamlit config
```

---

## 📈 Results and Metrics

### Analysis Time Impact

| Task | Before | Now | Reduction |
|------|--------|-----|-----------|
| Data consolidation | 3 days | 5 seconds | 99.998% |
| Indicator calculation | 5 days | 10 seconds | 99.997% |
| Chart generation | 3 days | Instant | 100% |
| Predictive analysis | 10 days | 2 minutes | 99.986% |
| **Total** | **21 days** | **5 minutes** | **99.976%** |

### Model Metrics

#### Random Forest
- **Overall Accuracy**: 87%
- **Cohen's Kappa**: 0.82 (substantial agreement)
- **Specificity**: 91%
- **Sensitivity**: 85%

#### Neural Network
- **RMSE**: 156 cases (2.9% relative error)
- **MAE**: 124 cases
- **R²**: 0.94 (explains 94% of variance)
- **Directional Accuracy**: 89% (correctly predicts trend)

#### K-Means
- **Silhouette Score**: 0.68 (good clustering)
- **Inertia**: 23.4
- **Davies-Bouldin Index**: 0.52 (well-separated clusters)

### Key 2030 Projections

| Factor | 2024 | 2030 (Projected) | Change |
|--------|------|------------------|--------|
| General Mental Health | 44.7% | 43.4% | -2.9% |
| Marijuana Use | 12.8% | 16.4% | +28.1% ⚠️ |
| Problematic Substance Use | 5.8/100k | 8.9/100k | +53.4% 🔴 |
| ADHD | 3.1% | 3.7% | +19.4% |
| Suicidal Ideation | 7.1% | 6.2% | -12.7% |

---

## 🏆 Recognition: Data to Ecosystem 2025

<div align="center">

### **5th Place Nationally**

![Data to Ecosystem](https://img.shields.io/badge/Data%20to%20Ecosystem-2025-gold?style=for-the-badge)

</div>

The **Bogotá School Mental Health Observatory** achieved **5th place nationally** in the **Data to Ecosystem 2025** competition, organized by the Ministry of Information and Communication Technologies (MinTIC) and Colombia Open Data.

### About the Competition

**Data to Ecosystem** is Colombia's most important national open data competition, challenging teams from across the country to create innovative solutions using public government information to solve social, economic, and environmental problems.

**2025 Edition**:
- **Participants**: 150+ teams from all over Colombia
- **Categories**: Health, Education, Security, Environment, Economy
- **Judged by**: Panel of expert judges (MinTIC, universities, private sector)

### Evaluation Criteria

Our project was evaluated on:

1. **Innovative use of open data** (25%)
   - Integration of 4 never-before-connected sources
   - Processing of 59,657 records + 4M students
   
2. **Technological innovation** (20%)
   - 3 ML/DL models (Random Forest, Neural Network, K-Means)
   - Predictions through 2030 with confidence intervals
   - Real-time interactive dashboard
   
3. **Measurable social impact** (25%)
   - 95% reduction in analysis time
   - Tool ready for use by Ministry of Health
   - Potential to save lives through prevention
   
4. **Scalability and replicability** (15%)
   - Model applicable to any city
   - Methodology extensible to other public health topics
   - 48-hour deployment for new cities
   
5. **Technical quality and documentation** (15%)
   - Clean and documented code
   - Complete technical README
   - Functional live demo

### Highlighted Achievements

🥇 **TOP 5 among 150+ teams**  
📊 **Highest technical complexity**: Only project with 3 integrated AI models  
🎯 **Immediate impact**: Production-ready tool  
🌐 **Live demo**: Publicly available 24/7  
📈 **Most ambitious predictions**: Only project with projections through 2030

### Judge Testimonials

> *"A project that demonstrates how artificial intelligence can transform public health decision-making. The integration of multiple data sources and predictive capability are exceptional."*  
> — **MinTIC Judge**

> *"The combination of technical rigor with social impact focus is exemplary. This is the type of solution Colombia needs."*  
> — **Academic Judge**

### Media Coverage

- **MinTIC**: Featured project on official social media
- **Colombia Open Data**: Success case on official portal

### Post-Competition Next Steps

✅ **Meeting with MinSalud**: Scheduled for institutional presentation  
✅ **Expansion to 3 cities**: Medellín, Cali, Barranquilla  
✅ **Academic publication**: Paper in public health journal  
✅ **Open Source**: Complete code release for replication

---

## 👥 Team SENSORY

### Dr. Diana Carolina Abad
**PhD in Neuropsychology**

- 🎓 PhD in Clinical Neuropsychology
- 🏥 15 years of experience in child cognitive assessment
- 🧠 Specialist in neurodevelopmental disorders
- 📚 Publications in indexed journals

**Project Contribution**:
- Clinical validation of disorder categorization
- Design of early warning protocols
- Interpretation of ECAS risk factors
- Evidence-based intervention recommendations

### Paula Andrea Abad
** Software Developer & Data Analyst**

- 💻 Software Developer & Data Engineering
- 📊 Specialist in predictive analysis
- 🐍 Python, TensorFlow, Scikit-Learn
- 🎨 Data visualization and interactive dashboards

**Project Contribution**:
- Complete system architecture
- ML/DL model development
- Feature engineering (70 variables)
- Streamlit dashboard implementation
- Deployment and technical documentation

### Work Methodology

**Interdisciplinary Integration**:
- Cross-validation: technical (Paula) + clinical (Diana)
- Testing with real users (school counselors)

---

## 📄 License

This project is distributed under **MIT License**.

```
MIT License

Copyright (c) 2025 Team SENSORY

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Open Data

The datasets used are in the **public domain** according to Colombia's open data policy (Law 1712 of 2014). Use, redistribution, and analysis of this data is permitted with proper attribution to original sources.

---

## 🙏 Acknowledgments

To the institutions that make open data access possible:

- **Datos.gov.co** - For democratizing public information
- **MinTIC** - For organizing Data to Ecosystem 2025
- **District Health Secretariat of Bogotá** - Morbidity data
- **Ministry of National Education** - Enrollment and GPI data
- **UNICEF Colombia** - Updated child mental health data
- **Legal Medicine Institute** - Suicide statistics
- **School counselors** - Invaluable feedback during testing

---

## 📚 References

### Data Sources
1. District Health Secretariat of Bogotá. (2024). Mental Health Morbidity. Colombia Open Data.
2. Ministry of National Education. (2024). Official Enrollment by Grade. Colombia Open Data.
3. Ministry of National Education. (2024). Gender, Disability, and Other Parity Index. Colombia Open Data.
4. Bogotá Education Secretariat. (2016). ECAS - School Climate and Environment Survey.

### Validation Sources
5. UNICEF Colombia. (May 2024). "Embrace Your Mind" Campaign - Mental Health in Childhood and Adolescence.
6. National Institute of Legal Medicine and Forensic Sciences. (2024). External Cause Injury Statistics.
7. Ministry of Justice and Law. (2022). National Study on Psychoactive Substance Use in School Population.
8. UNODC/District Health Secretariat. (2022). Substance Use Study in Bogotá.

### Technical Methodology
9. Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
11. Lloyd, S. (1982). "Least squares quantization in PCM". IEEE Transactions on Information Theory, 28(2), 129-137.
12. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python". JMLR, 12, 2825-2830.

### Public Policy
13. Ministry of Health and Social Protection. (2024). National Mental Health Policy 2024-2033.
14. Law 1712 of 2014. Transparency and Right of Access to Public Information Law.

---

<div align="center">

### 💙 "Data doesn't change the world. People who act on data do." 💙

**Made with ❤️ by Team SENSORY**  
*Transforming data into hope, one analysis at a time*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://observatorio-salud-mental-bogota-2ynkrapsjostnmrxfcxbxz.streamlit.app/)
[![Open Data](https://img.shields.io/badge/Open-Data-blue?style=for-the-badge)](https://herramientas.datos.gov.co/usos/observatorio-de-salud-mental-escolar-bogota)


</div>
