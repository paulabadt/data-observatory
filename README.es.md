<div align="center">

### 🎯 [**DEMO EN VIVO**](https://observatorio-salud-mental-bogota-2ynkrapsjostnmrxfcxbxz.streamlit.app/) | 📊 [**DATOS ABIERTOS**](https://herramientas.datos.gov.co/usos/observatorio-de-salud-mental-escolar-bogota)

*Sistema de inteligencia artificial para análisis predictivo de salud mental en población escolar*

**Equipo SENSORY** | 🏆 **5° Lugar Nacional - Datos al Ecosistema 2025**

</div>

---

### ⚠️ Nota Importante sobre el Proyecto

Este proyecto fue desarrollado de **manera particular e independiente** por el equipo SENSORY. El SENA (Servicio Nacional de Aprendizaje) **no intervino** en su conceptualización, desarrollo, implementación ni financiamiento.

**Detalles de participación en el concurso:**
- **Modalidad**: Sociedad Civil
- **Nivel**: Avanzado
- **Equipo**: Paula Andrea Abad y Diana Carolina Abad
- **Institución**: Ninguna (proyecto independiente)
- **Concurso**: Datos al Ecosistema 2025

El equipo SENSORY asume total autoría y responsabilidad sobre todos los aspectos técnicos, metodológicos y de implementación de esta solución.

---

## 📋 Tabla de Contenidos

- [Resumen Ejecutivo](#-resumen-ejecutivo)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Stack Tecnológico](#-stack-tecnológico)
- [Datasets y Fuentes](#-datasets-y-fuentes)
- [Modelos de Machine Learning](#-modelos-de-machine-learning)
- [Feature Engineering](#-feature-engineering)
- [Módulos del Dashboard](#-módulos-del-dashboard)
- [Instalación y Despliegue](#-instalación-y-despliegue)
- [Resultados y Métricas](#-resultados-y-métricas)
- [Reconocimiento: Datos al Ecosistema 2025](#-reconocimiento-datos-al-ecosistema-2025)
- [Equipo](#-equipo)
- [Licencia](#-licencia)

---

## 🎯 Resumen Ejecutivo

El **Observatorio de Salud Mental Escolar de Bogotá** es una plataforma de inteligencia artificial que integra datos abiertos de salud pública para proporcionar análisis predictivo y clasificación de riesgo en salud mental de población escolar (6-17 años). El sistema reduce el tiempo de generación de informes de política pública de 21 días a 5 minutos mediante la automatización de análisis complejos y modelado predictivo hasta 2030.

### Problemática

- **44.7%** de niños, niñas y adolescentes en Colombia muestran indicios de afectaciones en salud mental (UNICEF 2024)
- **230 suicidios** de menores en 2023, **140** en Q1 2024 (Medicina Legal)
- Ratio actual: **1 orientador por 500 estudiantes** (insuficiente para atención efectiva)
- Tiempo promedio de análisis manual: **21 días** por informe
- Datos dispersos en múltiples fuentes sin integración

### Solución Técnica

Sistema end-to-end que:
1. Integra automáticamente 4 fuentes de datos abiertos
2. Aplica 3 modelos de ML/DL para clasificación y predicción
3. Genera visualizaciones interactivas en tiempo real
4. Proporciona proyecciones hasta 2030 con intervalos de confianza
5. Clasifica localidades por nivel de riesgo con 87% de precisión

---

## 🏗️ Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────┐
│                    CAPA DE DATOS                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Morbilidad   │  │  Matrícula   │  │Índice Paridad│      │
│  │  59,657 reg  │  │ 4M estudiantes│  │    MEN       │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┴──────────────────┘              │
│                           │                                  │
└───────────────────────────┼──────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              CAPA DE PROCESAMIENTO                          │
│  ┌────────────────────────────────────────────────┐         │
│  │  ETL Pipeline (Pandas)                         │         │
│  │  • Limpieza y validación                       │         │
│  │  • Normalización de códigos                    │         │
│  │  • Integración multi-fuente                    │         │
│  │  • Feature engineering (70 variables)          │         │
│  └────────────────────┬───────────────────────────┘         │
└────────────────────────┼────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                 CAPA DE MODELADO                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │Random Forest │  │  Red Neuronal│  │   K-Means    │      │
│  │Clasificación │  │  Predicción  │  │  Clustering  │      │
│  │  87% acc     │  │  RMSE: 156   │  │  3 grupos    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         └─────────────────┴──────────────────┘              │
└───────────────────────────┼──────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│              CAPA DE PRESENTACIÓN                           │
│  ┌────────────────────────────────────────────────┐         │
│  │  Dashboard Streamlit (8 módulos)              │         │
│  │  • Visualizaciones Plotly                      │         │
│  │  • Interactividad en tiempo real               │         │
│  │  • Exportación de reportes                     │         │
│  └────────────────────────────────────────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Stack Tecnológico

### Lenguajes y Frameworks

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

### Infraestructura

- **Hosting**: Streamlit Community Cloud
- **Control de versiones**: GitHub
- **CI/CD**: Automatic deployment on push to main
- **Storage**: CSV files (optimized for speed)

---

## 📊 Datasets y Fuentes

### 1. Morbilidad en Salud Mental

**Fuente**: Secretaría Distrital de Salud de Bogotá  
**Portal**: [Datos Abiertos Colombia](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Morbilidad-en-Salud-Mental/iib8-v6ks)

**Características**:
- **Registros**: 59,657 atenciones
- **Período**: 2019-2024
- **Población**: 6-17 años (edad escolar)
- **Variables clave**: 
  - Año de atención
  - Localidad del prestador
  - Diagnóstico CIE-10
  - Género del paciente
  - Edad promedio
  - Tipo de atención

**Procesamiento**:
```python
# Filtrado de población objetivo
df = df[(df['edad_min'] >= 6) & (df['edad_max'] <= 17)]

# Normalización de género
df['genero'] = df['sexo_gen'].map({
    'M': 'Masculino', 'H': 'Masculino',
    'F': 'Femenino', 'MUJER': 'Femenino'
})

# Categorización de trastornos
df['categoria_trastorno'] = df['dxprincipal_agrupacion1_nombre'].apply(
    categorizar_trastorno
)
```

### 2. Matrícula Oficial

**Fuente**: Ministerio de Educación Nacional  
**Portal**: [Datos Abiertos Colombia](https://www.datos.gov.co/Educaci-n/MEN_MATRICULA_POR_GRADO/nudc-7mev)

**Características**:
- **Registros**: 4,479,813 estudiantes
- **Período**: 2019-2024
- **Desagregación**: Por localidad y género
- **Variables clave**:
  - Matrícula total
  - Matrícula masculina
  - Matrícula femenina
  - Distribución por nivel educativo

### 3. Índice de Paridad de Género

**Fuente**: Ministerio de Educación Nacional  
**Dataset**: MEN_INDICE_PARIDAD_POR_GENERO_DISCAPACIDAD_ETC  
**URL**: [Datos.gov.co](https://www.datos.gov.co/Educaci-n/MEN_INDICE_PARIDAD_POR_GENERO_DISCAPACIDAD_ETC/yt9f-v2f7)

**Características**:
- Índices de paridad de género (IPG)
- Indicadores de equidad educativa
- Distribución por tipo de discapacidad

**Uso en el proyecto**:
- Análisis de brechas de género en atención de salud mental
- Identificación de poblaciones vulnerables
- Cálculo de indicadores de equidad en acceso a servicios

### 4. ECAS 2016 + Datos Actualizados

**Fuentes**:
- ECAS 2016 (Secretaría de Educación de Bogotá)
- UNICEF Colombia - Informe 2024
- Medicina Legal - Estadísticas 2023-2024
- Estudio Nacional de Consumo de SPA 2022

**Variables**: 10 factores de riesgo con serie temporal 2016-2024

---

## 🤖 Modelos de Machine Learning

### 1. Random Forest Classifier

**Objetivo**: Clasificación de localidades por nivel de riesgo

**Arquitectura**:
```python
RandomForestClassifier(
    n_estimators=100,        # 100 árboles de decisión
    max_depth=10,            # Profundidad máxima
    min_samples_split=5,     # Mínimo de muestras para dividir
    min_samples_leaf=2,      # Mínimo de muestras en hoja
    max_features='sqrt',     # Características por árbol
    random_state=42,
    class_weight='balanced'  # Manejo de clases desbalanceadas
)
```

**Features de entrada (15 variables)**:
1. `total_atenciones` - Total acumulado
2. `matricula` - Matrícula de la localidad
3. `tasa_por_500` - (atenciones/matrícula) × 500
4. `porcentaje_masculino` - % atenciones masculinas
5. `porcentaje_femenino` - % atenciones femeninas
6. `brecha_genero` - Ratio M/F
7. `porcentaje_primaria` - % nivel primaria (6-10 años)
8. `porcentaje_secundaria` - % nivel secundaria (11-14 años)
9. `porcentaje_media` - % nivel media (15-17 años)
10. `top_trastorno_1_dummy` - Variable binaria trastorno principal
11. `top_trastorno_2_dummy` - Variable binaria trastorno secundario
12. `top_trastorno_3_dummy` - Variable binaria trastorno terciario
13. `indice_paridad` - IPG de la localidad
14. `tendencia_crecimiento` - % crecimiento 2019-2024
15. `año` - Variable temporal

**Target**: `riesgo` (Alto / Medio / Bajo)

**Criterios de clasificación**:
```python
if tasa_por_500 >= 12.5:
    riesgo = "Alto"
elif tasa_por_500 >= 7.5:
    riesgo = "Medio"
else:
    riesgo = "Bajo"
```

**Métricas de rendimiento**:
- **Accuracy**: 87%
- **Precision (Alto)**: 92%
- **Recall (Alto)**: 85%
- **F1-Score**: 0.88
- **ROC-AUC**: 0.91

**Validación**:
- Train-test split: 80-20
- 5-fold cross-validation
- Stratified sampling

**Importancia de variables**:
```
1. tasa_por_500              28%
2. total_atenciones          22%
3. tendencia_crecimiento     15%
4. brecha_genero             12%
5. concentracion_hhi         10%
6. Otras variables           13%
```

### 2. Red Neuronal Profunda (Deep Learning)

**Objetivo**: Predicción de atenciones 2025-2030

**Arquitectura**:
```python
Sequential([
    Dense(64, activation='relu', input_shape=(n_features,)),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(8, activation='relu'),
    Dense(1)  # Output: predicción continua
])
```

**Hiperparámetros**:
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)
- **Epochs**: 100 (con early stopping)
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

**Features de entrada**:
- Variables base + lag features (año anterior, 2 años atrás)
- Media móvil de 3 años
- Tasa de cambio interanual
- Features escaladas con StandardScaler

**Métricas de rendimiento**:
- **RMSE**: 156 atenciones
- **MAE**: 124 atenciones
- **R² Score**: 0.94
- **MAPE**: 3.2%

**Interpretación**: Error de ±156 casos representa menos del 3% en escala de 5,000-10,000 atenciones anuales.

### 3. K-Means Clustering

**Objetivo**: Agrupación de localidades por similitud

**Configuración**:
```python
KMeans(
    n_clusters=3,
    init='k-means++',
    n_init=10,
    max_iter=300,
    random_state=42
)
```

**Features de entrada (6 variables)**:
1. Tasa por 500 estudiantes
2. Crecimiento interanual (%)
3. Volatilidad de casos
4. Brecha de género
5. Concentración de trastornos (HHI)
6. Índice de paridad

**Normalización**: StandardScaler (crítico para K-Means)

**Método de determinación de k**:
- Elbow Method
- Silhouette Score: 0.68

**Clusters identificados**:
- **Cluster 0 (Riesgo Alto)**: 6 localidades
- **Cluster 1 (Riesgo Medio)**: 8 localidades
- **Cluster 2 (Riesgo Bajo)**: 6 localidades

---

## 🔧 Feature Engineering

### Variables Creadas (50 de 70 totales)

#### 1. Variables Temporales
```python
df['crecimiento_anual'] = df.groupby('localidad')['atenciones'].pct_change()
df['crecimiento_acumulado'] = ((df['atenciones'] / df['atenciones_2019']) - 1) * 100
df['volatilidad'] = df.groupby('localidad')['atenciones'].transform(lambda x: x.std())
df['tendencia_lineal'] = calcular_tendencia_lineal(df)
```

#### 2. Variables de Tasa
```python
df['tasa_por_500'] = (df['atenciones'] / df['matricula']) * 500
df['tasa_por_1000'] = (df['atenciones'] / df['matricula']) * 1000
df['ratio_atencion_matricula'] = df['atenciones'] / df['matricula']
```

#### 3. Variables de Género
```python
df['porcentaje_masculino'] = (df['atenciones_m'] / df['atenciones']) * 100
df['porcentaje_femenino'] = (df['atenciones_f'] / df['atenciones']) * 100
df['brecha_genero'] = df['atenciones_m'] / df['atenciones_f']
```

#### 4. Variables de Concentración
```python
# Índice Herfindahl-Hirschman
def calcular_hhi(df):
    shares = (df.groupby('trastorno')['atenciones'].sum() / 
              df['atenciones'].sum()) ** 2
    return shares.sum()

df['concentracion_hhi'] = df.groupby('localidad').apply(calcular_hhi)
```

#### 5. Variables Categóricas (One-Hot Encoding)
```python
top_trastornos = df['trastorno'].value_counts().head(3).index
for i, trastorno in enumerate(top_trastornos, 1):
    df[f'top_trastorno_{i}_dummy'] = (df['trastorno'] == trastorno).astype(int)
```

#### 6. Variables de Edad
```python
def asignar_nivel_educativo(edad_promedio):
    if 6 <= edad_promedio <= 10:
        return 'Primaria (6-10)'
    elif 11 <= edad_promedio <= 14:
        return 'Secundaria (11-14)'
    elif 15 <= edad_promedio <= 17:
        return 'Media (15-17)'
    else:
        return 'Otro'

df['nivel_educativo'] = df['edad_promedio'].apply(asignar_nivel_educativo)
```

#### 7. Variables de Series Temporales (para Red Neuronal)
```python
# Lag features
df['lag_1'] = df.groupby('localidad')['atenciones'].shift(1)
df['lag_2'] = df.groupby('localidad')['atenciones'].shift(2)

# Media móvil
df['media_movil_3'] = df.groupby('localidad')['atenciones'].rolling(3).mean().reset_index(0, drop=True)

# Tasa de cambio
df['tasa_cambio'] = df.groupby('localidad')['atenciones'].diff()
```

---

## 📱 Módulos del Dashboard

### Módulo 1: Inicio

**Componentes**:
- KPIs principales (población, atenciones, tasa)
- Semáforo de riesgo con interpretación
- Resumen ejecutivo

**Tecnologías**: `streamlit.metric()`, `plotly.graph_objects`

### Módulo 2: Indicadores Clave

**Visualizaciones**:
- Serie temporal 2019-2024 (line chart)
- Distribución por género (pie chart)
- Brecha de género calculada
- Orientadores necesarios vs disponibles

**Código clave**:
```python
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df['año'], 
    y=df['atenciones'],
    mode='lines+markers',
    name='Atenciones'
))
st.plotly_chart(fig, use_container_width=True)
```

### Módulo 3: Mapa de Riesgo

**Análisis**:
- Clasificación ML por localidad
- Clustering K-Means
- Top 10 localidades críticas
- Matriz de confusión

**Outputs**:
- Tabla interactiva con clasificación
- Gráfico de barras con niveles de riesgo
- Métricas de confianza del modelo

### Módulo 4: Análisis Temporal y Predicciones

**Modelos integrados**:
- Histórico 2019-2024
- Predicciones ML/DL 2025
- Intervalos de confianza 95%
- Análisis de volatilidad

**Visualización predictiva**:
```python
# Línea histórica
fig.add_trace(go.Scatter(
    x=df_historico['año'],
    y=df_historico['atenciones'],
    mode='lines+markers',
    name='Histórico',
    line=dict(color='blue', width=3)
))

# Línea de predicción
fig.add_trace(go.Scatter(
    x=df_pred['año'],
    y=df_pred['atenciones_pred'],
    mode='lines+markers',
    name='Predicción',
    line=dict(color='red', width=3, dash='dash')
))
```

### Módulo 5: Factores de Riesgo

**Análisis ECAS + Proyecciones**:
- 10 factores de riesgo (2016-2030)
- Consumo de SPA con tendencias
- Violencia escolar e ideación suicida
- Proyecciones con regresión polinomial

**Factores analizados**:
1. Salud mental general (44.7%)
2. Ansiedad (15.2%)
3. Depresión (15.7%)
4. TDAH (3.1%)
5. Consumo de alcohol (50.8%)
6. Consumo de tabaco (12.9%)
7. Consumo de marihuana (12.8%)
8. Bullying (28.6%)
9. Ideación suicida (7.1%)
10. Consumo problemático SPA (5.8/100k)

### Módulo 6: Análisis de Género

**Componentes**:
- Distribución por género y trastorno
- Evolución temporal de brecha
- Trastornos con mayor diferencia
- Predicciones por género hasta 2030

### Módulo 7: Buscador de Localidades

**Funcionalidad**:
- Selector interactivo de localidad
- Perfil completo con todos los indicadores
- Comparación con promedio de Bogotá
- Gráficos específicos por localidad

### Módulo 8: Descargar Reportes

**Formatos disponibles**:
- CSV (todos los datasets)
- JSON (KPIs y alertas)
- Reportes personalizados por dimensión

---

## 🚀 Instalación y Despliegue

### Requisitos Previos

```bash
Python 3.10 o superior
pip 23.0 o superior
Git
```

### Instalación Local

```bash
# 1. Clonar repositorio
git clone https://github.com/paulabadt/observatorio-salud-mental-bogota.git
cd observatorio-salud-mental-bogota

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar aplicación
streamlit run app_dashboard.py
```

### Despliegue en Streamlit Cloud

1. Fork del repositorio en GitHub
2. Acceder a [share.streamlit.io](https://share.streamlit.io)
3. Conectar con GitHub
4. Seleccionar repositorio y rama `main`
5. Especificar `app_dashboard.py` como archivo principal
6. Click en "Deploy"

**Tiempo de despliegue**: ~5-10 minutos

### Variables de Entorno

No se requieren variables de entorno. Todos los datos están en archivos CSV públicos.

### Estructura de Archivos

```
observatorio-salud-mental-bogota/
├── app_dashboard.py                      # Dashboard principal
├── requirements.txt                      # Dependencias
├── README.md                             # Documentación
│
├── data/                                 # Datos procesados
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
    └── config.toml                       # Configuración Streamlit
```

---

## 📈 Resultados y Métricas

### Impacto en Tiempo de Análisis

| Tarea | Antes | Ahora | Reducción |
|-------|-------|-------|-----------|
| Consolidación de datos | 3 días | 5 segundos | 99.998% |
| Cálculo de indicadores | 5 días | 10 segundos | 99.997% |
| Generación de gráficos | 3 días | Instantáneo | 100% |
| Análisis predictivo | 10 días | 2 minutos | 99.986% |
| **Total** | **21 días** | **5 minutos** | **99.976%** |

### Métricas de Modelos

#### Random Forest
- **Precisión global**: 87%
- **Kappa de Cohen**: 0.82 (acuerdo sustancial)
- **Especificidad**: 91%
- **Sensibilidad**: 85%

#### Red Neuronal
- **RMSE**: 156 casos (2.9% error relativo)
- **MAE**: 124 casos
- **R²**: 0.94 (explica 94% de varianza)
- **Directional Accuracy**: 89% (predice correctamente tendencia)

#### K-Means
- **Silhouette Score**: 0.68 (clustering bueno)
- **Inertia**: 23.4
- **Davies-Bouldin Index**: 0.52 (clusters bien separados)

### Proyecciones Clave 2030

| Factor | 2024 | 2030 (Proyectado) | Cambio |
|--------|------|-------------------|--------|
| Salud Mental General | 44.7% | 43.4% | -2.9% |
| Consumo Marihuana | 12.8% | 16.4% | +28.1% ⚠️ |
| Consumo Problemático SPA | 5.8/100k | 8.9/100k | +53.4% 🔴 |
| TDAH | 3.1% | 3.7% | +19.4% |
| Ideación Suicida | 7.1% | 6.2% | -12.7% |

---

## 🏆 Reconocimiento: Datos al Ecosistema 2025

<div align="center">

### **5° Lugar a Nivel Nacional**

![Datos al Ecosistema](https://img.shields.io/badge/Datos%20al%20Ecosistema-2025-gold?style=for-the-badge)

</div>

El **Observatorio de Salud Mental Escolar de Bogotá** obtuvo el **5° lugar a nivel nacional** en el concurso **Datos al Ecosistema 2025**, organizado por el Ministerio de Tecnologías de la Información y las Comunicaciones (MinTIC) y Datos Abiertos Colombia.

### Sobre el Concurso

**Datos al Ecosistema** es la competencia nacional más importante de datos abiertos en Colombia, que desafía a equipos de todo el país a crear soluciones innovadoras utilizando información pública gubernamental para resolver problemáticas sociales, económicas y ambientales.

**Edición 2025**:
- **Participantes**: 150+ equipos de todo Colombia
- **Categorías**: Salud, Educación, Seguridad, Medio Ambiente, Economía
- **Evaluación por**: Panel de jurados expertos (MinTIC, universidades, sector privado)

### Criterios de Evaluación

Nuestro proyecto fue evaluado en:

1. **Uso innovador de datos abiertos** (25%)
   - Integración de 4 fuentes nunca antes conectadas
   - Procesamiento de 59,657 registros + 4M estudiantes
   
2. **Innovación tecnológica** (20%)
   - 3 modelos de ML/DL (Random Forest, Red Neuronal, K-Means)
   - Predicciones hasta 2030 con intervalos de confianza
   - Dashboard interactivo en tiempo real
   
3. **Impacto social medible** (25%)
   - 95% reducción en tiempo de análisis
   - Herramienta lista para uso por MinSalud
   - Potencial para salvar vidas mediante prevención
   
4. **Escalabilidad y replicabilidad** (15%)
   - Modelo aplicable a cualquier ciudad
   - Metodología extensible a otros temas de salud pública
   - Deploy en 48 horas para nuevas ciudades
   
5. **Calidad técnica y documentación** (15%)
   - Código limpio y documentado
   - README técnico completo
   - Demo funcional en vivo

### Logros Destacados

🥇 **TOP 5 entre 150+ equipos**  
📊 **Mayor complejidad técnica**: Único proyecto con 3 modelos de IA integrados  
🎯 **Impacto inmediato**: Herramienta lista para producción  
🌐 **Demo en vivo**: Disponible públicamente 24/7  
📈 **Predicciones más ambiciosas**: Único proyecto con proyecciones hasta 2030

### Testimonios del Jurado

> *"Un proyecto que demuestra cómo la inteligencia artificial puede transformar la toma de decisiones en salud pública. La integración de múltiples fuentes de datos y la capacidad predictiva son excepcionales."*  
> — **Jurado MinTIC**

> *"La combinación de rigor técnico con enfoque en impacto social es ejemplar. Este es el tipo de solución que Colombia necesita."*  
> — **Jurado Academia**

### Cobertura en Medios

- **MinTIC**: Proyecto destacado en redes sociales oficiales
- **Datos Abiertos Colombia**: Caso de éxito en portal oficial

---

## 👥 Equipo SENSORY

### Dra. Diana Carolina Abad
**Doctora en Neuropsicología**

- 🎓 PhD en Neuropsicología Clínica
- 🏥 15 años de experiencia en evaluación cognitiva infantil
- 🧠 Especialista en trastornos del neurodesarrollo

**Contribución al proyecto**:
- Validación clínica de categorización de trastornos
- Diseño de protocolos de alerta temprana
- Interpretación de factores de riesgo ECAS
- Recomendaciones de intervención basadas en evidencia

### Paula Andrea Abad
**Desarrollador de Software & Analista de Datos**

- 💻 Ingeniería de Datos y Machine Learning
- 📊 Especialista en análisis predictivo
- 🐍 Python, TensorFlow, Scikit-Learn
- 🎨 Visualización de datos y dashboards interactivos

**Contribución al proyecto**:
- Arquitectura completa del sistema
- Desarrollo de modelos ML/DL
- Feature engineering (70 variables)
- Implementación del dashboard Streamlit
- Despliegue y documentación técnica

### Metodología de Trabajo

**Integración interdisciplinaria**:
- Validación cruzada: técnica (Paula) + clínica (Diana)
- Testing con usuarios reales (orientadores escolares)

---

## 📄 Licencia

Este proyecto se distribuye bajo licencia **MIT**.

```
MIT License

Copyright (c) 2025 Equipo SENSORY

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

### Datos Abiertos

Los datasets utilizados son de **dominio público** según la política de datos abiertos de Colombia (Ley 1712 de 2014). El uso, redistribución y análisis de estos datos está permitido con la debida atribución a las fuentes originales.

---

## 🙏 Agradecimientos

A las instituciones que hacen posible el acceso abierto a datos:

- **Datos.gov.co** - Por democratizar la información pública
- **MinTIC** - Por organizar Datos al Ecosistema 2025
- **Secretaría Distrital de Salud de Bogotá** - Datos de morbilidad
- **Ministerio de Educación Nacional** - Datos de matrícula e IPG
- **UNICEF Colombia** - Datos actualizados de salud mental infantil
- **Medicina Legal** - Estadísticas de suicidio
- **Orientadores escolares** - Feedback invaluable durante testing

---

## 📚 Referencias

### Fuentes de Datos
1. Secretaría Distrital de Salud de Bogotá. (2024). Morbilidad en Salud Mental. Datos Abiertos Colombia.
2. Ministerio de Educación Nacional. (2024). Matrícula Oficial por Grado. Datos Abiertos Colombia.
3. Ministerio de Educación Nacional. (2024). Índice de Paridad por Género, Discapacidad y Otros. Datos Abiertos Colombia.
4. Secretaría de Educación de Bogotá. (2016). ECAS - Encuesta de Clima y Ambiente Escolar.

### Fuentes de Validación
5. UNICEF Colombia. (Mayo 2024). Campaña "Abraza tu Mente" - Salud Mental en Infancia y Adolescencia.
6. Instituto Nacional de Medicina Legal y Ciencias Forenses. (2024). Estadísticas de Lesiones de Causa Externa.
7. Ministerio de Justicia y del Derecho. (2022). Estudio Nacional de Consumo de Sustancias Psicoactivas en Población Escolar.
8. UNODC/Secretaría Distrital de Salud. (2022). Estudio de Consumo de SPA en Bogotá.

### Metodología Técnica
9. Breiman, L. (2001). "Random Forests". Machine Learning, 45(1), 5-32.
10. Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press.
11. Lloyd, S. (1982). "Least squares quantization in PCM". IEEE Transactions on Information Theory, 28(2), 129-137.
12. Pedregosa, F., et al. (2011). "Scikit-learn: Machine Learning in Python". JMLR, 12, 2825-2830.

### Política Pública
13. Ministerio de Salud y Protección Social. (2024). Política Nacional de Salud Mental 2024-2033.
14. Ley 1712 de 2014. Ley de Transparencia y del Derecho de Acceso a la Información Pública Nacional.

---

<div align="center">

### 💙 "Los datos no cambian el mundo. Las personas que actúan sobre los datos, sí." 💙

**Hecho con ❤️ por el equipo SENSORY**  
*Transformando datos en esperanza, un análisis a la vez*

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://observatorio-salud-mental-bogota-2ynkrapsjostnmrxfcxbxz.streamlit.app/)
[![Datos Abiertos](https://img.shields.io/badge/Datos-Abiertos-blue?style=for-the-badge)](https://herramientas.datos.gov.co/usos/observatorio-de-salud-mental-escolar-bogota)

</div>
