# 🚗⚡ Predicción de Compra de Vehículos Eléctricos (Will_Buy_EV)

Proyecto de clasificación binaria para predecir si un cliente **comprará un vehículo eléctrico** (`Will_Buy_EV`: Yes / No) a partir de sus características demográficas y de comportamiento. Se entrenaron y compararon cinco modelos de Machine Learning bajo las mismas condiciones y se construyeron tres ensambles para evaluar si la combinación de modelos mejora el rendimiento.

---

## 📋 Tabla de contenidos

- [Objetivo](#-objetivo)
- [Dataset](#-dataset)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Metodología](#-metodología)
- [Resultados](#-resultados)
- [Análisis de los resultados](#-análisis-de-los-resultados)
- [Modelo seleccionado](#-modelo-seleccionado)
- [Cómo ejecutar el proyecto](#-cómo-ejecutar-el-proyecto)
- [Tecnologías](#-tecnologías)
- [Autor](#-autor)

---

## 🎯 Objetivo

Estimar la **probabilidad** de que un cliente compre un vehículo eléctrico. La métrica principal de evaluación es el **ROC-AUC**, por lo que el archivo de envío contiene probabilidades (no etiquetas 0/1).

---

## 📊 Dataset

| Característica | Detalle |
|---|---|
| Registros de entrenamiento | ~668 mil filas |
| Variable objetivo | `Will_Buy_EV` (No → 0, Yes → 1) |
| Tipo de problema | Clasificación binaria |
| Balance de clases | ~82.5 % No / ~17.5 % Yes (desbalanceado) |
| Tipos de variables | Numéricas y categóricas |
| Columnas descartadas | `id` |

---

## 📁 Estructura del repositorio

```

├── comparacion1.ipynb    # Comparación unificada de modelos + ensambles + predicción final
├── images/               # Gráficos de resultados
├── submission.csv        # Archivo de predicciones
└── README.md
```

---

## 🔬 Metodología

### 1. Partición de los datos
- **80 % entrenamiento / 20 % prueba**, con partición estratificada para conservar la proporción de clases.
- Sobre el conjunto de entrenamiento se aplicó **validación cruzada estratificada de 5 folds**, generando predicciones *out-of-fold* (OOF) para cada modelo.

### 2. Preprocesamiento
Cada familia de modelos recibió el preprocesamiento que necesita:

| Modelos | Numéricas | Categóricas |
|---|---|---|
| Naive Bayes, LinearSVC, Regresión Logística | Imputación (mediana) + `StandardScaler` | Imputación (moda) + `OneHotEncoder` |
| XGBoost, LightGBM | Sin transformación | Tipo `category` nativo |

Todo el preprocesamiento se aplica dentro de un `Pipeline`, evitando fuga de datos entre folds.

### 3. Modelos evaluados
- **Naive Bayes** (GaussianNB)
- **LinearSVC** (calibrado con `CalibratedClassifierCV` para obtener probabilidades)
- **Regresión Logística**
- **XGBoost**
- **LightGBM**

### 4. Ensambles
| Ensamble | Descripción |
|---|---|
| **Voting promedio** | Promedio simple de las probabilidades de los 5 modelos |
| **Voting ponderado** | Promedio con pesos optimizados (minimizando *log loss* sobre las predicciones OOF) |
| **Stacking** | Regresión logística como meta-modelo, entrenada sobre las probabilidades OOF |

### 5. Métricas
Accuracy, Precision, Recall, F1 macro y **ROC-AUC** sobre el conjunto de prueba, además del F1 en validación cruzada (OOF) y el tiempo de entrenamiento.

---

## 📈 Resultados

### Tabla comparativa

![Tabla de resultados](images/Matriz_de_resultados.png)

| Modelo | Accuracy | Precision | Recall | F1 macro | ROC-AUC | Tiempo (s) |
|---|---|---|---|---|---|---|
| Ensamble: Voting promedio | 0.8904 | 0.8046 | **0.8415** | **0.8209** | 0.9374 | – |
| LightGBM | **0.8983** | 0.8281 | 0.8103 | 0.8188 | 0.9416 | 19.5 |
| XGBoost | 0.8981 | 0.8277 | 0.8101 | 0.8184 | **0.9417** | 31.0 |
| Ensamble: Voting ponderado | 0.8979 | 0.8283 | 0.8075 | 0.8173 | 0.9413 | – |
| Ensamble: Stacking | **0.8983** | **0.8296** | 0.8062 | 0.8171 | 0.9379 | – |
| Regresión Logística | 0.8949 | 0.8228 | 0.8018 | 0.8117 | 0.9380 | 6.7 |
| LinearSVC | 0.8947 | 0.8228 | 0.8007 | 0.8110 | 0.9378 | 18.3 |
| Naive Bayes | 0.6587 | 0.6606 | 0.7786 | 0.6191 | 0.9215 | 3.3 |

### Comparación visual

![Comparación de modelos y ensambles](images/comparacion_de_modelos_y_ensambles.png)

### Curvas ROC

![Curvas ROC](images/comparacion_de_curvas_.png)

### Matrices de confusión

![Matrices de confusión](images/matriz_de_resultadtos_de_los_modelos.png)

### Acuerdo entre modelos

![Acuerdo entre modelos](images/Matriz_de_acuerdo_entre_modelos.png)

---

## 🧠 Análisis de los resultados

**Los modelos de boosting son los más fuertes.** XGBoost (AUC 0.9417) y LightGBM (AUC 0.9416) obtienen el mejor ROC-AUC, prácticamente empatados. Sus predicciones coinciden en un 99 %, lo que confirma que aprendieron patrones muy similares.

**Los modelos lineales quedan muy cerca.** Regresión Logística y LinearSVC alcanzan un AUC de ~0.938, apenas 0.004 por debajo del boosting, y la Regresión Logística es la más rápida de los modelos competitivos (6.7 s). Ambos modelos coinciden al 100 % en sus predicciones: en la práctica son el mismo modelo.

**Naive Bayes ordena bien, pero clasifica mal.** Su AUC de 0.92 indica que separa razonablemente a compradores de no compradores, pero con el umbral de 0.5 predice "Yes" en exceso: genera **44 768 falsos positivos**, lo que hunde su accuracy a 0.66. Esto se debe a que sus probabilidades están mal calibradas (el supuesto de independencia entre variables no se cumple).

**Los ensambles no superaron al mejor modelo individual en AUC.** La matriz de acuerdo lo explica: los cuatro modelos competitivos coinciden entre un 98 % y un 100 %, así que hay muy poca diversidad que aprovechar. El único modelo distinto (Naive Bayes) es también el más débil.

**El Voting promedio es un caso particular.** Logra el mejor F1 (0.8209) y el mejor Recall (0.8415), pero su AUC baja a 0.9374. Al promediar con Naive Bayes, las probabilidades se desplazan hacia "Yes": detecta más compradores reales (17 898 vs. 15 769 de LightGBM) a costa de más falsos positivos (9 204 vs. 6 011). Es útil si el objetivo del negocio es **no perder compradores potenciales**, pero no para maximizar el AUC.

**El Voting ponderado confirma la conclusión:** al optimizar los pesos, el ensamble termina apoyándose casi por completo en los modelos de boosting y alcanza un AUC (0.9413) equivalente al de LightGBM solo.

---

## 🏆 Modelo seleccionado

Como la métrica de evaluación es **ROC-AUC**, se eligió **LightGBM**:

- Empata con XGBoost en AUC (0.9416 vs. 0.9417) y en accuracy.
- Entrena **~37 % más rápido** que XGBoost (19.5 s vs. 31.0 s).
- Los ensambles no aportan una mejora que justifique su complejidad adicional.

El modelo final se reentrenó con el 100 % de los datos de entrenamiento antes de generar las predicciones.

### Formato del archivo de envío

```csv
id,Will_Buy_EV
668665,0.0342
668666,0.1187
668667,0.0051
...
```

---

## 🚀 Cómo ejecutar el proyecto

1. Clonar el repositorio:
```bash
git clone https://github.com/KREMC/Kaggle-Predicting-Electric-Vehicle-Purchases.git
cd <nombre-del-repositorio>
```

2. Instalar dependencias:
```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn scipy joblib
```

3. Colocar `train.csv` y `test.csv` en la raíz del proyecto.

4. Ejecutar `comparacion1.ipynb` de principio a fin. Al terminar se genera `submission.csv`.

---

## 🛠 Tecnologías

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189FDD)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B)
![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white)

---

## 👤 Autor

**Royer Elvis Moreano Condorcuya**
Ingeniero de Sistemas e Informática — Abancay, Perú
GitHub: [@KREMC](https://github.com/KREMC)