# 🚗⚡ Predicción de Compra de Vehículos Eléctricos

Modelo de clasificación con **XGBoost** para predecir si una persona comprará un vehículo eléctrico, desarrollado para la competencia de Kaggle [Predicting Electric Vehicle Purchases](https://www.kaggle.com/competitions/playground-series-s6e9/overview) (Playground Series – Season 6, Episode 9).

---

## 📌 Resumen

| Métrica (validación) | Valor |
|---|---|
| **AUC-ROC** | **0.942** |
| Accuracy | 0.89 |
| Recall clase "Yes" | 0.79 |
| Precision clase "Yes" | 0.65 |
| F1 clase "Yes" | 0.72 |
| Umbral de decisión | 0.36 |

---

## 📊 Datos

El conjunto de entrenamiento tiene **668 665 registros** y 13 variables predictoras:

- **Numéricas:** `Age`, `Annual_Income_USD`, `Daily_Commute_km`, `Number_of_Cars_Owned`, `Charging_Stations_Near_Home`, `Charging_Stations_Near_Work`, `Environmental_Concern_Level`
- **Categóricas:** `Gender`, `City_Type`, `Current_Car_Type`, `Home_Charging_Possible`, `Subsidy_Available`, `Range_Anxiety_Level`
- **Objetivo:** `Will_Buy_EV` (Yes / No)

### El reto: clases desbalanceadas

Solo el **17.5 %** de las personas compra un vehículo eléctrico. Un modelo que prediga siempre "No" tendría 82 % de accuracy sin aportar nada, por eso la evaluación se centró en **AUC, precision, recall y F1** en lugar de la accuracy.

---

## ⚙️ Metodología

1. **División estratificada 80/20** para mantener la proporción de clases en entrenamiento y validación.
2. **Variables categóricas nativas** de XGBoost (`enable_categorical=True`), sin one-hot encoding ni escalado, ya que los árboles no dependen de la escala de las variables.
3. **Early stopping** con 100 rondas de paciencia: el modelo alcanzó su mejor AUC en la **iteración 497**.
4. **Optimización del umbral de decisión** para maximizar el F1 de la clase minoritaria.
5. **Interpretación** con importancia por *gain* y valores **SHAP**.
6. **Persistencia** del modelo, el umbral y las categorías para predecir sobre datos nuevos sin reentrenar.

### Hiperparámetros

```python
XGBClassifier(
    n_estimators=3000,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method='hist',
    enable_categorical=True,
    eval_metric='auc',
    early_stopping_rounds=100,
)
```

---

## 🎯 Ajuste del umbral

Por defecto un clasificador predice "Yes" cuando la probabilidad supera **0.5**. Con clases desbalanceadas, ese corte deja escapar a muchos compradores reales.

Se evaluaron umbrales entre 0.10 y 0.90, y el que maximizó el F1 fue **0.36**:

| Clase | Precision | Recall | F1 | Soporte |
|---|---|---|---|---|
| No (0) | 0.95 | 0.91 | 0.93 | 110 377 |
| Yes (1) | 0.65 | 0.79 | 0.72 | 23 356 |

Con este umbral el modelo **detecta 8 de cada 10 compradores reales**, a cambio de que aproximadamente 1 de cada 3 personas marcadas como compradoras no lo sea. En un caso de marketing es un intercambio favorable: cuesta menos contactar a un cliente que no compra que perder a uno que sí lo haría.

---

## 🔍 ¿Qué variables influyen más?

![Importancia de variables](imagenes/importancia_variables.png)

- **Gain (izquierda):** mide cuánto mejora el modelo cada vez que usa una variable. Domina la **disponibilidad de subsidio**.
- **SHAP (derecha):** mide cuánto mueve cada variable la predicción individual. Lidera la **preocupación ambiental**, seguida muy de cerca por el **subsidio** y luego el **ingreso anual**.

Ambas medidas coinciden: la motivación ambiental y el incentivo económico explican la mayor parte de la decisión. En cambio, las estaciones de carga cercanas, la cantidad de autos y el género apenas influyen.

---

## 📁 Estructura del proyecto

```
├── datos/                 # train.csv y test.csv (descargar desde Kaggle)
├── modelo/
│   ├── modelo_xgb.json    # modelo entrenado
│   └── config_modelo.json # umbral, categorías y columnas
├── imagenes/
│   └── importancia_variables.png
├── XGBOOST.ipynb          # notebook principal
└── README.md
```

---

## ▶️ Cómo ejecutarlo

1. Clona el repositorio:
   ```bash
   git clone https://github.com/KREMC/Kaggle-Predicting-Electric-Vehicle-Purchases.git
   ```
2. Descarga `train.csv` y `test.csv` desde la [página de la competencia](https://www.kaggle.com/competitions/playground-series-s6e9/data) y colócalos en `datos/`.
3. Instala las dependencias:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib
   ```
4. Abre y ejecuta `XGBOOST.ipynb`.

---

## 🛠️ Tecnologías

Python · Pandas · NumPy · Scikit-learn · XGBoost · Matplotlib · Jupyter

---

## 👤 Autor

**Royer Elvis Moreano Condorcuya**
Ingeniero de Sistemas e Informática · Machine Learning / Data Science

[GitHub](https://github.com/KREMC) · [LinkedIn](https://www.linkedin.com/in/TU-PERFIL)
