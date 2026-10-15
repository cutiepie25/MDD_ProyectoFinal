# MDD_ProyectoFinal

Proyecto final de **Minería de Datos**. Predice si la financiación de un *bill* médico (cuenta de servicios médicos comprada a una firma de abogados) será **rentable** o tendrá **pérdida**. Incluye la limpieza de datos, el modelado con clasificadores individuales y de ensamble, y una app en **Streamlit** para hacer predicciones.

**Autores:** Cristian Usme Córdoba y María Alejandra Vargas Duque.

## Contenido

| Archivo | Descripción |
|---------|-------------|
| `final_limpieza_datos.ipynb` | Carga y tipado de datos, eliminación de variables irrelevantes y redundantes, análisis estadístico, limpieza de errores y atípicos, unificación de categorías con regex, eliminación de duplicados, imputación de nulos, creación y discretización de la variable **ROI** y análisis de correlaciones |
| `limpieza adicional.ipynb` | Pasos de limpieza adicionales |
| `Final_MDD.ipynb` | Modelado: balanceo, pipeline de ingeniería de características, validación cruzada, modelos base (Árbol, KNN, Red Neuronal, SVM), ensambles (Random Forest, AdaBoost, Gradient Boosting, votación *soft*), pruebas ANOVA/Tukey e hiperparametrización |
| `data_prepared.csv` | Dataset limpio para el modelado |
| `data_deploy.csv` | Datos de ejemplo para el despliegue |
| `modelo-ensamble-clas.pkl` | Mejor modelo serializado (modelo, variables, `LabelEncoder`, `MinMaxScaler`) |
| `app.py` | App de Streamlit para hacer predicciones |
| `Portfolio Puma MGM.xlsx` | Datos originales |
| `Proyecto final Maria - Cristian.pdf` | Informe final |

## Resultados

En la validación cruzada, **Random Forest** tuvo el mejor F1-macro (≈ 0.878), seguido de AdaBoost (≈ 0.857) y Gradient Boosting (≈ 0.850). El test ANOVA/Tukey mostró diferencias significativas entre los modelos. Los detalles están en `Final_MDD.ipynb` y en el informe PDF.

## App de predicción

La app recibe **Account Name** (firma de abogados), **Provider Client** (proveedor médico), **Subject** (tipo de servicio), **Purchase Amount** y **Bill Amount**, y responde si el caso será `rentable` o `pérdida`.

```bash
python -m venv env
source env/bin/activate          # En Windows: env\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Tecnologías

Python, pandas, NumPy, Matplotlib, scikit-learn, imbalanced-learn, Streamlit.
