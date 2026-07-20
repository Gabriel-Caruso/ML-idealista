# ML-idealista
Proyecto ML predicción de precios de vivienda a la venta en Madrid.

**Autores**: Ana Manzanares y Ramiro Caruso

### Objetivo
Predecir el precio estimado actual de una vivienda a la venta en madrid a partir de las características publicadas en el anuncio. El objetivo es disponer de un estimador de referencia consistente que oriente sobre el precio del mercado sin necesidad de una tasación formal.

### Dataset
**Fuente**: [Dataset de Kaggle](https://www.kaggle.com/datasets/fjcob1/idealista-madrid) con un listado de anuncios de vivienda en venta en Madrid.
**Tamaño:** 11.836 anuncios.
**Columnas:** 14: provincia, zona, titulo, PrecioActual, PrecioAnterior, metros, habitaciones, ascensor, localizacion, planta, baños, tags, descripcion, Enlace.
**Target:** PrecioActual

Los datos vienen ne crudo en incluyen valores centinela, duplicados ocultos y campos de texto libre que requieren limpieza.

### Estructura del repositorio
En la raíz está main.ipynb, el notebook que reproduce el proyecto entero de principio a fin. El resto del código vive dentro de src/, organizado así:

src/data_sample/ — una muestra de los datos.
src/img/ — las figuras que genera el proyecto.
src/models/ — el modelo final ya entrenado y el estudio de Optuna (guardado en SQLite).
src/notebooks/ — los notebooks de trabajo, uno por fase (preprocesado, EDA y modelado).
src/utils/ — funciones.py, con limpiar_ausentes y demás utilidades.

### Metodología
#### 1. Preprocesado y limpieza
- Tratamiento de valores centinela y clasificación de los NaNs estructurales y los verdaderamente faltantes
- Eliminar los duplicados sobre una clave de 9 campos (titulo + PrecioActual + metros + habitaciones + planta + baños + ascensor + localizacion + zona).
- Normalización de planta, limpieza en baños y habitaciones.
- Creación de flags a partir de las etiquetas de tags que resultan relevantes.

#### 2. EDA
- Análisis de la target y las features significativas.

#### 3. Modelado y optimización
- Split 80/20 estratificado por zona
- Modelos comparados: Regresión lineal como baseline, XGBoost, LightGBM y CatBoost
- Métrica: MAE
- Optimización: Optuna sobre CatBoost. Estudio guardado en SQLite.
- Modelo ganador: CatBoost

### Resultados

Comparativa de modelos (MAE en test)
| Modelo                  | MAE (test) |
|-------------------------|-----------:|
| Regresión lineal (base) | 358.708 €  |
| LightGBM                | 214.886 €  |
| XGBoost                 | 214.581 €  |
| CatBoost (ganador)      | 183.771 €  |

Evaluación final contra el test
| Métrica | Valor     |
|---------|----------:|
| MAE     | 181.256 € |
| RMSE    | 443.717 € |
| R²      | 0.861     |

Error por tramos de precio
| Tramo de precio (€)    | Precio mediano | MAE       | Error relativo |
|------------------------|---------------:|----------:|---------------:|
| 35.000 – 250.000       | 188.604 €      | 30.977 €  | ~16 %          |
| 250.000 – 435.360      | 335.000 €      | 48.994 €  | ~15 %          |
| 435.360 – 835.600      | 599.900 €      | 93.136 €  | ~16 %          |
| 835.600 – 1.490.000    | 1.150.000 €    | 177.660 € | ~15 %          |
| 1.490.000 – 13.000.000 | 2.350.000 €    | 558.820 € | ~24 %          |
| GLOBAL                 | 599.900 €      | 181.256 € | —              |

## Cómo ejecutar

**Clonar el repositorio:**
git clone https://github.com/Gabriel-Caruso/ML-idealista.git
cd ML-idealista

**Crear entorno virtual** 
python -m venv .venv
Source .venv/bin/activate        # En Windows: .venv\Scripts\activate

**Instalar dependencias**
pip install -r requirements.txt
