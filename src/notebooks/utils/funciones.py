import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

# --- FUNCIONES DE ANÁLISIS ---

# TIPIFICAR VARIABLES
# Default de umbral de categoría entre 15 y 80
def tipifica_variables(df: pd.DataFrame, umbral_categoria: int = 15, umbral_continua: float = 80.0) -> pd.DataFrame:
    """Returns a DataFrame with two columns, "nombre_variable" and "tipo_sugerido,
    with as many rows as columns in the input DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame.
    umbral_categoria (int): Cardinality threshold below which a column 
        is considered categorical. Must be a positive integer. Defaults to 15.
    umbral_continua (float): Cardinality percentage threshold above which 
        a numeric column is considered continuous. Must be between 0 and 100. 
        Defaults to 80.0.

    Returns:
        pd.DataFrame: DataFrame with columns 'nombre_variable' and 'tipo_sugerido', 
        where each row corresponds to a column of the input. Returns None if any 
        input argument is invalid.
    """
    
    # Checks: that it is a DataFrame, that umbral_categoria is a positive integer
    # and that umbral_continua is a float between 0 and 100
    if not isinstance(df, pd.DataFrame):
        print(f"The argument is of type {type(df)} when it should be a DataFrame")
        return None
    if not isinstance(umbral_categoria, int) or umbral_categoria < 0:
        print("umbral_categoria must be a positive integer")
        return None
    if not isinstance(umbral_continua, (int, float)) or umbral_continua < 0 or umbral_continua > 100:
        print("umbral_continua must be a float between 0 and 100")
        return None
    
    # Normalize umbral_continua in case an integer was passed as an argument
    umbral_continua = float(umbral_continua)
    describe = describe_df(df)

    cardinalidad = describe["valores_unicos"]
    porcentaje_cardinalidad = describe["porcentaje_cardinalidad"]

    lista = []
    for clave, valor in cardinalidad.items():
        # Cardinalidad = 2 then Binary
        if valor == 2:
            lista.append("Binaria")
        # Cardinalidad < umbral_categoria then Categórica
        elif valor < umbral_categoria:
            lista.append("Categórica")
        # Cardinalidad >= umbral_categoria AND porcentaje_cardinalidad >= umbral_continua then Numérica Continua
        elif valor >= umbral_categoria and porcentaje_cardinalidad[clave] >= umbral_continua:
            lista.append("Numérica Continua")
        # Cardinalidad >= umbral_categoria AND porcentaje_cardinalidad <= umbral_continua then Numérica Discreta
        elif valor >= umbral_categoria and porcentaje_cardinalidad[clave] <= umbral_continua:
            lista.append("Numérica Discreta")

    columnas = pd.DataFrame({
        "nombre_variable": cardinalidad.index,
        "tipo_sugerido": lista
        })
    
    return columnas

# DESCRIBIR EL DATAFRAME
def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Receives a DataFrame and returns another DataFrame with one row
    per column of the original DataFrame. The index of the result is
    the name of each column.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.

    Returns:
        pd.DataFrame: DataFrame with one row per column of the input, containing 
        'tipo', 'porcentaje_nulos', 'valores_unicos' and 'porcentaje_cardinalidad'. 
        Returns None if the input is not a DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        print(f"The argument is of type {type(df)} when it should be a DataFrame")
        return None
    
    columnas = pd.DataFrame({
        "tipo" : df.dtypes,
        "porcentaje_nulos": round((df.isnull().sum() / len(df)) * 100, 2),
        "valores_unicos": df.nunique(),
        "porcentaje_cardinalidad": round((df.nunique() / len(df)) * 100, 2)
    })

    return columnas

# --- FUNCIONES DE REGRESIÓN CATEGÓRICA---

def get_features_cat_regression( df: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> list:
    """
    Returns all categorical variables significantly correlated to target in dataset.

    By default, level of significance is set to 0.05.

        Parameters:
            df (pd.DataFrame): dataframe where each column is a variable and each row is an observation
            target_col (str):  target variable to compare.
            pvalue (float): level of significance to use for test statistic.

        Returns:
            features (list): list of cualitative features significantly correlated to target.
    """

    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue < 1)):
        print("Set p-value must be a floating point number between 0 and 1.")
        return

    features = []
    df_tipos = tipifica_variables(df)

    if df_tipos.loc[df_tipos["nombre_variable"]==target_col, "tipo_sugerido"].isin(["Binaria", "Categórica"]).iloc[0]:
        print("Target variable must contain numerical data.")
        return

    variables = df_tipos.loc[df_tipos.tipo_sugerido.isin(["Binaria", "Categórica"])]
    for indep_var, tipo in zip(variables["nombre_variable"].tolist(), variables["tipo_sugerido"].tolist()):
        if tipo == "Binaria":
            # Estadístico U de Mann-Whitney
            grupos = [df.loc[df[indep_var]==clase][target_col] for clase in df[indep_var].unique()]
            pval_stat = scipy.stats.mannwhitneyu(grupos[0], grupos[1])[1]

        elif tipo == "Categórica":
            # Estadístico F de ANOVA
            grupos = [df.loc[df[indep_var] == clase, target_col] for clase in df[indep_var].dropna(inplace=False).unique()]
            if 1 in [len(grupo) for grupo in grupos]:
                pval_stat = scipy.stats.kruskal(*grupos)[1] # para poblaciones con tamaño muestral pequeño
            else:
                pval_stat = scipy.stats.f_oneway(*grupos)[1]

        if pval_stat in [np.nan, np.inf]: # Con algunos sets de datos puede devolver np.nan o np.inf
            print(f"Data size for target variable groups by {indep_var} is not sufficient.")
        elif pval_stat <= pvalue:
            features.append(indep_var)

    return features

def plot_features_cat_regression(df: pd.DataFrame, target_col: str = "", columns: list = [], pvalue: float = 0.05, with_individual_plot: bool = False) -> list:
    """
    Plot of target variable classified by given categorical column variables.

    Default level of significance is set to 0.05.

        Parameters:
            df (pd.DataFrame): dataframe where each column is a variable and each row is an observation
            target_col (str):  target variable to compare.
            columns (list of strings): variables within df to use.
            with_individual_plot (bool): if True, draws one plot per categorical variable.
            pvalue (float): level of significance to use for test statistic.

        Returns:
            features (list): list of cualitative features used in plot.
    """
    # Comprobaciones
    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue < 1)):
        print("P-value must be a floating point number between 0 and 1.")
        return

    df_tipos = tipifica_variables(df)
    if df_tipos.loc[df_tipos["nombre_variable"]==target_col, "tipo_sugerido"].isin(["Binaria", "Categórica"]).iloc[0]:
        print("Target variable must contain numerical data.")
        return
    
    features = columns

    # Plot de target según las features
    if with_individual_plot:
        for var in features:
            categorias = df[var].unique().tolist()
            plt.figure(figsize=(8, 6))

            sns.histplot(data=df, x=target_col, hue=var, hue_order=categorias, bins=20, kde=True, stat="count", alpha=0.4,)

            plt.title(f"{target_col} según {var}: "f"{', '.join(map(str, categorias))}")    # map convierte bools y numeros en str
            plt.xlabel(target_col)
            plt.show()

    else:
        n = len(features)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*min(2, ncols), 5*nrows))
        axes = np.array(axes).reshape(-1)

        for i, var in enumerate(features):
            ax = axes[i]
            tabla = pd.crosstab(df[var], df[target_col]) # Tabla de contingencia
            ylabel = "Frecuencia absoluta"

            sns.histplot(data=df, x=target_col, hue=var, hue_order=df[var].unique().tolist(), bins=20, kde=True, stat="count", alpha=0.4, ax=ax)

            ax.set_title(f"{target_col} según {var}")
            ax.set_xlabel(var)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    return features

# --- FUNCIONES DE REGRESIÓN NUMÉRICA---

def plot_features_num_regression(df: pd.DataFrame, target_col: str = "", columns: list = [], umbral_corr: float = 0., pvalue: float = 1.) -> list:
    """
    Plot of target variable classified by given numerical column variables.

    Default level of significance is set to 1 and correlation threshold is 0 (all variables are considered).

        Parameters:
            df (pd.DataFrame): dataframe where each column is a variable and each row is an observation
            target_col (str):  target variable to compare.
            columns (list of strings): variables within df to use.
            umbral_corr (float): correlation threshold to consider a variable statistically significant.
            pvalue (float): level of significance to use for test statistic.

        Returns:
            features (list): list of cuantitative features used in plot.
    """
    # Comprobaciones
    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue <= 1)):    # sin permitir none
        print("Set p-value must be a float between 0 and 1.")
        return
    if not (isinstance(umbral_corr, float) and (0 <= umbral_corr and umbral_corr < 1)):
        print("Set correlation threshold must be a float between 0 and 1.")
        return

    features = columns

    # Plot de target según las features
    grupos = [features[i:i + 5] for i in range(0, len(features), 5)]

    for var_set in grupos:
        var_set.insert(0, target_col)
        sns.pairplot(df, vars=var_set)
        plt.show()

    return features

# -- FUNCIONES EDA --

def diagramas_barras (df, columnas, etiquetas=False, relativo=False, guardar=False):
    n = len(columnas)
    fig, axes = plt.subplots(nrows=int(np.ceil(n/3)), ncols=3, figsize=(15, 1.5*n))

    if n == 1:
        axes = [axes]

    for i, columna in enumerate(columnas):
        fila = i // 3
        columna_grafico = i % 3
        ax = axes[fila][columna_grafico]

        if df[columna].dtype == bool:
            frecuencias_abs = (df[columna].map({False:"False", True:"True"})).value_counts()    # matplotlib interpreta booleanos como 1 y 0
        else:
            frecuencias_abs = df[columna].value_counts()

        if relativo:
            frecuencias = frecuencias_abs / frecuencias_abs.sum() * 100
            ax.set_ylabel("Frecuencia relativa (%)")
        else:
            frecuencias = frecuencias_abs
            ax.set_ylabel("Frecuencia absoluta")

        colores = plt.cm.Blues(np.linspace(0.35, 0.85, len(frecuencias)))
        barras = ax.bar(frecuencias.index, frecuencias.values, color = colores,  edgecolor = "black")

        ax.set_title(f"Diagrama de barras de '{columna}'")
        ax.set_xlabel(columna)
        ax.grid(axis="y", alpha=0.3)

        if etiquetas:
            for barra, frec in zip(barras, frecuencias.values):
                x = barra.get_x() + barra.get_width() / 3
                y = barra.get_height()

                if relativo:
                    texto = f"{frec:.2f}%"
                else:
                    texto = f"{frec}"

                ax.text(x, y, texto, ha="center", va="bottom", fontsize=9)
        plt.setp(ax.get_xticklabels(), rotation=60, ha="right")

    plt.tight_layout()

    if guardar: plt.savefig(f"./src/img/diagrama_barras_{"_".join(columnas)}.png", dpi=300, bbox_inches="tight") 
    plt.show()
    return


def boxplot_histograma(df, columna, relleno="#89D2A2", color_linea = "green", guardar=False):
    # Cuartiles para el boxplot
    q1 = np.percentile(df[columna], 25)
    q2 = np.percentile(df[columna], 50)
    q3 = np.percentile(df[columna], 75)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    # BOXPLOT
    sns.boxplot(y=df[columna], ax=axes[0], color=relleno, width=0.25)

    axes[0].set_title(f"Diagrama de caja de {columna}")
    axes[0].set_ylabel(columna)

    axes[0].text(0.15, q1, f"Q1 = {q1:.2f}", va="center")
    axes[0].text(0.15, q2, f"Q2 = {q2:.2f}", va="center")
    axes[0].text(0.15, q3, f"Q3 = {q3:.2f}", va="center")

    # HISTOGRAMA Y FUNCION DE DENSIDAD
    sns.histplot(df[columna], bins="auto", ax=axes[1], color=relleno, edgecolor="black")

    media = np.mean(df[columna])
    axes[1].axvline(media, color="red", linestyle="--", linewidth=1, label=f"Media = {media:.2f}")
    axes[1].legend()

    sns.kdeplot(df[columna], ax=axes[1].twinx(), color=color_linea, linewidth=2)

    axes[1].set_title(f"Histograma de {columna}")
    axes[1].set_xlabel(columna)
    axes[1].set_ylabel("Frecuencia")

    plt.tight_layout()
    if guardar: plt.savefig(f"./src/img/boxpot_histograma_{columna}.png", dpi=300, bbox_inches="tight") 
    plt.show()
    return


def histograma_por_categorias(df, var_cuantitativa, var_cualitativa, bins=20, max_cat_por_grafico = 3,  estadistico = "count", guardar=False):
    categorias = df[var_cualitativa].unique()
    grupos = [categorias[i:i + max_cat_por_grafico] for i in range(0, len(categorias), max_cat_por_grafico)]
    
    fig, axes = plt.subplots(nrows=(int(np.ceil(len(grupos)/2))), ncols=min(2, len(grupos)), figsize=(7*min(2, len(grupos)), 3 * len(grupos)))
    axes = np.array(axes).reshape(-1)   # Para recorrer con un for 

    for ax, grupo in zip(axes, grupos):

        sns.histplot(
            data=df,
            x=var_cuantitativa,
            hue=var_cualitativa,
            hue_order=grupo,  # Limitar a las variables en el grupo actual
            bins=bins,
            kde=True,
            stat=estadistico,   # "density" para valores relativos
            common_norm=True,  # Si se usa density, normaliza respecto del dataset completo
            alpha=0.4,
            ax=ax
        )
        ax.set_title(f"{var_cuantitativa} según {var_cualitativa}: "f"{', '.join(map(str, grupo))}")    # map convierte bools y numeros en str
        ax.set_xlabel(var_cuantitativa)

    if guardar: plt.savefig(f"./src/img/histograma_{var_cuantitativa}_{var_cualitativa}.png", dpi=300, bbox_inches="tight") 
    plt.show()
    return


def comparar_cualitativas(df, var_principal, *vars_comparacion, ncols = 2, relativo=False, apilado=False, etiquetas=True, paleta="mako", guardar=False):
    n = len(vars_comparacion)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*min(2, ncols), 6*nrows))
    axes = np.array(axes).reshape(-1)

    for i, var in enumerate(vars_comparacion):
        ax = axes[i]
        tabla = pd.crosstab(df[var], df[var_principal]) # Tabla de contingencia

        if relativo:
            tabla = tabla.div(tabla.sum(axis=1), axis=0) * 100
            ylabel = "Frecuencia relativa (%)"
        else:
            ylabel = "Frecuencia absoluta"

        if apilado:
            colores = sns.color_palette(paleta, n_colors=tabla.shape[1])
            tabla.plot( kind="bar", stacked=True, ax=ax, color=colores)

            if etiquetas:
                for contenedor in ax.containers:
                    ax.bar_label(contenedor,fmt="%.1f" if relativo else "%.0f", label_type="center")

        else:
            tabla_larga = tabla.reset_index().melt(id_vars=var, var_name=var_principal, value_name="frecuencia")    # Transformar a formato columnas

            sns.barplot(data=tabla_larga, x=var, y="frecuencia", hue=var_principal, ax=ax, palette=paleta)

            if etiquetas:
                for contenedor in ax.containers:
                    ax.bar_label(contenedor, fmt="%.1f" if relativo else "%.0f", padding=2)

        ax.set_title(f"{var_principal} según {var}")
        ax.set_xlabel(var)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=45)
        ax.legend(title=var_principal)

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if guardar: plt.savefig(f"./src/img/diagrama_barras_{var_principal}_{"_".join(vars_comparacion)}.png", dpi=300, bbox_inches="tight") 
    plt.show()
    return


def diagrama_dispersion(df, variable_x, variable_y, guardar=False):
    correlacion = df[variable_x].corr(df[variable_y])

    plt.figure(figsize=(8, 6))

    sns.scatterplot(data=df, x=variable_x, y=variable_y)

    plt.title(f"Diagrama de dispersión: {variable_x} vs {variable_y}\n"
        f"Coeficiente de correlación: {correlacion:.4f}")

    plt.xlabel(variable_x)
    plt.ylabel(variable_y)
    plt.grid(True, alpha=0.3)

    if guardar: plt.savefig(f"./src/img/diag_dispersion_{variable_x}_{variable_y}.png", dpi=300, bbox_inches="tight")
    plt.show()
    return


# ---- FUNCIONES DE PREPROCESADO ----

def limpiar_ausentes_estructurales(df, columna, es_unifamiliar):
    """
    Crea una "columna_limpio" de una columna con valores ausentes.
    Los NaN se recodifican segun el tipo de vivienda:
      - NO_APLICA para chalets y casas
      - DESCONOCIDO para el resto (el dato falta, pero deberia existir)
    Conserva la columna original intacta.
    """
    columna_limpia = columna + "_limpio"
    df[columna_limpia] = df[columna]
    df.loc[es_unifamiliar & df[columna].isna(), columna_limpia] = "NO_APLICA"
    df.loc[~es_unifamiliar & df[columna].isna(), columna_limpia] = "DESCONOCIDO"
    return df