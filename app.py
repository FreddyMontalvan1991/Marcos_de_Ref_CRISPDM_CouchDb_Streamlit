import streamlit as st
import pandas as pd
import numpy as np
import io
from datetime import datetime
import requests
from requests.auth import HTTPBasicAuth
import base64
import joblib
import warnings
warnings.filterwarnings("ignore")

# sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# visualización
import plotly.express as px

# --------------------------
# Configuración general
# --------------------------
st.set_page_config(layout="wide", page_title="CRISP-DM — Graiman (Regresión)")

NOMBRE_ARCHIVO = "Freddy_M_BI_Dataset_Clientes_Marketing_Graiman_4000_Registros.csv"

URL_COUCHDB = "http://127.0.0.1:5984"
USUARIO_COUCHDB = "Admin"
CLAVE_COUCHDB = "12345"
NOMBRE_DB = "graiman_crispdm"

# --------------------------
# Funciones CouchDB (HTTP)
# --------------------------
def solicitud_couchdb(metodo, ruta, **kwargs):
    url = URL_COUCHDB.rstrip("/") + "/" + ruta.lstrip("/")
    auth = HTTPBasicAuth(USUARIO_COUCHDB, CLAVE_COUCHDB)
    resp = requests.request(metodo, url, auth=auth, **kwargs)
    try:
        contenido = resp.json()
    except Exception:
        contenido = resp.text
    return resp.status_code, contenido

def asegurar_base(nombre_db=NOMBRE_DB):
    status, resp = solicitud_couchdb("PUT", f"/{nombre_db}")
    return status in (201, 412)

def guardar_documento_id(nombre_db, id_doc, documento):
    return solicitud_couchdb("PUT", f"/{nombre_db}/{id_doc}", json=documento)

def guardar_documento(nombre_db, documento):
    return solicitud_couchdb("POST", f"/{nombre_db}", json=documento)

def obtener_documento(nombre_db, id_doc):
    return solicitud_couchdb("GET", f"/{nombre_db}/{id_doc}")

def consultar_todos_documentos(nombre_db, limite=500):
    status, resp = solicitud_couchdb("GET", f"/{nombre_db}/_all_docs?include_docs=true&limit={limite}")
    if status == 200 and isinstance(resp, dict):
        filas = resp.get("rows", [])
        docs = [r.get("doc") for r in filas]
        return docs
    return []

def guardar_modelo_en_couch(nombre_db, objeto_modelo, id_doc, meta=None):
    bio = io.BytesIO()
    joblib.dump(objeto_modelo, bio)
    bio.seek(0)
    b64 = base64.b64encode(bio.read()).decode('utf-8')
    documento = {
        "_id": id_doc,
        "type": "model",
        "payload_b64": b64,
        "meta": meta or {},
        "timestamp": datetime.utcnow().isoformat()
    }
    return guardar_documento_id(nombre_db, id_doc, documento)

def cargar_modelo_de_couch(nombre_db, id_doc):
    status, doc = obtener_documento(nombre_db, id_doc)
    if status == 200 and isinstance(doc, dict) and doc.get("payload_b64"):
        b = base64.b64decode(doc["payload_b64"].encode('utf-8'))
        bio = io.BytesIO(b)
        return joblib.load(bio)
    return None

def _serializar_valor_para_json(v):
    """
    Convierte valores no serializables (Timestamp, datetime, numpy types) a representaciones JSON-friendly.
    """
    # pandas Timestamp
    if isinstance(v, pd.Timestamp):
        return v.isoformat()
    # datetime.datetime
    if isinstance(v, datetime):
        return v.isoformat()
    # numpy scalar types
    if isinstance(v, (np.integer, np.int64, np.int32)):
        return int(v)
    if isinstance(v, (np.floating, np.float64, np.float32)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    # numpy arrays -> convertir a lista
    if isinstance(v, (np.ndarray,)):
        try:
            return v.tolist()
        except Exception:
            return str(v)
    # otros tipos pandas (Period, Timedelta)
    try:
        # intentamos convertir con isoformat si existe
        if hasattr(v, "isoformat"):
            return v.isoformat()
    except Exception:
        pass
    # por defecto, devolver tal cual si es serializable, o string
    try:
        # chequeo simple: intentar serializar a JSON mediante str para fallback
        return v
    except Exception:
        return str(v)

def guardar_dataframe_como_docs(nombre_db, df, prefijo="data"):
    """
    Guarda cada fila del dataframe como documento independiente en CouchDB.
    Convierte valores no serializables (Timestamps, numpy types) antes de enviar.
    Devuelve número de documentos guardados.
    """
    asegurar_base(nombre_db)
    guardados = 0
    for i, fila in df.reset_index(drop=True).iterrows():
        doc = {}
        for k, v in fila.to_dict().items():
            doc[k] = _serializar_valor_para_json(v)
        doc["_id"] = f"{prefijo}_{i}_{int(datetime.utcnow().timestamp())}"
        doc["type"] = "record"
        status, _ = guardar_documento_id(nombre_db, doc["_id"], doc)
        if status in (201, 202):
            guardados += 1
    return guardados

# --------------------------
# Utilidades de datos
# --------------------------
def sanitizar_nombres(columnas):
    nuevos = []
    for c in columnas:
        s = str(c).strip()
        s = s.replace(" ", "_").replace("%", "pct").replace("/", "_").replace("-", "_")
        # arreglos básicos de caracteres corruptos
        s = s.replace("�", "a").replace("á", "a").replace("Á", "A").replace("é", "e").replace("ó", "o").replace("ñ", "n").replace("�", "a")
        nuevos.append(s)
    return nuevos

@st.cache_data
def cargar_csv_local(ruta=NOMBRE_ARCHIVO):
    try:
        df = pd.read_csv(ruta, encoding="latin1")
    except Exception:
        df = pd.read_csv(ruta, encoding="utf-8", errors="ignore")
    df.columns = sanitizar_nombres(df.columns)
    return df

def resumen_dataframe(df):
    tabla = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "nulos": df.isnull().sum(),
        "cardinalidad": df.nunique()
    })
    return tabla

def limpieza_basica(df):
    df = df.copy()
    for c in df.select_dtypes(include=['object']).columns:
        df[c] = df[c].astype(str).str.strip()
    for c in df.columns:
        if "fecha" in c.lower() or "date" in c.lower() or "registro" in c.lower():
            try:
                df[c] = pd.to_datetime(df[c], errors="coerce")
            except Exception:
                pass
    for c in df.columns:
        if df[c].dtype == object:
            muestra = df[c].dropna().astype(str).head(10)
            parecido_num = True
            for v in muestra:
                v2 = str(v).replace(",", "").replace(" ", "")
                try:
                    float(v2)
                except Exception:
                    parecido_num = False
                    break
            if parecido_num:
                df[c] = df[c].astype(str).str.replace(",", "").str.replace(" ", "")
                df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.drop_duplicates()
    return df

def construir_pipeline_preproc(df, excluir=None):
    excluir = excluir or []
    columnas_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    columnas_numericas = [c for c in columnas_numericas if c not in excluir]
    columnas_datetime = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]
    columnas_categoricas = [c for c in df.columns if c not in columnas_numericas + columnas_datetime and c not in excluir]

    pipeline_num = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # OneHotEncoder compatible con múltiples versiones de scikit-learn
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    pipeline_cat = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", ohe)
    ])
    transformers = []
    if columnas_numericas:
        transformers.append(("num", pipeline_num, columnas_numericas))
    if columnas_categoricas:
        transformers.append(("cat", pipeline_cat, columnas_categoricas))
    preproc = ColumnTransformer(transformers, remainder="drop")
    return preproc, columnas_numericas, columnas_categoricas, columnas_datetime

def dataframe_preparado(preproc, df):
    arr = preproc.transform(df)
    col_num = []
    col_cat = []
    for t in preproc.transformers_:
        nombre, trans, cols = t
        if nombre == "num":
            col_num = cols
        elif nombre == "cat":
            try:
                oh = trans.named_steps["onehot"]
                try:
                    names = oh.get_feature_names_out(cols).tolist()
                except Exception:
                    # fallback: generar nombres simples
                    names = []
                    for c in cols:
                        names.append(f"{c}_ohe")
                col_cat = names
            except Exception:
                col_cat = []
    nombres = list(col_num) + list(col_cat)
    try:
        df_prep = pd.DataFrame(arr, columns=nombres, index=df.index)
    except Exception:
        df_prep = pd.DataFrame(arr, index=df.index)
    return df_prep

# --------------------------
# Interfaz Streamlit
# --------------------------
st.title("CRISP-DM — Graiman (Regresión Lineal)")

st.markdown("""
Aplicación educativa que implementa CRISP-DM (simplificado) y entrena un modelo de regresión lineal
para predecir **Ingresos_Mensuales**. Con persistencia opcional en CouchDB.
""")

# Cargar CSV
with st.spinner("Cargando dataset..."):
    try:
        df_raw = cargar_csv_local()
    except FileNotFoundError:
        st.error(f"No se encontró el archivo '{NOMBRE_ARCHIVO}'. Colócalo en la misma carpeta que este .py.")
        st.stop()
    except Exception as e:
        st.error(f"Error cargando CSV: {e}")
        st.stop()

# Inicializar sesión
if "df_raw" not in st.session_state:
    st.session_state.df_raw = df_raw
if "df_limpio" not in st.session_state:
    st.session_state.df_limpio = None
if "modelo_regresion" not in st.session_state:
    st.session_state.modelo_regresion = None
if "modelos_en_db" not in st.session_state:
    st.session_state.modelos_en_db = []

# Tabs (pestañas) CRISP-DM
p1, p2, p3, p4, p5, p6 = st.tabs([
    "1. Comprensión del negocio",
    "2. Comprensión de los datos",
    "3. Preparación de los datos",
    "4. Modelado (Regresión)",
    "5. Evaluación",
    "6. Despliegue / Dashboard"
])

# --------------------------
# 1. Comprensión del negocio
# --------------------------
with p1:
    st.header("1. Comprensión del negocio")
    st.markdown("""
    Contexto: Graiman (Cuenca, Ecuador). Analizamos ventas, marketing, inventario y producción.
    Objetivo de ejemplo: predecir **Ingresos_Mensuales** para apoyar decisiones de inventario y marketing.
    """)
    objetivo = st.text_area("Describe objetivo de negocio (opcional)", height=120)
    if st.button("Guardar objetivo en CouchDB"):
        if objetivo.strip() == "":
            st.warning("Escribe un objetivo antes de guardar.")
        else:
            asegurar_base(NOMBRE_DB)
            doc = {"type": "business_goal", "goal": objetivo, "timestamp": datetime.utcnow().isoformat()}
            status, resp = guardar_documento(NOMBRE_DB, doc)
            if status in (201, 202):
                st.success("Objetivo guardado en CouchDB.")
            else:
                st.error(f"Error guardando objetivo: {resp}")

# --------------------------
# 2. Comprensión de los datos
# --------------------------
with p2:
    st.header("2. Comprensión de los datos")
    st.subheader("Vista previa")
    st.dataframe(st.session_state.df_raw.head(10))
    st.subheader("Información general")
    st.write("Registros:", st.session_state.df_raw.shape[0], " | Columnas:", st.session_state.df_raw.shape[1])
    st.subheader("Tipos, nulos y cardinalidad")
    st.dataframe(resumen_dataframe(st.session_state.df_raw))
    st.subheader("Estadísticas descriptivas")
    st.dataframe(st.session_state.df_raw.describe(include='all').T)

    st.subheader("Gráficos exploratorios")
    cols_numericas = st.session_state.df_raw.select_dtypes(include=[np.number]).columns.tolist()
    cols_categoricas = st.session_state.df_raw.select_dtypes(exclude=[np.number]).columns.tolist()

    if cols_numericas:
        var_hist = st.selectbox("Histograma - variable numérica", cols_numericas, key="hist")
        fig_hist = px.histogram(st.session_state.df_raw, x=var_hist, nbins=50, title=f"Histograma: {var_hist}")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No se detectaron columnas numéricas para histogramas.")

    if cols_categoricas:
        var_bar = st.selectbox("Barras - variable categórica", cols_categoricas, key="bar")
        vc = st.session_state.df_raw[var_bar].value_counts().reset_index()
        # renombrado seguro para evitar duplicados
        vc.columns = [var_bar, "frecuencia"]
        fig_bar = px.bar(vc.head(30), x=var_bar, y="frecuencia", title=f"Frecuencia: {var_bar}")
        st.plotly_chart(fig_bar, use_container_width=True)

    # serie temporal si existen columnas esperadas
    if ("Fecha_Registro" in st.session_state.df_raw.columns) and ("Ingresos_Mensuales" in st.session_state.df_raw.columns):
        st.subheader("Serie temporal de Ingresos Mensuales")
        df_time = st.session_state.df_raw.copy()
        df_time["Fecha_Registro"] = pd.to_datetime(df_time["Fecha_Registro"], errors="coerce")
        df_time = df_time.dropna(subset=["Fecha_Registro"])
        monthly = df_time.groupby(pd.Grouper(key="Fecha_Registro", freq="M"))["Ingresos_Mensuales"].sum().reset_index()
        fig_time = px.line(monthly, x="Fecha_Registro", y="Ingresos_Mensuales", title="Ingresos Mensuales por mes")
        st.plotly_chart(fig_time, use_container_width=True)
    else:
        st.info("No hay columnas Fecha_Registro e Ingresos_Mensuales juntas para serie temporal.")

    if cols_numericas and len(cols_numericas) >= 2:
        st.subheader("Correlación (numéricas)")
        corr = st.session_state.df_raw[cols_numericas].corr()
        fig_corr = px.imshow(corr, title="Matriz de correlación")
        st.plotly_chart(fig_corr, use_container_width=True)

# --------------------------
# 3. Preparación de los datos
# --------------------------
with p3:
    st.header("3. Preparación de los datos")
    st.markdown("Aplicar limpieza automática y operaciones manuales opcionales.")

    if st.session_state.df_limpio is None:
        st.session_state.df_limpio = limpieza_basica(st.session_state.df_raw)
        st.success("Limpieza básica aplicada (trim, tipos, duplicados).")

    st.subheader("Vista previa dataset limpio")
    st.dataframe(st.session_state.df_limpio.head(20))
    st.write("Forma:", st.session_state.df_limpio.shape)

    st.subheader("Operaciones manuales")
    columnas_a_eliminar = st.multiselect("Eliminar columnas (opcional)", options=st.session_state.df_limpio.columns.tolist())
    if st.button("Eliminar columnas seleccionadas"):
        st.session_state.df_limpio = st.session_state.df_limpio.drop(columns=columnas_a_eliminar, errors="ignore")
        st.success("Columnas eliminadas.")

    if st.button("Rellenar nulos (num: mediana, cat: 'missing')"):
        dfp = st.session_state.df_limpio
        for c in dfp.select_dtypes(include=[np.number]).columns:
            dfp[c] = dfp[c].fillna(dfp[c].median())
        for c in dfp.select_dtypes(exclude=[np.number]).columns:
            dfp[c] = dfp[c].fillna("missing")
        st.session_state.df_limpio = dfp
        st.success("Nulos rellenados.")

    st.subheader("Guardar dataset limpio en CouchDB")
    prefijo_docs = st.text_input("Prefijo para documentos", value="graiman_clean")
    if st.button("Guardar dataset limpio (fila por documento) en CouchDB"):
        asegurar_base(NOMBRE_DB)
        guardados = guardar_dataframe_como_docs(NOMBRE_DB, st.session_state.df_limpio, prefijo_docs)
        st.success(f"Guardados {guardados} documentos con prefijo '{prefijo_docs}'.")

    st.subheader("Resumen del dataset limpio")
    st.dataframe(st.session_state.df_limpio.describe(include='all').T)

# --------------------------
# 4. Modelado (Regresión)
# --------------------------
with p4:
    st.header("4. Modelado — Regresión Lineal")
    st.markdown("Entrenaremos un modelo de regresión lineal para predecir Ingresos_Mensuales.")

    if st.session_state.df_limpio is None:
        st.warning("Prepara primero los datos en la pestaña 3.")
        st.stop()

    df = st.session_state.df_limpio.copy()

    # Target fijo según tu instrucción
    target = "Ingresos_Mensuales"
    if target not in df.columns:
        st.error(f"La columna target '{target}' no existe en el dataset. Selecciona otro target o revisa el CSV.")
        numeric_targets = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_targets:
            alt = st.selectbox("Selecciona target alternativo", numeric_targets)
            target = alt
        else:
            st.stop()

    st.write(f"Target seleccionado para regresión: {target}")

    # Selección de features (por defecto, todas numéricas excepto target)
    caracteristicas = st.multiselect(
        "Selecciona features (vacío = todas las numéricas menos target)",
        options=[c for c in df.columns if c != target],
        default=[c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target][:8]
    )

    if len(caracteristicas) == 0:
        caracteristicas = [c for c in df.select_dtypes(include=[np.number]).columns.tolist() if c != target]

    st.write("Número de features seleccionadas:", len(caracteristicas))

    test_size_pct = st.slider("Tamaño del set de prueba (%)", 10, 50, 20)

    if st.button("Entrenar regresión lineal"):
        # preparar datos
        df_reg = df[[target] + caracteristicas].dropna()
        if df_reg.shape[0] < 10:
            st.error("No hay suficientes registros completos para entrenar (se requieren al menos 10).")
        else:
            X = df_reg[caracteristicas]
            y = df_reg[target]

            preproc, cols_num, cols_cat, cols_dt = construir_pipeline_preproc(X)
            try:
                preproc.fit(X)
            except Exception as e:
                st.warning(f"Preprocesador tuvo problemas al ajustar: {e}")
            X_preparado = dataframe_preparado(preproc, X)

            X_train, X_test, y_train, y_test = train_test_split(X_preparado, y, test_size=test_size_pct/100.0, random_state=42)
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)

            # RMSE compatible con versiones de sklearn (sin usar argumento 'squared')
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            st.write(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")

            df_res = pd.DataFrame({"real": y_test, "predicho": y_pred})
            fig_res = px.scatter(df_res, x="real", y="predicho", trendline="ols", title="Real vs Predicho")
            st.plotly_chart(fig_res, use_container_width=True)

            # Guardar modelo en sesión
            st.session_state.modelo_regresion = {"modelo": modelo, "preproc": preproc, "target": target, "features": caracteristicas}
            st.success("Modelo entrenado y guardado en sesión.")

            if st.button("Guardar modelo de regresión en CouchDB"):
                asegurar_base(NOMBRE_DB)
                id_modelo = f"regresion_{int(datetime.utcnow().timestamp())}"
                status, resp = guardar_modelo_en_couch(NOMBRE_DB, st.session_state.modelo_regresion, id_modelo, meta={"target": target, "features": caracteristicas})
                if status in (201, 202):
                    st.success(f"Modelo guardado en CouchDB con id: {id_modelo}")
                else:
                    st.error(f"Error guardando modelo: {resp}")

# --------------------------
# 5. Evaluación
# --------------------------
with p5:
    st.header("5. Evaluación")
    st.markdown("Evaluación de modelos guardados y métricas rápidas si hay modelo en sesión.")

    asegurar_base(NOMBRE_DB)
    docs = consultar_todos_documentos(NOMBRE_DB, limite=1000)
    modelos_db = [d for d in docs if isinstance(d, dict) and d.get("type") == "model"]
    st.write(f"Modelos guardados en CouchDB: {len(modelos_db)}")

    if modelos_db:
        ids = [m["_id"] for m in modelos_db]
        seleccionado = st.selectbox("Selecciona modelo para cargar", ids)
        if st.button("Cargar modelo"):
            cargado = cargar_modelo_de_couch(NOMBRE_DB, seleccionado)
            if cargado is None:
                st.error("No se pudo cargar el modelo.")
            else:
                st.session_state.modelo_cargado = cargado
                st.success("Modelo cargado en sesión.")
                st.write("Metadatos:", cargado.get("meta") if isinstance(cargado, dict) and cargado.get("meta") else "Sin metadatos.")

    if "modelo_regresion" in st.session_state and st.session_state.modelo_regresion is not None:
        st.subheader("Evaluación del modelo en sesión")
        bundle = st.session_state.modelo_regresion
        modelo = bundle["modelo"]
        preproc = bundle["preproc"]
        target = bundle["target"]
        features = bundle["features"]
        df_eval = st.session_state.df_limpio.copy().dropna(subset=features + [target])
        if df_eval.shape[0] > 20:
            muestra = df_eval.sample(n=min(200, df_eval.shape[0]), random_state=42)
            X_eval = dataframe_preparado(preproc, muestra[features])
            y_eval = muestra[target]
            y_pred = modelo.predict(X_eval)
            rmse = np.sqrt(mean_squared_error(y_eval, y_pred))
            mae = mean_absolute_error(y_eval, y_pred)
            r2 = r2_score(y_eval, y_pred)
            st.write("Evaluación en muestra:")
            st.write(f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | R2: {r2:.4f}")
        else:
            st.info("No hay suficientes registros limpios para evaluar el modelo en sesión.")

# --------------------------
# 6. Despliegue / Dashboard
# --------------------------
with p6:
    st.header("6. Despliegue / Dashboard")
    st.markdown("KPIs, visualizaciones y módulo de predicción usando el modelo guardado en CouchDB o en sesión.")

    df_use = st.session_state.df_limpio.copy() if (st.session_state.df_limpio is not None) else st.session_state.df_raw.copy()

    st.subheader("KPIs")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total registros", df_use.shape[0])
    if "Ingresos_Mensuales" in df_use.columns:
        tot = df_use["Ingresos_Mensuales"].sum()
        avg = df_use["Ingresos_Mensuales"].mean()
        c2.metric("Total Ingresos", f"{tot:,.2f}")
        c3.metric("Ingreso promedio", f"{avg:,.2f}")
    else:
        c2.metric("Total Ingresos", "N/A")
        c3.metric("Ingreso promedio", "N/A")
    c4.metric("Columnas", df_use.shape[1])

    st.subheader("Top regiones (si existe columna de región)")
    region_col = None
    for c in df_use.columns:
        if "region" in c.lower() or "regi" in c.lower():
            region_col = c
            break
    if region_col:
        vc_region = df_use[region_col].value_counts().reset_index()
        vc_region.columns = [region_col, "frecuencia"]
        fig_region = px.bar(vc_region.head(10), x=region_col, y="frecuencia", title="Top regiones")
        st.plotly_chart(fig_region, use_container_width=True)
    else:
        st.info("No se encontró columna de región para mostrar top regiones.")

    if "Segmento_Cliente" in df_use.columns:
        vc_seg = df_use["Segmento_Cliente"].value_counts().reset_index()
        vc_seg.columns = ["Segmento", "frecuencia"]
        fig_seg = px.pie(vc_seg.head(10), names="Segmento", values="frecuencia", title="Segmento Cliente (Top)")
        st.plotly_chart(fig_seg, use_container_width=True)

    st.subheader("Serie temporal (Ingresos por mes)")
    if ("Fecha_Registro" in df_use.columns) and ("Ingresos_Mensuales" in df_use.columns):
        df_tmp = df_use.copy()
        df_tmp["Fecha_Registro"] = pd.to_datetime(df_tmp["Fecha_Registro"], errors="coerce")
        df_tmp = df_tmp.dropna(subset=["Fecha_Registro"])
        mensual = df_tmp.groupby(pd.Grouper(key="Fecha_Registro", freq="M"))["Ingresos_Mensuales"].sum().reset_index()
        fig_mensual = px.line(mensual, x="Fecha_Registro", y="Ingresos_Mensuales", title="Ingresos Mensuales")
        st.plotly_chart(fig_mensual, use_container_width=True)
    else:
        st.info("No hay datos suficientes para serie temporal.")

    st.subheader("Interfaz de predicción")
    asegurar_base(NOMBRE_DB)
    docs_db = consultar_todos_documentos(NOMBRE_DB, limite=500)
    modelos_db = [d for d in docs_db if isinstance(d, dict) and d.get("type") == "model"]
    ids_modelos = [m["_id"] for m in modelos_db]
    if ids_modelos:
        id_sel = st.selectbox("Selecciona modelo (doc_id) para predicción", ids_modelos)
        if st.button("Cargar modelo seleccionado para predicción"):
            modelo_cargado = cargar_modelo_de_couch(NOMBRE_DB, id_sel)
            if modelo_cargado is None:
                st.error("No se pudo cargar el modelo.")
            else:
                st.session_state.modelo_desplegado = modelo_cargado
                st.success("Modelo cargado en sesión para predicción.")
    else:
        st.info("No hay modelos guardados en CouchDB.")

    # usar modelo en sesión o desplegado
    modelo_bundle = None
    if "modelo_desplegado" in st.session_state:
        modelo_bundle = st.session_state.modelo_desplegado
    elif "modelo_regresion" in st.session_state:
        modelo_bundle = st.session_state.modelo_regresion

    if modelo_bundle is not None and isinstance(modelo_bundle, dict) and "modelo" in modelo_bundle and "preproc" in modelo_bundle:
        st.write("Modelo listo para predecir. Features esperadas:", modelo_bundle.get("features", "No disponible"))
        modo = st.radio("Fuente de entrada", ["Seleccionar fila del dataset", "Subir CSV (1 fila)"])
        df_input = None
        if modo == "Seleccionar fila del dataset":
            idx = st.number_input("Índice (0-based)", min_value=0, max_value=max(0, df_use.shape[0]-1), value=0, step=1)
            df_input = df_use.iloc[[int(idx)]].copy()
        else:
            archivo_in = st.file_uploader("Sube CSV (una fila)", type=["csv"])
            if archivo_in:
                try:
                    df_input = pd.read_csv(archivo_in)
                except Exception:
                    archivo_in.seek(0)
                    df_input = pd.read_csv(io.StringIO(archivo_in.getvalue().decode('latin1')))

        if df_input is not None:
            st.write("Entrada (vista previa):")
            st.write(df_input.head())
            try:
                preproc = modelo_bundle["preproc"]
                modelo = modelo_bundle["modelo"]
                # intentar preparar usando columnas que espera el preproc (si es posible)
                X_in = None
                try:
                    cols_esperadas = []
                    for t in preproc.transformers_:
                        cols_esperadas += list(t[2])
                    cols_esperadas = list(dict.fromkeys(cols_esperadas))
                    if set(cols_esperadas).issubset(set(df_input.columns)):
                        X_in = dataframe_preparado(preproc, df_input[cols_esperadas])
                    else:
                        X_in = dataframe_preparado(preproc, df_input)
                except Exception:
                    X_in = dataframe_preparado(preproc, df_input)
            except Exception as e:
                st.error(f"Error preparando entrada: {e}")
                X_in = None

            if X_in is not None:
                try:
                    preds = modelo.predict(X_in)
                except Exception as e:
                    st.error(f"Error al predecir: {e}")
                    preds = None
                if preds is not None:
                    st.subheader("Predicción")
                    st.write(preds)
                    if st.button("Guardar predicción en CouchDB"):
                        asegurar_base(NOMBRE_DB)
                        doc_pred = {
                            "type": "prediction",
                            "input": {k: _serializar_valor_para_json(v) for k, v in df_input.iloc[0].to_dict().items()},
                            "prediction": float(preds[0]) if hasattr(preds[0], "item") or isinstance(preds[0], (float, np.floating)) else preds[0],
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        status, resp = guardar_documento(NOMBRE_DB, doc_pred)
                        if status in (201, 202):
                            st.success("Predicción guardada en CouchDB.")
                        else:
                            st.error(f"Error guardando predicción: {resp}")
    else:
        st.info("No hay modelo disponible en sesión ni cargado desde la base de datos.")

    st.subheader("Muestra de documentos en CouchDB")
    docs_muestra = consultar_todos_documentos(NOMBRE_DB, limite=200)
    if docs_muestra and isinstance(docs_muestra, list):
        df_docs = pd.DataFrame([d for d in docs_muestra if isinstance(d, dict)])
        if not df_docs.empty:
            st.dataframe(df_docs.head(50))
            csv_docs = df_docs.to_csv(index=False)
            b64 = base64.b64encode(csv_docs.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="graiman_docs.csv">Descargar documentos (CSV)</a>'
            st.markdown(href, unsafe_allow_html=True)
        else:
            st.info("No hay documentos para mostrar.")
    else:
        st.info("No se pudo consultar CouchDB o no hay documentos.")

# --------------------------
# Barra lateral - notas
# --------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("Notas:")
st.sidebar.markdown("""
- Esta app es un ejemplo educativo. Para producción, proteja credenciales y use almacenamiento de modelos especializado.
- Evite guardar modelos muy grandes en CouchDB como base64 en entornos reales.
- Para inserciones masivas en CouchDB utilice _bulk_docs.
""")
