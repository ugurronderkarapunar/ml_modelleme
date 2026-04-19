import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, silhouette_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AutoML Studio", layout="wide")

# ------------------ SESSION STATE ------------------
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.stage = "Veri Yükleme"
    st.session_state.task_type = None
    st.session_state.target_col = None
    st.session_state.drop_cols = []
    st.session_state.col_types = {}
    st.session_state.preprocessing = {}
    st.session_state.model = None
    st.session_state.metrics = None

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## AutoML Pipeline")
stages = ["Veri Yükleme", "Görev Seçimi", "EDA", "Değişken Tipleri", "Ön İşleme", "Modelleme", "Sonuçlar", "Canlı Tahmin"]
current_stage = st.sidebar.radio("Adımlar", stages, index=stages.index(st.session_state.stage))
st.session_state.stage = current_stage

st.title("AutoML Uygulaması")

# 1. VERİ YÜKLEME
if st.session_state.stage == "Veri Yükleme":
    st.header("Veri Yükleme")
    uploaded = st.file_uploader("CSV veya Excel dosyası seçin", type=["csv", "xlsx", "xls"])
    if uploaded and st.session_state.df is None:
        try:
            if uploaded.name.endswith('.csv'):
                df = pd.read_csv(uploaded)
            else:
                df = pd.read_excel(uploaded)
            st.session_state.df = df
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.session_state.col_types[col] = "numeric"
                else:
                    st.session_state.col_types[col] = "categorical"
            st.success("Veri yüklendi")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Hata: {e}")
    if st.session_state.df is not None:
        if st.button("Görev Seçimine Git"):
            st.session_state.stage = "Görev Seçimi"
            st.rerun()

# 2. GÖREV SEÇİMİ
elif st.session_state.stage == "Görev Seçimi":
    st.header("Görev Tipi")
    task = st.radio("Görev", ["Sınıflandırma", "Regresyon", "Kümeleme"], horizontal=True)
    st.session_state.task_type = task.lower()
    if st.button("Değişken Tiplerine Git"):
        st.session_state.stage = "Değişken Tipleri"
        st.rerun()

# 3. EDA (basit)
elif st.session_state.stage == "EDA":
    st.header("EDA")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yok")
    else:
        st.write(df.describe())
        numeric_cols = df.select_dtypes(include=np.number).columns
        for col in numeric_cols[:2]:
            fig, ax = plt.subplots()
            df[col].hist(ax=ax)
            st.pyplot(fig)
        if st.button("Değişken Tiplerine Git"):
            st.session_state.stage = "Değişken Tipleri"
            st.rerun()

# 4. DEĞİŞKEN TİPLERİ (manuel atama)
elif st.session_state.stage == "Değişken Tipleri":
    st.header("Değişken Tiplerini Ayarlayın")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yok")
    else:
        col_types_new = {}
        cols = df.columns.tolist()
        for col in cols:
            col_types_new[col] = st.selectbox(
                col,
                ["numeric", "categorical"],
                index=0 if st.session_state.col_types.get(col) == "numeric" else 1
            )
        st.session_state.col_types = col_types_new
        target = st.selectbox("Hedef değişken", ["Seçiniz"] + cols)
        if target != "Seçiniz":
            st.session_state.target_col = target
        drop = st.multiselect("Çıkarılacak sütunlar", [c for c in cols if c != target])
        st.session_state.drop_cols = drop
        if st.button("Ön İşlemeye Git"):
            st.session_state.stage = "Ön İşleme"
            st.rerun()

# 5. ÖN İŞLEME
elif st.session_state.stage == "Ön İşleme":
    st.header("Ön İşleme Ayarları")
    missing_num = st.selectbox("Sayısal eksik değer stratejisi", ["ortalama", "medyan", "knn"])
    scaling = st.selectbox("Ölçeklendirme", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    encoding = st.selectbox("Kategorik kodlama", ["OneHot", "Label"])
    st.session_state.preprocessing = {
        "missing_num": missing_num,
        "scaling": scaling,
        "encoding": encoding
    }
    if st.button("Modellemeye Git"):
        st.session_state.stage = "Modelleme"
        st.rerun()

# 6. MODELLEME
elif st.session_state.stage == "Modelleme":
    st.header("Model Seçimi ve Eğitim")
    df = st.session_state.df
    target = st.session_state.target_col
    drop = st.session_state.drop_cols
    col_types = st.session_state.col_types
    prep = st.session_state.preprocessing

    if df is None or target is None:
        st.warning("Hedef değişken seçilmemiş")
    else:
        X = df.drop(columns=[target] + drop, errors='ignore')
        y = df[target]
        numeric_cols = [c for c in X.columns if col_types.get(c) == "numeric"]
        categorical_cols = [c for c in X.columns if col_types.get(c) == "categorical"]

        # Preprocessor
        transformers = []
        if numeric_cols:
            num_steps = []
            if prep["missing_num"] == "ortalama":
                num_steps.append(("imputer", SimpleImputer(strategy="mean")))
            elif prep["missing_num"] == "medyan":
                num_steps.append(("imputer", SimpleImputer(strategy="median")))
            else:
                num_steps.append(("imputer", KNNImputer(n_neighbors=5)))
            if prep["scaling"] == "StandardScaler":
                num_steps.append(("scaler", StandardScaler()))
            elif prep["scaling"] == "MinMaxScaler":
                num_steps.append(("scaler", MinMaxScaler()))
            else:
                num_steps.append(("scaler", RobustScaler()))
            transformers.append(("num", Pipeline(num_steps), numeric_cols))

        if categorical_cols:
            cat_steps = [("imputer", SimpleImputer(strategy="most_frequent"))]
            if prep["encoding"] == "OneHot":
                cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
            else:
                cat_steps.append(("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)))
            transformers.append(("cat", Pipeline(cat_steps), categorical_cols))

        preprocessor = ColumnTransformer(transformers, remainder="drop")

        task = st.session_state.task_type
        if task == "sınıflandırma":
            model_name = st.selectbox("Model", ["Lojistik Regresyon", "Random Forest", "XGBoost"])
            if model_name == "Lojistik Regresyon":
                model = LogisticRegression(max_iter=1000)
            elif model_name == "Random Forest":
                model = RandomForestClassifier(n_estimators=100)
            else:
                model = xgb.XGBClassifier(eval_metric="logloss")
        elif task == "regresyon":
            model_name = st.selectbox("Model", ["Linear Regresyon", "Random Forest", "XGBoost"])
            if model_name == "Linear Regresyon":
                model = LinearRegression()
            elif model_name == "Random Forest":
                model = RandomForestRegressor(n_estimators=100)
            else:
                model = xgb.XGBRegressor()
        else:
            model_name = st.selectbox("Model", ["K-Means", "DBSCAN"])
            if model_name == "K-Means":
                model = KMeans(n_clusters=3, n_init=10)
            else:
                model = DBSCAN(eps=0.5)

        test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100

        if st.button("Eğit"):
            with st.spinner("Eğitim devam ediyor..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
                pipeline.fit(X_train, y_train)
                st.session_state.model = pipeline
                y_pred = pipeline.predict(X_test)
                if task == "sınıflandırma":
                    acc = accuracy_score(y_test, y_pred)
                    st.session_state.metrics = {"accuracy": acc}
                elif task == "regresyon":
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    st.session_state.metrics = {"rmse": rmse, "r2": r2}
                else:
                    X_processed = preprocessor.fit_transform(X)
                    model.fit(X_processed)
                    sil = silhouette_score(X_processed, model.labels_)
                    st.session_state.metrics = {"silhouette": sil}
                st.success("Eğitim tamamlandı")
                st.session_state.stage = "Sonuçlar"
                st.rerun()

# 7. SONUÇLAR
elif st.session_state.stage == "Sonuçlar":
    st.header("Sonuçlar")
    if st.session_state.metrics is None:
        st.warning("Model eğitilmemiş")
    else:
        m = st.session_state.metrics
        if "accuracy" in m:
            st.metric("Doğruluk", f"{m['accuracy']:.2%}")
        elif "rmse" in m:
            st.metric("RMSE", f"{m['rmse']:.2f}")
            st.metric("R²", f"{m['r2']:.3f}")
        else:
            st.metric("Silhouette", f"{m['silhouette']:.3f}")
        if st.button("Canlı Tahmin"):
            st.session_state.stage = "Canlı Tahmin"
            st.rerun()

# 8. CANLI TAHMİN
elif st.session_state.stage == "Canlı Tahmin":
    st.header("Canlı Tahmin")
    if st.session_state.model is None:
        st.warning("Model yok")
    else:
        df = st.session_state.df
        target = st.session_state.target_col
        drop = st.session_state.drop_cols
        X_cols = [c for c in df.columns if c not in drop and c != target]
        inputs = {}
        for col in X_cols:
            inputs[col] = st.text_input(col)
        if st.button("Tahmin Et"):
            input_df = pd.DataFrame([inputs])
            try:
                pred = st.session_state.model.predict(input_df)[0]
                st.success(f"Tahmin: {pred}")
            except Exception as e:
                st.error(f"Hata: {e}")
