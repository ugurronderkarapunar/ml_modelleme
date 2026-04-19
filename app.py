import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, 
    OneHotEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, f1_score, silhouette_score
)
import xgboost as xgb
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AutoML Studio Pro", layout="wide")

# ------------------ YARDIMCI FONKSİYONLAR ------------------
def advanced_stats(df, col):
    ser = df[col].dropna()
    if len(ser) == 0:
        return {}
    stats_dict = {
        'mean': ser.mean(),
        'std': ser.std(),
        'skew': ser.skew(),
        'kurtosis': ser.kurtosis(),
        'iqr': ser.quantile(0.75) - ser.quantile(0.25),
        'missing_ratio': df[col].isnull().mean(),
        'outlier_ratio_iqr': ((ser < (ser.quantile(0.25) - 1.5*(ser.quantile(0.75)-ser.quantile(0.25)))) |
                              (ser > (ser.quantile(0.75) + 1.5*(ser.quantile(0.75)-ser.quantile(0.25)))).mean()
    }
    if len(ser) >= 8:
        shapiro = stats.shapiro(ser)
        stats_dict['shapiro_p'] = shapiro.pvalue
    return stats_dict

# ------------------ SESSION STATE ------------------
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.stage = "📁 Veri Yükleme"
    st.session_state.task_type = None
    st.session_state.target_col = None
    st.session_state.drop_cols = []
    st.session_state.col_types = {}
    st.session_state.preprocessing = {}
    st.session_state.model = None
    st.session_state.metrics = None
    st.session_state.feature_importance = None

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## 🔬 AutoML Pipeline")
stages = ["📁 Veri Yükleme", "🎯 Görev Seçimi", "📊 EDA", "⚙️ Değişken Tipleri", "🧹 Ön İşleme", "🤖 Modelleme", "📈 Sonuçlar", "🔮 Canlı Tahmin"]
stage_idx = st.sidebar.radio("Adımlar", stages, index=stages.index(st.session_state.stage) if st.session_state.stage in stages else 0)
st.session_state.stage = stage_idx

st.title("📊 AutoML Studio Pro")

# ----- 1. VERİ YÜKLEME -----
if st.session_state.stage == "📁 Veri Yükleme":
    st.header("Veri Yükleme")
    uploaded_file = st.file_uploader("CSV, Excel veya TSV", type=["csv", "xlsx", "xls", "tsv"])
    if uploaded_file and st.session_state.df is None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
            st.session_state.df = df.copy()
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    st.session_state.col_types[col] = "numeric"
                else:
                    try:
                        pd.to_datetime(df[col], errors='raise')
                        st.session_state.col_types[col] = "date"
                    except:
                        st.session_state.col_types[col] = "categorical"
            st.success(f"✅ {df.shape[0]} satır, {df.shape[1]} sütun yüklendi.")
            st.dataframe(df.head(5))
        except Exception as e:
            st.error(f"Hata: {e}")
    if st.session_state.df is not None:
        if st.button("Görev Seçimine Geç →"):
            st.session_state.stage = "🎯 Görev Seçimi"
            st.rerun()

# ----- 2. GÖREV SEÇİMİ -----
elif st.session_state.stage == "🎯 Görev Seçimi":
    st.header("Görev Tipi Seçimi")
    if st.session_state.df is None:
        st.warning("Lütfen önce veri yükleyin.")
    else:
        if st.session_state.target_col:
            st.info("🤖 **Öneri:** Hedef değişkene göre görev tipi belirlenebilir.")
        task = st.radio("Görev tipi:", ["Sınıflandırma", "Regresyon", "Kümeleme"], horizontal=True)
        st.session_state.task_type = task.lower()
        if st.button("Değişken Tiplerine Git →"):
            st.session_state.stage = "⚙️ Değişken Tipleri"
            st.rerun()

# ----- 3. EDA -----
elif st.session_state.stage == "📊 EDA":
    st.header("Keşifsel Veri Analizi")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yüklenmemiş.")
    else:
        numeric_cols = [c for c in df.columns if st.session_state.col_types.get(c) == "numeric"]
        for col in numeric_cols[:3]:
            stat = advanced_stats(df, col)
            st.subheader(col)
            col1, col2 = st.columns(2)
            col1.metric("Ortalama", f"{stat['mean']:.2f}")
            col2.metric("Std", f"{stat['std']:.2f}")
            st.write(f"Çarpıklık: {stat['skew']:.2f}, Basıklık: {stat['kurtosis']:.2f}")
            fig, ax = plt.subplots(figsize=(4,2))
            sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#3dff8f')
            st.pyplot(fig)
        if st.button("Değişken Tiplerine Git →"):
            st.session_state.stage = "⚙️ Değişken Tipleri"
            st.rerun()

# ----- 4. DEĞİŞKEN TİPLERİ -----
elif st.session_state.stage == "⚙️ Değişken Tipleri":
    st.header("Değişken Tiplerini Manuel Atama")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yükleyin.")
    else:
        col_types_new = {}
        cols = df.columns.tolist()
        for col in cols:
            col_types_new[col] = st.selectbox(
                f"{col}", 
                ["numeric", "categorical", "date"],
                index=["numeric","categorical","date"].index(st.session_state.col_types.get(col, "numeric"))
            )
        st.session_state.col_types = col_types_new
        
        target = st.selectbox("Hedef Değişken", ["Seçiniz"] + cols)
        if target != "Seçiniz":
            st.session_state.target_col = target
        
        drop = st.multiselect("Çıkarılacak sütunlar", [c for c in cols if c != target])
        st.session_state.drop_cols = drop
        
        if st.button("Ön İşleme Adımına Git →"):
            st.session_state.stage = "🧹 Ön İşleme"
            st.rerun()

# ----- 5. ÖN İŞLEME -----
elif st.session_state.stage == "🧹 Ön İşleme":
    st.header("Ön İşleme Stratejileri")
    df = st.session_state.df
    target = st.session_state.target_col
    drop = st.session_state.drop_cols
    col_types = st.session_state.col_types
    if df is None or target is None:
        st.warning("Hedef değişken tanımlanmamış.")
    else:
        missing_num = st.selectbox("Sayısal eksik değer stratejisi", ["Ortalama", "Medyan", "KNN Impute"])
        scaling = st.selectbox("Ölçeklendirme", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        encoding = st.selectbox("Kategorik kodlama", ["OneHot", "Label"])
        
        st.session_state.preprocessing = {
            'missing_num': missing_num,
            'scaling': scaling,
            'encoding': encoding
        }
        
        if st.button("Modellemeye Git →"):
            st.session_state.stage = "🤖 Modelleme"
            st.rerun()

# ----- 6. MODEL SEÇİMİ -----
elif st.session_state.stage == "🤖 Modelleme":
    st.header("Model Seçimi ve Hiperparametreler")
    df = st.session_state.df
    target = st.session_state.target_col
    drop = st.session_state.drop_cols
    col_types = st.session_state.col_types
    prep = st.session_state.preprocessing
    if df is None or target is None:
        st.warning("Önce değişkenleri tanımlayın.")
    else:
        X = df.drop(columns=[target] + drop, errors='ignore')
        y = df[target]
        numeric_cols = [c for c in X.columns if col_types.get(c)=="numeric"]
        categorical_cols = [c for c in X.columns if col_types.get(c)=="categorical"]
        
        # Pipeline
        transformers = []
        if numeric_cols:
            num_steps = []
            if prep['missing_num'] == "Ortalama":
                num_steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif prep['missing_num'] == "Medyan":
                num_steps.append(('imputer', SimpleImputer(strategy='median')))
            else:
                num_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            if prep['scaling'] == "StandardScaler":
                num_steps.append(('scaler', StandardScaler()))
            elif prep['scaling'] == "MinMaxScaler":
                num_steps.append(('scaler', MinMaxScaler()))
            else:
                num_steps.append(('scaler', RobustScaler()))
            transformers.append(('num', Pipeline(num_steps), numeric_cols))
        
        if categorical_cols:
            cat_steps = []
            cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            if prep['encoding'] == "OneHot":
                cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            else:
                cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            transformers.append(('cat', Pipeline(cat_steps), categorical_cols))
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        task = st.session_state.task_type
        if task == "sınıflandırma":
            model_name = st.selectbox("Model", ["Lojistik Regresyon", "Random Forest", "XGBoost"])
            if model_name == "Lojistik Regresyon":
                C = st.number_input("C", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=C, max_iter=1000)
            elif model_name == "Random Forest":
                n_est = st.slider("n_estimators", 10, 300, 100)
                model = RandomForestClassifier(n_estimators=n_est)
            else:
                n_est = st.slider("n_estimators", 10, 300, 100)
                model = xgb.XGBClassifier(n_estimators=n_est, eval_metric='logloss')
        elif task == "regresyon":
            model_name = st.selectbox("Model", ["Linear Regresyon", "Random Forest", "XGBoost"])
            if model_name == "Linear Regresyon":
                model = LinearRegression()
            elif model_name == "Random Forest":
                n_est = st.slider("n_estimators", 10, 300, 100)
                model = RandomForestRegressor(n_estimators=n_est)
            else:
                n_est = st.slider("n_estimators", 10, 300, 100)
                model = xgb.XGBRegressor(n_estimators=n_est)
        else:
            model_name = st.selectbox("Model", ["K-Means", "DBSCAN"])
            if model_name == "K-Means":
                n_clusters = st.slider("küme sayısı", 2, 10, 3)
                model = KMeans(n_clusters=n_clusters, n_init=10)
            else:
                eps = st.number_input("eps", 0.1, 2.0, 0.5)
                model = DBSCAN(eps=eps)
        
        test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100
        
        if st.button("Modeli Eğit"):
            with st.spinner("Eğitim başladı..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                pipeline.fit(X_train, y_train)
                st.session_state.model = pipeline
                
                y_pred = pipeline.predict(X_test)
                if task == "sınıflandırma":
                    acc = accuracy_score(y_test, y_pred)
                    st.session_state.metrics = {'accuracy': acc}
                elif task == "regresyon":
                    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                    r2 = r2_score(y_test, y_pred)
                    st.session_state.metrics = {'rmse': rmse, 'r2': r2}
                else:
                    X_processed = preprocessor.fit_transform(X)
                    model.fit(X_processed)
                    sil = silhouette_score(X_processed, model.labels_)
                    st.session_state.metrics = {'silhouette': sil}
                
                st.success("Eğitim tamamlandı!")
                st.session_state.stage = "📈 Sonuçlar"
                st.rerun()

# ----- 7. SONUÇLAR -----
elif st.session_state.stage == "📈 Sonuçlar":
    st.header("Model Performansı")
    if st.session_state.metrics is None:
        st.warning("Model eğitilmedi.")
    else:
        metrics = st.session_state.metrics
        task = st.session_state.task_type
        if task == "sınıflandırma":
            st.metric("Doğruluk (Accuracy)", f"{metrics['accuracy']:.2%}")
        elif task == "regresyon":
            st.metric("RMSE", f"{metrics['rmse']:.2f}")
            st.metric("R²", f"{metrics['r2']:.3f}")
        else:
            st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
        
        if st.button("Canlı Tahmin Yap →"):
            st.session_state.stage = "🔮 Canlı Tahmin"
            st.rerun()

# ----- 8. CANLI TAHMİN -----
elif st.session_state.stage == "🔮 Canlı Tahmin":
    st.header("Canlı Tahmin Aracı")
    if st.session_state.model is None:
        st.warning("Model eğitilmemiş.")
    else:
        df = st.session_state.df
        target = st.session_state.target_col
        drop = st.session_state.drop_cols
        X_columns = [c for c in df.columns if c not in drop and c != target]
        user_inputs = {}
        with st.form("tahmin_formu"):
            for col in X_columns:
                val = st.text_input(f"{col}", key=col)
                user_inputs[col] = val
            submitted = st.form_submit_button("Tahmin Et")
            if submitted:
                input_df = pd.DataFrame([user_inputs])
                try:
                    pred = st.session_state.model.predict(input_df)[0]
                    st.success(f"Tahmin sonucu: {pred}")
                except Exception as e:
                    st.error(f"Hata: {e}")
