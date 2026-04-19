import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, 
    OneHotEncoder, OrdinalEncoder
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score, silhouette_score
)
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AutoML Studio Pro", layout="wide")

# ------------------ YARDIMCI FONKSİYONLAR ------------------
def advanced_stats(df, col):
    ser = df[col].dropna()
    if len(ser) == 0:
        return {}
    
    q1 = ser.quantile(0.25)
    q3 = ser.quantile(0.75)
    iqr = q3 - q1
    
    stats_dict = {
        'mean': ser.mean(),
        'std': ser.std(),
        'skew': ser.skew(),
        'kurtosis': ser.kurtosis(),
        'iqr': iqr,
        'missing_ratio': df[col].isnull().mean(),
        # HATA DÜZELTİLDİ: Parantez ve hesaplama mantığı
        'outlier_ratio_iqr': ((ser < (q1 - 1.5 * iqr)) | (ser > (q3 + 1.5 * iqr))).mean()
    }
    
    if len(ser) >= 8:
        try:
            shapiro = stats.shapiro(ser)
            stats_dict['shapiro_p'] = shapiro.pvalue
        except:
            stats_dict['shapiro_p'] = 0
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

# ------------------ SIDEBAR ------------------
st.sidebar.markdown("## 🔬 AutoML Pipeline")
stages = ["📁 Veri Yükleme", "🎯 Görev Seçimi", "📊 EDA", "⚙️ Değişken Tipleri", "🧹 Ön İşleme", "🤖 Modelleme", "📈 Sonuçlar", "🔮 Canlı Tahmin"]
# Geçerli stage indexini bulma
current_idx = stages.index(st.session_state.stage) if st.session_state.stage in stages else 0
stage_idx = st.sidebar.radio("Adımlar", stages, index=current_idx)
st.session_state.stage = stage_idx

st.title("📊 AutoML Studio Pro")

# ----- 1. VERİ YÜKLEME -----
if st.session_state.stage == "📁 Veri Yükleme":
    st.header("Veri Yükleme")
    uploaded_file = st.file_uploader("CSV, Excel veya TSV", type=["csv", "xlsx", "xls", "tsv"])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
            
            st.session_state.df = df
            # Başlangıç veri tiplerini belirle
            new_types = {}
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    new_types[col] = "numeric"
                else:
                    new_types[col] = "categorical"
            st.session_state.col_types = new_types
            
            st.success(f"✅ {df.shape[0]} satır yüklendi.")
            st.dataframe(df.head(10))
            
            if st.button("Görev Seçimine Geç →"):
                st.session_state.stage = "🎯 Görev Seçimi"
                st.rerun()
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")

# ----- 2. GÖREV SEÇİMİ -----
elif st.session_state.stage == "🎯 Görev Seçimi":
    st.header("Görev Tipi Seçimi")
    if st.session_state.df is None:
        st.warning("Lütfen önce veri yükleyin.")
    else:
        task = st.radio("Görev tipi:", ["Sınıflandırma", "Regresyon", "Kümeleme"], horizontal=True)
        st.session_state.task_type = task.lower()
        if st.button("EDA'ya Git →"):
            st.session_state.stage = "📊 EDA"
            st.rerun()

# ----- 3. EDA -----
elif st.session_state.stage == "📊 EDA":
    st.header("Keşifsel Veri Analizi")
    if st.session_state.df is None:
        st.warning("Veri yok.")
    else:
        df = st.session_state.df
        numeric_cols = [c for c in df.columns if st.session_state.col_types.get(c) == "numeric"]
        
        for col in numeric_cols[:5]: # İlk 5 sayısal sütun
            stat = advanced_stats(df, col)
            with st.expander(f"İstatistikler: {col}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("Ortalama", f"{stat['mean']:.2f}")
                c2.metric("Eksik Oranı", f"{stat.get('missing_ratio',0):.1%}")
                c3.metric("Outlier (IQR)", f"{stat.get('outlier_ratio_iqr',0):.1%}")
                
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.histplot(df[col].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
                plt.close()

        if st.button("Değişken Tiplerine Git →"):
            st.session_state.stage = "⚙️ Değişken Tipleri"
            st.rerun()

# ----- 4. DEĞİŞKEN TİPLERİ -----
elif st.session_state.stage == "⚙️ Değişken Tipleri":
    st.header("Değişken Yönetimi")
    df = st.session_state.df
    if df is not None:
        cols = df.columns.tolist()
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Tip Atama")
            for col in cols:
                st.session_state.col_types[col] = st.selectbox(
                    f"{col} tipi:", ["numeric", "categorical", "date"],
                    index=["numeric", "categorical", "date"].index(st.session_state.col_types.get(col, "numeric")),
                    key=f"type_{col}"
                )
        
        with col2:
            st.subheader("Hedef ve Filtre")
            st.session_state.target_col = st.selectbox("🎯 Hedef Değişken (Y)", [None] + cols)
            st.session_state.drop_cols = st.multiselect("🗑️ Atılacak Sütunlar (ID vb.)", 
                                                      [c for c in cols if c != st.session_state.target_col])
        
        if st.button("Ön İşleme Git →"):
            st.session_state.stage = "🧹 Ön İşleme"
            st.rerun()

# ----- 5. ÖN İŞLEME -----
elif st.session_state.stage == "🧹 Ön İşleme":
    st.header("Pipeline Ayarları")
    st.session_state.preprocessing['missing_num'] = st.selectbox("Sayısal Boş Değer", ["Ortalama", "Medyan", "KNN Impute"])
    st.session_state.preprocessing['scaling'] = st.selectbox("Ölçeklendirme", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
    st.session_state.preprocessing['encoding'] = st.selectbox("Kategorik Kodlama", ["OneHot", "Ordinal"])
    
    if st.button("Modellemeye Geç →"):
        st.session_state.stage = "🤖 Modelleme"
        st.rerun()

# ----- 6. MODELLEME -----
elif st.session_state.stage == "🤖 Modelleme":
    st.header("Eğitim")
    df = st.session_state.df
    target = st.session_state.target_col
    
    if df is not None and target is not None:
        X = df.drop(columns=[target] + st.session_state.drop_cols, errors='ignore')
        y = df[target]
        
        # Pipeline İnşası
        num_cols = [c for c in X.columns if st.session_state.col_types[c] == "numeric"]
        cat_cols = [c for c in X.columns if st.session_state.col_types[c] == "categorical"]
        
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy=st.session_state.preprocessing['missing_num'].replace("Ortalama","mean").replace("Medyan","median")) if "KNN" not in st.session_state.preprocessing['missing_num'] else KNNImputer()),
            ('scaler', globals()[st.session_state.preprocessing['scaling']]())
        ])
        
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False) if st.session_state.preprocessing['encoding']=="OneHot" else OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])
        
        preprocessor = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)])
        
        # Model Seçimi
        task = st.session_state.task_type
        if task == "sınıflandırma":
            m_choice = st.selectbox("Algoritma", ["Random Forest", "XGBoost", "Logistic Regression"])
            model = RandomForestClassifier() if m_choice == "Random Forest" else (xgb.XGBClassifier() if m_choice == "XGBoost" else LogisticRegression())
        else:
            m_choice = st.selectbox("Algoritma", ["Random Forest", "XGBoost", "Linear Regression"])
            model = RandomForestRegressor() if m_choice == "Random Forest" else (xgb.XGBRegressor() if m_choice == "XGBoost" else LinearRegression())
            
        if st.button("🚀 Eğitimi Başlat"):
            with st.spinner("Model eğitiliyor..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                full_pipe = Pipeline([('pre', preprocessor), ('model', model)])
                full_pipe.fit(X_train, y_train)
                
                st.session_state.model = full_pipe
                y_pred = full_pipe.predict(X_test)
                
                if task == "sınıflandırma":
                    st.session_state.metrics = {"Accuracy": accuracy_score(y_test, y_pred)}
                else:
                    st.session_state.metrics = {"R2": r2_score(y_test, y_pred), "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))}
                
                st.success("Eğitim Tamamlandı!")
                st.session_state.stage = "📈 Sonuçlar"
                st.rerun()

# ----- 7. SONUÇLAR -----
elif st.session_state.stage == "📈 Sonuçlar":
    st.header("Model Performansı")
    if st.session_state.metrics:
        for k, v in st.session_state.metrics.items():
            st.metric(k, f"{v:.4f}")
        
        if st.button("Canlı Tahmin Ekranı →"):
            st.session_state.stage = "🔮 Canlı Tahmin"
            st.rerun()

# ----- 8. CANLI TAHMİN -----
elif st.session_state.stage == "🔮 Canlı Tahmin":
    st.header("Yeni Veri ile Tahmin")
    if st.session_state.model:
        df = st.session_state.df
        target = st.session_state.target_col
        # Eğitimde kullanılan sütunları al
        X_cols = [c for c in df.columns if c not in st.session_state.drop_cols and c != target]
        
        with st.form("predict_form"):
            user_data = {}
            cols_grid = st.columns(3)
            for i, col in enumerate(X_cols):
                with cols_grid[i % 3]:
                    user_data[col] = st.text_input(f"{col}", value=str(df[col].iloc[0]))
            
            if st.form_submit_button("Tahmin Et"):
                # HATA DÜZELTİLDİ: Veriyi DataFrame yaparken tipleri zorla (coerce)
                input_df = pd.DataFrame([user_data])
                for c in X_cols:
                    if st.session_state.col_types[c] == "numeric":
                        input_df[c] = pd.to_numeric(input_df[c], errors='coerce')
                
                prediction = st.session_state.model.predict(input_df)
                st.balloons()
                st.success(f"### Sonuç: {prediction[0]}")
