import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, 
    OneHotEncoder, LabelEncoder, OrdinalEncoder, FunctionTransformer
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, 
    silhouette_score, davies_bouldin_score
)
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ------------------ SAYFA YAPILANDIRMASI ------------------
st.set_page_config(page_title="AutoML Studio Pro", layout="wide", page_icon="📊")
st.markdown("""
<style>
    .main { background-color: #0a0c0e; }
    .stButton>button { background-color: #0f1a14; color: #3dff8f; border: 1px solid #3dff8f; border-radius: 3px; }
    .stButton>button:hover { background-color: #1a2a22; color: #6affa3; }
    .css-1xarl3l { font-family: 'IBM Plex Mono', monospace; }
    .block-container { padding-top: 1rem; padding-bottom: 0rem; }
</style>
""", unsafe_allow_html=True)

# ------------------ YARDIMCI FONKSİYONLAR ------------------
def advanced_stats(df, col):
    """Derinlemesine istatistikler: çarpıklık, basıklık, normallik testi, aykırı değer oranı, IQR, varyans"""
    ser = df[col].dropna()
    if len(ser) == 0:
        return {}
    stats_dict = {
        'mean': ser.mean(),
        'std': ser.std(),
        'skew': ser.skew(),
        'kurtosis': ser.kurtosis(),
        'min': ser.min(),
        'q1': ser.quantile(0.25),
        'median': ser.median(),
        'q3': ser.quantile(0.75),
        'max': ser.max(),
        'iqr': ser.quantile(0.75) - ser.quantile(0.25),
        'missing_ratio': df[col].isnull().mean(),
        'outlier_ratio_iqr': ((ser < (ser.quantile(0.25) - 1.5*(ser.quantile(0.75)-ser.quantile(0.25)))) |
                              (ser > (ser.quantile(0.75) + 1.5*(ser.quantile(0.75)-ser.quantile(0.25)))).mean(),
        'outlier_ratio_zscore': (np.abs(stats.zscore(ser)) > 3).mean(),
        'variance': ser.var()
    }
    if len(ser) >= 8:
        shapiro = stats.shapiro(ser)
        stats_dict['shapiro_p'] = shapiro.pvalue
        stats_dict['normal_distribution'] = shapiro.pvalue > 0.05
    else:
        stats_dict['normal_distribution'] = None
    return stats_dict

def normality_suggestion(skew, kurt, p_val):
    if p_val is not None and p_val > 0.05:
        return "✅ Normal dağılım (Shapiro-Wilk p>0.05)"
    if abs(skew) > 1.5:
        return "⚠️ Yüksek çarpıklık → Box-Cox veya Yeo-Johnson önerilir"
    if abs(kurt) > 3:
        return "⚠️ Yüksek basıklık → Log dönüşümü önerilir"
    return "◈ Normal dağılım göstermiyor ancak dönüşüm gerekli değil"

def outlier_suggestion(outlier_ratio):
    if outlier_ratio > 0.05:
        return f"🔍 Aykırı değer oranı %{outlier_ratio*100:.1f} → Winsorize veya IQR clipping önerilir"
    return "✓ Aykırı değer oranı düşük"

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
    st.session_state.cv_results = None
    st.session_state.completed_stages = []

# ------------------ SIDEBAR (SIRALI ADIMLAR) ------------------
st.sidebar.markdown("## 🔬 AutoML Pipeline")
stages = ["📁 Veri Yükleme", "🎯 Görev Seçimi", "📊 EDA", "⚙️ Değişken Tipleri", "🧹 Ön İşleme", "🤖 Modelleme", "📈 Sonuçlar", "🔮 Canlı Tahmin"]
stage_idx = st.sidebar.radio("Adımlar", stages, index=stages.index(st.session_state.stage) if st.session_state.stage in stages else 0)
st.session_state.stage = stage_idx

# ------------------ ANA İÇERİK ------------------
st.title("📊 AutoML Studio Pro - Yüksek Lisans Düzeyinde İstatistik")

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
        df = st.session_state.df
        if st.session_state.target_col:
            target = df[st.session_state.target_col]
            if st.session_state.col_types.get(st.session_state.target_col) in ["categorical"] or target.nunique() <= 10:
                st.info("🤖 **Öneri:** Hedef kategorik → **Sınıflandırma** görevi önerilir.")
            else:
                st.info("🤖 **Öneri:** Hedef sayısal → **Regresyon** görevi önerilir.")
        else:
            st.info("🤖 Önce değişken tipleri sayfasında hedef değişkeni seçin.")
        task = st.radio("Görev tipi:", ["Sınıflandırma", "Regresyon", "Kümeleme"], horizontal=True)
        st.session_state.task_type = task.lower()
        if st.button("Değişken Tiplerine Git →"):
            st.session_state.stage = "⚙️ Değişken Tipleri"
            st.rerun()

# ----- 3. EDA -----
elif st.session_state.stage == "📊 EDA":
    st.header("Keşifsel Veri Analizi - İstatistik Raporu")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yüklenmemiş.")
    else:
        numeric_cols = [c for c in df.columns if st.session_state.col_types.get(c) == "numeric"]
        cat_cols = [c for c in df.columns if st.session_state.col_types.get(c) == "categorical"]
        date_cols = [c for c in df.columns if st.session_state.col_types.get(c) == "date"]
        
        with st.expander("📈 Sayısal Değişkenlerin İleri İstatistikleri", expanded=True):
            for col in numeric_cols:
                stat = advanced_stats(df, col)
                if not stat:
                    continue
                col1, col2 = st.columns([1,2])
                with col1:
                    st.metric(col, f"μ={stat['mean']:.2f}", delta=f"σ={stat['std']:.2f}")
                with col2:
                    st.markdown(f"""
                    - Çarpıklık: {stat['skew']:.2f} | Basıklık: {stat['kurtosis']:.2f}
                    - IQR: {stat['iqr']:.2f} | Aykırı (IQR): %{stat['outlier_ratio_iqr']*100:.1f}
                    - Shapiro-Wilk p: {stat.get('shapiro_p', 'N/A'):.4f}
                    - {normality_suggestion(stat['skew'], stat['kurtosis'], stat.get('shapiro_p'))}
                    - {outlier_suggestion(stat['outlier_ratio_iqr'])}
                    """)
                fig, ax = plt.subplots(figsize=(4,2))
                sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#3dff8f')
                ax.set_title(col, fontsize=8)
                ax.tick_params(labelsize=6)
                st.pyplot(fig)
        
        with st.expander("📊 Korelasyon Matrisi (Sayısal)"):
            if len(numeric_cols) >= 2:
                corr = df[numeric_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='Greens', height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📦 Aykırı Değer Boxplot'ları"):
            for col in numeric_cols[:6]:
                fig, ax = plt.subplots(figsize=(3,2))
                sns.boxplot(x=df[col], ax=ax, color='#1e3a2f')
                ax.set_title(col, fontsize=8)
                st.pyplot(fig)
        
        with st.expander("🔬 Kategorik Değişken Frekansları"):
            for col in cat_cols[:5]:
                fig, ax = plt.subplots(figsize=(4,2))
                df[col].value_counts().head(8).plot(kind='bar', ax=ax, color='#3dff8f')
                ax.set_title(col)
                ax.tick_params(axis='x', rotation=45, labelsize=6)
                st.pyplot(fig)
        
        if st.button("Değişken Tiplerine Git →"):
            st.session_state.stage = "⚙️ Değişken Tipleri"
            st.rerun()

# ----- 4. DEĞİŞKEN TİPLERİ (Kullanıcı atamalı) -----
elif st.session_state.stage == "⚙️ Değişken Tipleri":
    st.header("Değişken Tiplerini Manuel Atama")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yükleyin.")
    else:
        st.subheader("Her sütun için tip seçin")
        col_types_new = {}
        cols = df.columns.tolist()
        for col in cols:
            col_types_new[col] = st.selectbox(
                f"{col}", 
                ["numeric", "categorical", "date"],
                index=["numeric","categorical","date"].index(st.session_state.col_types.get(col, "numeric"))
            )
        st.session_state.col_types = col_types_new
        
        st.subheader("Hedef Değişken Seçimi")
        target = st.selectbox("Tahmin edilecek değişken (hedef)", ["Seçiniz"] + cols)
        if target != "Seçiniz":
            st.session_state.target_col = target
        
        st.subheader("Model dışı bırakılacak değişkenler")
        drop = st.multiselect("Kullanılmayacak sütunlar", [c for c in cols if c != target])
        st.session_state.drop_cols = drop
        
        if st.button("Ön İşleme Adımına Git →"):
            st.session_state.stage = "🧹 Ön İşleme"
            st.rerun()

# ----- 5. ÖN İŞLEME (Gelişmiş) -----
elif st.session_state.stage == "🧹 Ön İşleme":
    st.header("Gelişmiş Ön İşleme Stratejileri")
    df = st.session_state.df
    target = st.session_state.target_col
    drop = st.session_state.drop_cols
    col_types = st.session_state.col_types
    if df is None or target is None:
        st.warning("Hedef değişken tanımlanmamış.")
    else:
        numeric_cols = [c for c in df.columns if col_types.get(c)=="numeric" and c not in drop and c!=target]
        categorical_cols = [c for c in df.columns if col_types.get(c)=="categorical" and c not in drop and c!=target]
        date_cols = [c for c in df.columns if col_types.get(c)=="date" and c not in drop and c!=target]
        
        st.subheader("Eksik Değer Stratejileri")
        missing_num = st.selectbox("Sayısal eksik değer stratejisi", ["Ortalama", "Medyan", "KNN Impute", "Sil"])
        missing_cat = st.selectbox("Kategorik eksik değer stratejisi", ["Mod", "Sabit (Missing)", "Sil"])
        
        st.subheader("Aykırı Değer Stratejisi (Sayısal)")
        outlier_method = st.selectbox("Yöntem", ["Winsorize (IQR)", "Z-score threshold", "Clip (min-max)", "Sil"])
        outlier_threshold = st.slider("Z-score eşiği (sadece Z-score için)", 2.0, 5.0, 3.0) if outlier_method == "Z-score threshold" else 3.0
        
        st.subheader("Dönüşümler (Sayısal)")
        power_transform = st.selectbox("Güç dönüşümü", ["Yok", "Box-Cox", "Yeo-Johnson"])
        scaling = st.selectbox("Ölçeklendirme", ["StandardScaler", "MinMaxScaler", "RobustScaler", "Yok"])
        
        st.subheader("Kategorik Değişken Kodlama")
        encoding = st.selectbox("Kodlama yöntemi", ["OneHot", "Label", "Frequency Encoding"])
        
        st.subheader("Feature Engineering (İsteğe bağlı)")
        add_poly = st.checkbox("Polinom özellikler (2. derece)")
        add_interaction = st.checkbox("Etkileşim terimleri")
        
        st.session_state.preprocessing = {
            'missing_num': missing_num,
            'missing_cat': missing_cat,
            'outlier_method': outlier_method,
            'outlier_thresh': outlier_threshold,
            'power': power_transform,
            'scaling': scaling,
            'encoding': encoding,
            'poly': add_poly,
            'interaction': add_interaction
        }
        
        if st.button("Modellemeye Git →"):
            st.session_state.stage = "🤖 Modelleme"
            st.rerun()

# ----- 6. MODEL SEÇİMİ VE HİPERPARAMETRE (Manuel) -----
elif st.session_state.stage == "🤖 Modelleme":
    st.header("Model Seçimi ve Manuel Hiperparametreler")
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
        
        transformers = []
        # Sayısal pipeline
        if numeric_cols:
            num_steps = []
            if prep['missing_num'] == "Ortalama":
                num_steps.append(('imputer', SimpleImputer(strategy='mean')))
            elif prep['missing_num'] == "Medyan":
                num_steps.append(('imputer', SimpleImputer(strategy='median')))
            elif prep['missing_num'] == "KNN Impute":
                num_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            else:
                num_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value=0)))
            # Aykırı değer clipping (basit)
            if prep['outlier_method'] == "Clip (min-max)":
                num_steps.append(('clip', FunctionTransformer(lambda x: np.clip(x, np.percentile(x, 1), np.percentile(x, 99)))))
            if prep['power'] == "Box-Cox":
                num_steps.append(('power', PowerTransformer(method='box-cox')))
            elif prep['power'] == "Yeo-Johnson":
                num_steps.append(('power', PowerTransformer(method='yeo-johnson')))
            if prep['scaling'] == "StandardScaler":
                num_steps.append(('scaler', StandardScaler()))
            elif prep['scaling'] == "MinMaxScaler":
                num_steps.append(('scaler', MinMaxScaler()))
            elif prep['scaling'] == "RobustScaler":
                num_steps.append(('scaler', RobustScaler()))
            transformers.append(('num', Pipeline(num_steps), numeric_cols))
        
        # Kategorik pipeline
        if categorical_cols:
            cat_steps = []
            if prep['missing_cat'] == "Mod":
                cat_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
            else:
                cat_steps.append(('imputer', SimpleImputer(strategy='constant', fill_value='missing')))
            if prep['encoding'] == "OneHot":
                cat_steps.append(('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            elif prep['encoding'] == "Label":
                cat_steps.append(('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)))
            else:  # Frequency Encoding (basit)
                freq_map = {col: df[col].value_counts().to_dict() for col in categorical_cols}
                def freq_encode(X):
                    X_enc = X.copy()
                    for col in categorical_cols:
                        X_enc[col] = X[col].map(freq_map.get(col, {}))
                    return X_enc
                cat_steps.append(('encoder', FunctionTransformer(freq_encode)))
            transformers.append(('cat', Pipeline(cat_steps), categorical_cols))
        
        preprocessor = ColumnTransformer(transformers, remainder='drop')
        
        task = st.session_state.task_type
        if task == "sınıflandırma":
            model_name = st.selectbox("Model", ["Lojistik Regresyon", "Random Forest", "XGBoost", "SVM"])
            st.subheader("Hiperparametreler (manuel)")
            if model_name == "Lojistik Regresyon":
                C = st.number_input("C (regularizasyon)", 0.01, 10.0, 1.0)
                model = LogisticRegression(C=C, max_iter=1000)
            elif model_name == "Random Forest":
                n_est = st.slider("n_estimators", 10, 300, 100)
                max_depth = st.slider("max_depth", 2, 30, 10)
                model = RandomForestClassifier(n_estimators=n_est, max_depth=max_depth)
            elif model_name == "XGBoost":
                n_est = st.slider("n_estimators", 10, 300, 100)
                lr = st.number_input("learning_rate", 0.01, 0.5, 0.1)
                model = xgb.XGBClassifier(n_estimators=n_est, learning_rate=lr, eval_metric='logloss')
            else:
                C = st.number_input("C", 0.01, 10.0, 1.0)
                kernel = st.selectbox("kernel", ["rbf", "linear", "poly"])
                model = SVC(C=C, kernel=kernel, probability=True)
        elif task == "regresyon":
            model_name = st.selectbox("Model", ["Linear Regresyon", "Lasso", "Random Forest", "XGBoost"])
            if model_name == "Linear Regresyon":
                model = LinearRegression()
            elif model_name == "Lasso":
                alpha = st.number_input("alpha", 0.001, 10.0, 1.0)
                model = Lasso(alpha=alpha)
            elif model_name == "Random Forest":
                n_est = st.slider("n_estimators", 10, 300, 100)
                model = RandomForestRegressor(n_estimators=n_est)
            else:
                n_est = st.slider("n_estimators", 10, 300, 100)
                lr = st.number_input("learning_rate", 0.01, 0.5, 0.1)
                model = xgb.XGBRegressor(n_estimators=n_est, learning_rate=lr)
        else:
            model_name = st.selectbox("Model", ["K-Means", "DBSCAN"])
            if model_name == "K-Means":
                n_clusters = st.slider("küme sayısı", 2, 10, 3)
                model = KMeans(n_clusters=n_clusters, n_init=10)
            else:
                eps = st.number_input("eps", 0.1, 2.0, 0.5)
                min_samples = st.slider("min_samples", 2, 20, 5)
                model = DBSCAN(eps=eps, min_samples=min_samples)
        
        test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100
        cv_folds = st.slider("Cross-Validation kat sayısı", 2, 10, 5)
        
        if st.button("Modeli Eğit (Data Leakage Yok)"):
            with st.spinner("Eğitim başladı..."):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task=="sınıflandırma" else None)
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
                pipeline.fit(X_train, y_train)
                st.session_state.model = pipeline
                
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)
                
                if task == "sınıflandırma":
                    train_acc = accuracy_score(y_train, y_pred_train)
                    test_acc = accuracy_score(y_test, y_pred_test)
                    f1 = f1_score(y_test, y_pred_test, average='weighted')
                    st.session_state.metrics = {
                        'train_accuracy': train_acc,
                        'test_accuracy': test_acc,
                        'f1_score': f1,
                        'overfit': (train_acc - test_acc) > 0.1
                    }
                elif task == "regresyon":
                    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    r2 = r2_score(y_test, y_pred_test)
                    st.session_state.metrics = {
                        'train_rmse': train_rmse,
                        'test_rmse': test_rmse,
                        'r2': r2,
                        'overfit': (train_rmse < test_rmse * 0.8)
                    }
                else:
                    X_processed = preprocessor.fit_transform(X)
                    model.fit(X_processed)
                    if hasattr(model, 'labels_') and len(set(model.labels_)) > 1:
                        sil = silhouette_score(X_processed, model.labels_)
                    else:
                        sil = -1
                    st.session_state.metrics = {'silhouette': sil}
                
                if hasattr(pipeline.named_steps['model'], 'feature_importances_'):
                    st.session_state.feature_importance = pipeline.named_steps['model'].feature_importances_
                
                st.success("Eğitim tamamlandı!")
                st.session_state.stage = "📈 Sonuçlar"
                st.rerun()

# ----- 7. SONUÇLAR -----
elif st.session_state.stage == "📈 Sonuçlar":
    st.header("Model Performansı ve İstatistiksel Değerlendirme")
    if st.session_state.metrics is None:
        st.warning("Model eğitilmedi.")
    else:
        metrics = st.session_state.metrics
        task = st.session_state.task_type
        if task != "kümeleme":
            col1, col2 = st.columns(2)
            if 'test_accuracy' in metrics:
                col1.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
                col2.metric("Train Accuracy", f"{metrics['train_accuracy']:.2%}")
                if metrics['overfit']:
                    st.error("⚠️ **Overfit tespit edildi!** Eğitim ve test başarımı arasında büyük fark var.")
                else:
                    st.success("✅ Overfit gözlenmedi.")
                st.metric("F1 Score (weighted)", f"{metrics['f1_score']:.3f}")
            else:
                col1.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")
                col2.metric("Train RMSE", f"{metrics['train_rmse']:.2f}")
                st.metric("R²", f"{metrics['r2']:.3f}")
                if metrics['overfit']:
                    st.error("⚠️ **Overfit riski:** Train RMSE test RMSE'den çok düşük.")
        else:
            st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")
        
        if st.session_state.feature_importance is not None:
            st.subheader("Önemli Özellikler (Feature Importance)")
            st.bar_chart(st.session_state.feature_importance)
        
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
        col_types = st.session_state.col_types
        X_columns = [c for c in df.columns if c not in drop and c != target]
        user_inputs = {}
        with st.form("tahmin_formu"):
            for col in X_columns:
                tip = col_types.get(col, "numeric")
                if tip == "numeric":
                    min_val = float(df[col].min())
                    max_val = float(df[col].max())
                    mean_val = float(df[col].mean())
                    st.markdown(f"**{col}** (min: {min_val:.2f}, max: {max_val:.2f}, ortalama: {mean_val:.2f})")
                    val = st.number_input(f"{col}", value=mean_val, step=0.1, key=col)
                else:
                    unique_vals = df[col].dropna().unique().tolist()
                    st.markdown(f"**{col}** (Kategorik: {', '.join(map(str, unique_vals[:5]))})")
                    val = st.selectbox(f"{col}", options=unique_vals, key=col)
                user_inputs[col] = val
            submitted = st.form_submit_button("Tahmin Et")
            if submitted:
                input_df = pd.DataFrame([user_inputs])
                model = st.session_state.model
                try:
                    pred = model.predict(input_df)[0]
                    st.success(f"🎯 Tahmin sonucu: {pred}")
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}")
