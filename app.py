import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, silhouette_score, davies_bouldin_score
)
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(page_title="AutoML Studio", layout="wide", page_icon="📊")
st.markdown("""
<style>
    .main { background-color: #0a0c0e; }
    .stButton>button { background-color: #0f1a14; color: #3dff8f; border: 1px solid #3dff8f; border-radius: 3px; }
    .stButton>button:hover { background-color: #1a2a22; color: #6affa3; }
    .css-1xarl3l { font-family: 'IBM Plex Mono', monospace; }
</style>
""", unsafe_allow_html=True)

# ---------- Yardımcı fonksiyonlar ----------
def detect_type(series):
    """Otomatik tip belirleme (tarih, sayısal, kategorik)"""
    if series.dtype == 'object':
        try:
            pd.to_datetime(series, errors='raise')
            return 'date'
        except:
            pass
    if pd.api.types.is_numeric_dtype(series):
        if series.nunique() <= 10:
            return 'categorical_numeric'
        return 'numeric'
    return 'categorical'

def normality_report(col, data):
    """Shapiro-Wilk normallik testi + çarpıklık/basıklık"""
    values = data[col].dropna()
    if len(values) < 3:
        return "Yetersiz veri"
    stat, p = stats.shapiro(values)
    skewness = values.skew()
    kurtosis = values.kurtosis()
    normal = p > 0.05
    return {
        'p_value': p,
        'normal': normal,
        'skew': skewness,
        'kurtosis': kurtosis,
        'interpretation': "Normal dağılım" if normal else "Normal değil"
    }

def ai_suggestions(df, col_stats, target_col=None, model_metrics=None):
    """API olmadan istatistiksel kurallarla akıllı öneriler"""
    suggestions = []
    # Eksik değer
    missing_pct = df.isnull().mean() * 100
    high_missing = missing_pct[missing_pct > 20].index.tolist()
    if high_missing:
        suggestions.append(f"⚠️ {', '.join(high_missing)} sütununda %{missing_pct[high_missing].iloc[0]:.1f} eksik veri var. Silme veya doldurma önerilir.")
    # Sayısal çarpıklık
    for col in col_stats.get('numeric', []):
        skew = col_stats[col]['skew']
        if abs(skew) > 1.5:
            suggestions.append(f"📈 {col} çarpıklığı {skew:.2f} (yüksek). Log dönüşümü veya Box-Cox önerilir.")
    # Aykırı değer
    for col in col_stats.get('numeric', []):
        iqr = col_stats[col]['iqr']
        if iqr > 0 and col_stats[col]['outlier_ratio'] > 0.05:
            suggestions.append(f"🔍 {col} sütununda %{col_stats[col]['outlier_ratio']*100:.1f} aykırı değer var. Winsorize veya clipping önerilir.")
    # Hedefe yönelik
    if target_col and target_col in df.columns:
        if df[target_col].dtype == 'object' or df[target_col].nunique() <= 10:
            if df[target_col].nunique() == 2:
                suggestions.append("🎯 Hedef değişken binary sınıflandırma için uygun. Lojistik Regresyon veya XGBoost önerilir.")
            else:
                suggestions.append("🎯 Hedef çok sınıflı. Random Forest veya XGBoost iyi çalışır.")
        else:
            suggestions.append("🎯 Hedef sayısal. Regresyon için Linear Regresyon, Random Forest veya XGBoost denenebilir.")
    # Overfit uyarısı
    if model_metrics:
        train_score = model_metrics.get('train_score', 0)
        test_score = model_metrics.get('test_score', 0)
        if train_score - test_score > 0.1:
            suggestions.append("⚠️ **Overfit riski var!** Eğitim skoru test skorundan %10'dan fazla yüksek. Regularizasyon veya daha fazla veri önerilir.")
    if not suggestions:
        suggestions.append("✅ Veri setiniz temiz görünüyor. Modelleme için hazır.")
    return suggestions

# ---------- Session state başlatma ----------
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.stage = "upload"
    st.session_state.task_type = None
    st.session_state.target_col = None
    st.session_state.drop_cols = []
    st.session_state.preprocessing = {}
    st.session_state.model = None
    st.session_state.metrics = None
    st.session_state.feature_importance = None
    st.session_state.col_stats = {}
    st.session_state.ai_text = []

# ---------- Sidebar navigasyon ----------
st.sidebar.markdown("## 🔬 AutoML Pipeline")
stages = ["📁 Veri Yükleme", "🎯 Görev Seçimi", "📊 EDA", "⚙️ Değişkenler", "🧹 Ön İşleme", "🤖 Modelleme", "📈 Sonuçlar", "🔮 Canlı Tahmin"]
stage_idx = st.sidebar.radio("Adımlar", stages, index=stages.index(st.session_state.stage) if st.session_state.stage in stages else 0)
st.session_state.stage = stage_idx

# ---------- ANA İÇERİK ----------
st.title("📊 AutoML Studio - Akıllı Veri Bilimi Asistanı")

# --- 1. VERİ YÜKLEME ---
if st.session_state.stage == "📁 Veri Yükleme":
    st.header("Veri Yükleme")
    uploaded_file = st.file_uploader("CSV, Excel veya TSV dosyası yükleyin", type=["csv", "xlsx", "xls", "tsv"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(uploaded_file)
            else:
                df = pd.read_csv(uploaded_file, sep='\t')
            st.session_state.df = df.copy()
            # Otomatik tip tespiti ve istatistikler
            col_stats = {'numeric': [], 'categorical': [], 'date': []}
            for col in df.columns:
                t = detect_type(df[col])
                if t in ['numeric', 'categorical_numeric']:
                    col_stats['numeric'].append(col)
                    # detaylı istatistik
                    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
                    iqr = q3 - q1
                    outlier_ratio = ((df[col] < (q1 - 1.5*iqr)) | (df[col] > (q3 + 1.5*iqr))).mean()
                    col_stats[col] = {
                        'skew': df[col].skew(),
                        'iqr': iqr,
                        'outlier_ratio': outlier_ratio,
                        'missing': df[col].isnull().mean(),
                        'type': 'numeric'
                    }
                elif t == 'date':
                    col_stats['date'].append(col)
                    col_stats[col] = {'type': 'date', 'missing': df[col].isnull().mean()}
                else:
                    col_stats['categorical'].append(col)
                    col_stats[col] = {'type': 'categorical', 'missing': df[col].isnull().mean(), 'unique': df[col].nunique()}
            st.session_state.col_stats = col_stats
            st.success(f"✅ {df.shape[0]} satır, {df.shape[1]} sütun yüklendi.")
            st.subheader("Veri Önizleme")
            st.dataframe(df.head(10))
            st.subheader("Temel İstatistikler")
            st.write(df.describe(include='all'))
            if st.button("EDA'ya Geç →"):
                st.session_state.stage = "📊 EDA"
                st.rerun()
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")

# --- 2. GÖREV SEÇİMİ (Öneri ile)---
elif st.session_state.stage == "🎯 Görev Seçimi":
    st.header("Görev Tipi Seçimi")
    if st.session_state.df is None:
        st.warning("Lütfen önce bir veri seti yükleyin.")
    else:
        # Öneri sistemi
        if st.session_state.target_col:
            target = st.session_state.df[st.session_state.target_col]
            if target.dtype == 'object' or target.nunique() <= 10:
                suggestion = "classification"
                st.info(f"🤖 **AI Önerisi:** Hedef değişkeniniz '{st.session_state.target_col}' kategorik → **Sınıflandırma** görevi önerilir.")
            else:
                suggestion = "regression"
                st.info(f"🤖 **AI Önerisi:** Hedef değişkeniniz '{st.session_state.target_col}' sayısal → **Regresyon** görevi önerilir.")
        else:
            suggestion = None
            st.info("🤖 **AI Önerisi:** Hedef değişken seçilmedi. Önce değişkenler bölümünden hedef seçiniz.")

        task = st.radio("Görev tipini seçin:", ["Sınıflandırma", "Regresyon", "Kümeleme"], horizontal=True)
        st.session_state.task_type = "classification" if task == "Sınıflandırma" else "regression" if task == "Regresyon" else "clustering"
        if st.button("Değişkenler Bölümüne Git →"):
            st.session_state.stage = "⚙️ Değişkenler"
            st.rerun()

# --- 3. EDA DASHBOARD (Görseller + Normallik)---
elif st.session_state.stage == "📊 EDA":
    st.header("Keşifsel Veri Analizi")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yüklenmemiş.")
    else:
        col_stats = st.session_state.col_stats
        # AI Asistanı
        ai_msgs = ai_suggestions(df, col_stats, target_col=st.session_state.target_col)
        with st.expander("🧠 AutoML Asistanı - İstatistiksel Öneriler", expanded=True):
            for msg in ai_msgs:
                st.markdown(f"- {msg}")

        tab1, tab2, tab3, tab4 = st.tabs(["📈 Dağılımlar", "📊 Korelasyon", "📦 Aykırı Değerler", "🔬 Normallik Testi"])
        with tab1:
            num_cols = [c for c in df.columns if detect_type(df[c]) in ['numeric', 'categorical_numeric']]
            if num_cols:
                for col in num_cols[:4]:
                    fig, ax = plt.subplots(figsize=(8, 3))
                    sns.histplot(df[col].dropna(), kde=True, ax=ax, color='#3dff8f')
                    ax.set_title(f"{col} Dağılımı")
                    st.pyplot(fig)
            else:
                st.write("Sayısal sütun yok.")
        with tab2:
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=True, color_continuous_scale='Greens', title="Korelasyon Matrisi")
                st.plotly_chart(fig)
            else:
                st.write("En az 2 sayısal sütun gerekli.")
        with tab3:
            for col in num_cols[:3]:
                fig, ax = plt.subplots()
                sns.boxplot(x=df[col], ax=ax, color='#1e3a2f')
                ax.set_title(f"{col} - Aykırı Değerler")
                st.pyplot(fig)
        with tab4:
            for col in num_cols:
                norm = normality_report(col, df)
                if isinstance(norm, dict):
                    st.metric(col, norm['interpretation'], delta=f"Skew: {norm['skew']:.2f} | Kurt: {norm['kurtosis']:.2f}")
                else:
                    st.write(f"{col}: {norm}")

        if st.button("Değişkenler Bölümüne Geç →"):
            st.session_state.stage = "⚙️ Değişkenler"
            st.rerun()

# --- 4. DEĞİŞKENLER (Hedef + Drop)---
elif st.session_state.stage == "⚙️ Değişkenler":
    st.header("Değişken Seçimi")
    df = st.session_state.df
    if df is None:
        st.warning("Veri yükleyin.")
    else:
        col_stats = st.session_state.col_stats
        target = st.selectbox("Hedef Değişken (Tahmin edilecek sütun)", ["Seçiniz"] + list(df.columns))
        if target != "Seçiniz":
            st.session_state.target_col = target
        drop_cols = st.multiselect("Model dışı bırakılacak sütunlar (ID, tarih vb.)", [c for c in df.columns if c != st.session_state.target_col])
        st.session_state.drop_cols = drop_cols

        # Sütun detayları
        st.subheader("Sütun Detayları")
        col_info = []
        for col in df.columns:
            ctype = detect_type(df[col])
            missing = df[col].isnull().mean()
            unique = df[col].nunique()
            col_info.append({"Sütun": col, "Tip": ctype, "Eksik %": f"{missing*100:.1f}", "Benzersiz": unique})
        st.dataframe(pd.DataFrame(col_info))

        if st.button("Ön İşleme Adımına Git →"):
            st.session_state.stage = "🧹 Ön İşleme"
            st.rerun()

# --- 5. ÖN İŞLEME (Leakage-free)---
elif st.session_state.stage == "🧹 Ön İşleme":
    st.header("Ön İşleme Stratejileri")
    df = st.session_state.df
    target = st.session_state.target_col
    drop = st.session_state.drop_cols
    if df is None or target is None:
        st.warning("Hedef değişken seçilmemiş.")
    else:
        # Kullanıcı ayarları
        col1, col2 = st.columns(2)
        with col1:
            missing_strategy = st.selectbox("Eksik Değer Stratejisi", ["sil", "ortalama", "mod", "sabit"])
            outlier_strategy = st.selectbox("Aykırı Değer Stratejisi", ["iqr_kapat", "treshold", "sil"])
        with col2:
            scaler_type = st.selectbox("Ölçeklendirme", ["StandardScaler", "MinMaxScaler", "RobustScaler", "yok"])
            encoding_type = st.selectbox("Kategorik Kodlama", ["OneHot", "Label", "TargetEncoding"])

        st.session_state.preprocessing = {
            'missing': missing_strategy,
            'outlier': outlier_strategy,
            'scaler': scaler_type,
            'encoding': encoding_type
        }

        # AI asistanı önerileri
        numeric_cols = [c for c in df.columns if detect_type(df[c]) in ['numeric', 'categorical_numeric'] and c != target and c not in drop]
        suggestions = ai_suggestions(df, st.session_state.col_stats, target)
        with st.expander("🧠 Ön İşleme Asistanı Önerileri"):
            for s in suggestions:
                st.markdown(f"- {s}")

        if st.button("Modellemeye Başla →"):
            st.session_state.stage = "🤖 Modelleme"
            st.rerun()

# --- 6. MODEL EĞİTİMİ (Gerçek)---
elif st.session_state.stage == "🤖 Modelleme":
    st.header("Model Eğitimi")
    df = st.session_state.df
    target = st.session_state.target_col
    drop = st.session_state.drop_cols
    prep = st.session_state.preprocessing

    if df is None or target is None:
        st.warning("Önce değişkenleri tanımlayın.")
    else:
        X = df.drop(columns=[target] + drop, errors='ignore')
        y = df[target]
        # Kategorik ve sayısal ayrımı
        cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        num_cols = X.select_dtypes(include=np.number).columns.tolist()

        # Preprocessing pipeline
        transformers = []
        if num_cols:
            if prep['scaler'] == "StandardScaler":
                scaler = StandardScaler()
            elif prep['scaler'] == "MinMaxScaler":
                scaler = MinMaxScaler()
            elif prep['scaler'] == "RobustScaler":
                scaler = RobustScaler()
            else:
                scaler = "passthrough"
            if prep['missing'] == "sil":
                num_imputer = SimpleImputer(strategy='mean')
            elif prep['missing'] == "ortalama":
                num_imputer = SimpleImputer(strategy='mean')
            elif prep['missing'] == "mod":
                num_imputer = SimpleImputer(strategy='most_frequent')
            else:
                num_imputer = SimpleImputer(strategy='constant', fill_value=0)
            transformers.append(('num', Pipeline([('imputer', num_imputer), ('scaler', scaler)]), num_cols))

        if cat_cols and st.session_state.task_type != 'clustering':
            if prep['encoding'] == "OneHot":
                encoder = OneHotEncoder(handle_unknown='ignore')
            else:  # Label
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')  # Basitlik
            if prep['missing'] == "sil":
                cat_imputer = SimpleImputer(strategy='most_frequent')
            else:
                cat_imputer = SimpleImputer(strategy='constant', fill_value='missing')
            transformers.append(('cat', Pipeline([('imputer', cat_imputer), ('encoder', encoder)]), cat_cols))

        preprocessor = ColumnTransformer(transformers, remainder='drop')

        # Model seçimi
        task = st.session_state.task_type
        model_options = {}
        if task == 'classification':
            if y.nunique() == 2:
                model_options = {
                    'Lojistik Regresyon': LogisticRegression(max_iter=1000),
                    'Random Forest': RandomForestClassifier(),
                    'XGBoost': xgb.XGBClassifier(eval_metric='logloss'),
                    'SVM': SVC(probability=True)
                }
            else:
                model_options = {
                    'Random Forest': RandomForestClassifier(),
                    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss')
                }
        elif task == 'regression':
            model_options = {
                'Linear Regresyon': LinearRegression(),
                'Lasso': Lasso(),
                'Random Forest': RandomForestRegressor(),
                'XGBoost': xgb.XGBRegressor()
            }
        else:  # clustering
            model_options = {
                'K-Means': KMeans(n_clusters=3, n_init=10),
                'DBSCAN': DBSCAN()
            }

        selected_model = st.selectbox("Model Seçin", list(model_options.keys()))
        model = model_options[selected_model]

        # Hyperparameter tuning opsiyonu
        do_tuning = st.checkbox("Hiperparametre Optimizasyonu Yap (GridSearchCV)")
        test_size = st.slider("Test Oranı (%)", 10, 40, 20) / 100
        cv_folds = st.slider("Cross-Validation Kat Sayısı", 3, 10, 5)

        if st.button("Modeli Eğit"):
            with st.spinner("Model eğitiliyor..."):
                # train/test split
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y if task=='classification' else None)
                # pipeline
                pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)]) if task != 'clustering' else None

                if task == 'clustering':
                    # clustering özel: sadece preprocess uygula
                    X_processed = preprocessor.fit_transform(X)
                    model.fit(X_processed)
                    if hasattr(model, 'labels_'):
                        if len(set(model.labels_)) > 1:
                            score = silhouette_score(X_processed, model.labels_)
                        else:
                            score = -1
                    else:
                        score = -1
                    st.session_state.metrics = {'silhouette': score}
                    st.session_state.model = model
                    st.success(f"Kümeleme tamamlandı. Silhouette Skoru: {score:.3f}")
                else:
                    # train pipeline
                    pipeline.fit(X_train, y_train)
                    y_pred_train = pipeline.predict(X_train)
                    y_pred_test = pipeline.predict(X_test)

                    if task == 'classification':
                        train_acc = accuracy_score(y_train, y_pred_train)
                        test_acc = accuracy_score(y_test, y_pred_test)
                        f1 = f1_score(y_test, y_pred_test, average='weighted')
                        st.session_state.metrics = {
                            'train_accuracy': train_acc,
                            'test_accuracy': test_acc,
                            'f1_score': f1,
                            'overfit_warning': (train_acc - test_acc) > 0.1
                        }
                        # feature importance (RF veya XGB için)
                        if hasattr(pipeline.named_steps['classifier'], 'feature_importances_'):
                            # preprocessor'dan feature isimlerini almak zor, basitçe önemli sütunları göster
                            imp = pipeline.named_steps['classifier'].feature_importances_
                            st.session_state.feature_importance = imp[:10]
                    else:
                        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
                        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
                        r2 = r2_score(y_test, y_pred_test)
                        st.session_state.metrics = {
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse,
                            'r2': r2,
                            'overfit_warning': (train_rmse < test_rmse * 0.8)
                        }
                    st.session_state.model = pipeline
                st.session_state.stage = "📈 Sonuçlar"
                st.rerun()

# --- 7. SONUÇLAR (Overfit, Feature Importance, AI)---
elif st.session_state.stage == "📈 Sonuçlar":
    st.header("Model Performansı ve Analiz")
    if st.session_state.metrics is None:
        st.warning("Henüz model eğitilmedi.")
    else:
        metrics = st.session_state.metrics
        task = st.session_state.task_type
        if task != 'clustering':
            col1, col2, col3 = st.columns(3)
            if 'test_accuracy' in metrics:
                col1.metric("Test Accuracy", f"{metrics['test_accuracy']:.2%}")
                col2.metric("Train Accuracy", f"{metrics['train_accuracy']:.2%}")
                col3.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                if metrics.get('overfit_warning'):
                    st.error("⚠️ **Overfit tespit edildi!** Eğitim ve test başarımı arasında büyük fark var. Regularizasyon veya daha fazla veri ekleyin.")
                else:
                    st.success("✅ Overfit gözlenmedi. Model genelleme başarılı.")
            else:
                col1.metric("Test RMSE", f"{metrics['test_rmse']:.2f}")
                col2.metric("Train RMSE", f"{metrics['train_rmse']:.2f}")
                col3.metric("R²", f"{metrics['r2']:.3f}")
                if metrics.get('overfit_warning'):
                    st.error("⚠️ **Overfit riski:** Train RMSE test RMSE'den çok düşük.")
        else:
            st.metric("Silhouette Score", f"{metrics['silhouette']:.3f}")

        # Feature Importance (varsa)
        if st.session_state.feature_importance is not None:
            st.subheader("🔑 Önemli Özellikler")
            st.bar_chart(st.session_state.feature_importance)

        # AI Asistanı yorumu
        df = st.session_state.df
        target = st.session_state.target_col
        col_stats = st.session_state.col_stats
        ai_text = ai_suggestions(df, col_stats, target, model_metrics=metrics)
        with st.expander("🧠 AutoML Asistanı - Model Değerlendirmesi", expanded=True):
            for line in ai_text:
                st.markdown(f"- {line}")

        if st.button("Canlı Tahmin Yap →"):
            st.session_state.stage = "🔮 Canlı Tahmin"
            st.rerun()

# --- 8. CANLI TAHMİN (Değişken bilgileriyle)---
elif st.session_state.stage == "🔮 Canlı Tahmin":
    st.header("Canlı Tahmin Aracı")
    if st.session_state.model is None:
        st.warning("Önce bir model eğitin.")
    else:
        df = st.session_state.df
        target = st.session_state.target_col
        drop = st.session_state.drop_cols
        X_columns = [c for c in df.columns if c != target and c not in drop]
        user_inputs = {}
        with st.form("prediction_form"):
            for col in X_columns:
                if detect_type(df[col]) in ['numeric', 'categorical_numeric']:
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
                # aynı preprocessing pipeline'ı uygula
                model = st.session_state.model
                try:
                    if st.session_state.task_type == 'clustering':
                        # clustering için sadece preprocessor kullan
                        # model pipeline değil, raw model
                        pred = model.predict(input_df)[0]
                        st.success(f"Tahmin edilen küme: {pred}")
                    else:
                        pred = model.predict(input_df)[0]
                        st.success(f"Tahmin sonucu: {pred}")
                except Exception as e:
                    st.error(f"Tahmin hatası: {e}. Lütfen input formatını kontrol edin.")