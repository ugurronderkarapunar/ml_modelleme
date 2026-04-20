import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix, r2_score, mean_absolute_error, mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import shap

warnings.filterwarnings('ignore')

# ------------------ KONFİG ------------------
st.set_page_config(page_title="AutoML Studio", layout="wide")
st.title("🧠 Senior Full Stack Data Science Studio")
st.markdown("**Veri Yükleme → EDA → Problem Tanımı → Feature Engineering → Modelleme → Ensemble → Yorumlama**")

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.df_cleaned = None
    st.session_state.target = None
    st.session_state.problem_type = None
    st.session_state.models = {}
    st.session_state.best_model = None

# ------------------ YARDIMCI FONKSİYONLAR ------------------
def handle_missing_values(df, num_strategy='median', cat_strategy='most_frequent'):
    """Eksik değerleri kullanıcı stratejisine göre doldur"""
    df_out = df.copy()
    num_cols = df_out.select_dtypes(include=[np.number]).columns
    cat_cols = df_out.select_dtypes(include=['object', 'category']).columns
    
    for col in num_cols:
        if num_strategy == 'median':
            df_out[col].fillna(df_out[col].median(), inplace=True)
        elif num_strategy == 'mean':
            df_out[col].fillna(df_out[col].mean(), inplace=True)
        elif num_strategy == 'zero':
            df_out[col].fillna(0, inplace=True)
    
    for col in cat_cols:
        if cat_strategy == 'most_frequent':
            mode_val = df_out[col].mode()
            if not mode_val.empty:
                df_out[col].fillna(mode_val[0], inplace=True)
            else:
                df_out[col].fillna('Unknown', inplace=True)
        elif cat_strategy == 'new_category':
            df_out[col].fillna('Missing', inplace=True)
    return df_out

def detect_outliers_iqr(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (df[col] < lower) | (df[col] > upper)

def remove_outliers(df, col, method='iqr'):
    if method == 'iqr':
        outliers = detect_outliers_iqr(df, col)
        return df[~outliers]
    return df

# ------------------ SIDEBAR: VERİ YÜKLEME ------------------
with st.sidebar:
    st.header("1️⃣ Veri Yükleme")
    uploaded_file = st.file_uploader("CSV veya Excel", type=['csv', 'xlsx'])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.session_state.df = df
            st.success(f"{df.shape[0]} satır, {df.shape[1]} sütun")
        except Exception as e:
            st.error(f"Hata: {e}")

if st.session_state.df is None:
    st.info("Lütfen bir veri seti yükleyin.")
    st.stop()

df = st.session_state.df.copy()

# ------------------ TAB 1: VERİ YÜKLEME & İLK BAKIŞ ------------------
with st.expander("📂 Veri Yükleme & İlk Bakış", expanded=True):
    st.subheader("İlk 10 Gözlem")
    st.dataframe(df.head(10))
    st.subheader("Veri Bilgileri")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())
    st.subheader("İstatistiksel Özet")
    st.dataframe(df.describe(include='all'))

# ------------------ TAB 2: GENEL EDA ------------------
with st.expander("📊 Genel Keşifsel Veri Analizi (EDA)"):
    col_sel = st.selectbox("Değişken seçin", df.columns)
    
    if df[col_sel].dtype in ['int64', 'float64']:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col_sel], kde=True, ax=axes[0])
        sns.boxplot(y=df[col_sel], ax=axes[1])
        st.pyplot(fig)
        
        # Normallik testi
        from scipy.stats import shapiro, normaltest
        if len(df[col_sel].dropna()) >= 3:
            stat, p = shapiro(df[col_sel].dropna())
            st.write(f"**Shapiro-Wilk p-value:** {p:.4f} → {'Normal dağılım gösteriyor' if p>0.05 else 'Normal dağılım göstermiyor'}")
    else:
        st.bar_chart(df[col_sel].value_counts().head(10))
    
    # Korelasyon matrisi
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        st.subheader("Korelasyon Matrisi")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ------------------ TAB 3: PROBLEM TANIMI ------------------
with st.expander("🎯 Problem Tanımı"):
    st.session_state.problem_type = st.radio(
        "Problem tipini seçin:",
        ["Sınıflandırma", "Regresyon", "Öneri Sistemi", "Kümeleme"]
    )
    st.session_state.target = st.selectbox("Hedef Değişken (Y)", df.columns)
    drop_cols = st.multiselect("Çıkarılacak gereksiz değişkenler", [c for c in df.columns if c != st.session_state.target])
    X = df.drop(columns=[st.session_state.target] + drop_cols)
    y = df[st.session_state.target]

# ------------------ TAB 4: VERİ ÖN İŞLEME ------------------
with st.expander("🛠️ Veri Ön İşleme Adımları"):
    st.subheader("Tip Dönüşümleri")
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                st.info(f"{col} sayısala çevrildi")
            except:
                pass
    
    st.subheader("Değişken Sınıflandırma")
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    st.write(f"**Sayısal değişkenler:** {numeric_cols}")
    st.write(f"**Kategorik değişkenler:** {categorical_cols}")
    
    st.subheader("Eksik Veri Analizi")
    missing = X.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing)
        num_strategy = st.selectbox("Sayısal eksik stratejisi", ["median", "mean", "zero"])
        cat_strategy = st.selectbox("Kategorik eksik stratejisi", ["most_frequent", "new_category"])
    else:
        num_strategy = "median"
        cat_strategy = "most_frequent"
        st.success("Hiç eksik veri yok!")
    
    st.subheader("Aykırı Değer Analizi")
    outlier_method = st.selectbox("Aykırı değer temizleme yöntemi", ["iqr", "zscore", "none"])
    if outlier_method != "none":
        for col in numeric_cols:
            X = remove_outliers(X, col, method=outlier_method)
            y = y.loc[X.index]  # senkronize
        st.success(f"Aykırı değerler temizlendi. Yeni boyut: {X.shape}")
    
    # Eksik değer doldurma
    X = handle_missing_values(X, num_strategy, cat_strategy)
    st.success("Eksik değerler dolduruldu.")
    
    st.subheader("Korelasyon & İstatistik")
    if numeric_cols:
        corr = X[numeric_cols].corr()
        st.dataframe(corr)
    
    st.subheader("EDA Görselleştirme")
    # Güvenli boxplot: boş grupları engelle
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        for cat in categorical_cols[:2]:  # ilk 2 kategorik
            if X[cat].nunique() <= 10:
                fig, ax = plt.subplots()
                # Boş grupları filtrele
                plot_data = X[[cat, numeric_cols[0]]].dropna()
                if plot_data[cat].nunique() > 0:
                    sns.boxplot(data=plot_data, x=cat, y=numeric_cols[0], ax=ax)
                    st.pyplot(fig)
    
    st.subheader("Feature Engineering")
    if st.checkbox("Polinomik özellikler ekle"):
        from sklearn.preprocessing import PolynomialFeatures
        poly = PolynomialFeatures(degree=2, include_bias=False)
        num_feat = poly.fit_transform(X[numeric_cols])
        X = pd.concat([X, pd.DataFrame(num_feat, columns=poly.get_feature_names_out(numeric_cols))], axis=1)
        st.success("Polinomik özellikler eklendi.")

# ------------------ TAB 5: TRAIN-TEST SPLIT ------------------
with st.expander("🔀 Train-Test Split"):
    test_size = st.slider("Test seti oranı (%)", 10, 40, 20) / 100
    cv_folds = st.slider("Cross-validation kat sayısı", 3, 10, 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.write(f"Eğitim: {X_train.shape[0]}, Test: {X_test.shape[0]}")

# ------------------ TAB 6: MODEL PIPELINE ------------------
with st.expander("⚙️ Model Pipeline & Karşılaştırma"):
    # Preprocessor
    num_transformer = Pipeline([('scaler', RobustScaler())])
    cat_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ])
    
    # Model seçenekleri
    models = {}
    if st.session_state.problem_type == "Sınıflandırma":
        models = {
            "Lojistik Regresyon": LogisticRegression(max_iter=1000),
            "Karar Ağacı": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "XGBoost": XGBClassifier(eval_metric='logloss'),
            "Gradient Boosting": GradientBoostingClassifier()
        }
    elif st.session_state.problem_type == "Regresyon":
        models = {
            "Ridge Regresyon": Ridge(),
            "Karar Ağacı": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "XGBoost": XGBRegressor(),
            "Gradient Boosting": GradientBoostingRegressor()
        }
    
    if st.button("Model Karşılaştırmasını Başlat"):
        results = []
        for name, model in models.items():
            pipe = Pipeline([('prep', preprocessor), ('model', model)])
            # Cross-validation
            cv_score = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring='accuracy' if st.session_state.problem_type=="Sınıflandırma" else 'r2')
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            if st.session_state.problem_type == "Sınıflandırma":
                acc = accuracy_score(y_test, y_pred)
                results.append({"Model": name, "CV Ortalama": cv_score.mean(), "Test Accuracy": acc})
            else:
                r2 = r2_score(y_test, y_pred)
                results.append({"Model": name, "CV Ortalama (R²)": cv_score.mean(), "Test R²": r2})
        st.dataframe(pd.DataFrame(results))
        st.session_state.results_df = pd.DataFrame(results)

# ------------------ TAB 7: HİPERPARAMETRE TUNING ------------------
with st.expander("🎛️ Hiperparametre Tuning (GridSearch)"):
    selected_model = st.selectbox("Tuning yapılacak modeli seçin", list(models.keys()) if models else [])
    if selected_model and st.button("GridSearch Başlat"):
        base_model = models[selected_model]
        param_grid = {}
        if "Random Forest" in selected_model:
            param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10]}
        elif "XGBoost" in selected_model:
            param_grid = {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
        else:
            param_grid = {}  # basit tanım
        
        pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
        grid = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring='accuracy' if st.session_state.problem_type=="Sınıflandırma" else 'r2', n_jobs=-1)
        grid.fit(X_train, y_train)
        st.success(f"En iyi parametreler: {grid.best_params_}")
        st.metric("En iyi skor", f"{grid.best_score_:.4f}")
        st.session_state.best_model = grid.best_estimator_

# ------------------ TAB 8: STACKING & ENSEMBLE ------------------
with st.expander("🧬 Stacking & Ensemble"):
    if st.button("Stacking Modeli Oluştur"):
        if st.session_state.problem_type == "Sınıflandırma":
            estimators = [('rf', RandomForestClassifier(n_estimators=50)), ('xgb', XGBClassifier(n_estimators=50))]
            stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
        else:
            estimators = [('rf', RandomForestRegressor(n_estimators=50)), ('xgb', XGBRegressor(n_estimators=50))]
            stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
        
        pipe_stack = Pipeline([('prep', preprocessor), ('model', stack_model)])
        pipe_stack.fit(X_train, y_train)
        y_pred_stack = pipe_stack.predict(X_test)
        if st.session_state.problem_type == "Sınıflandırma":
            st.metric("Stacking Accuracy", f"{accuracy_score(y_test, y_pred_stack):.4f}")
        else:
            st.metric("Stacking R²", f"{r2_score(y_test, y_pred_stack):.4f}")
        st.session_state.stack_model = pipe_stack

# ------------------ TAB 9: MODEL YORUMLAMA ------------------
with st.expander("📖 Model Yorumlama (SHAP)"):
    if st.session_state.best_model is not None:
        try:
            # SHAP için basitleştirilmiş
            model = st.session_state.best_model.named_steps['model']
            X_sample = X_test.sample(min(100, len(X_test)))
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(preprocessor.transform(X_sample))
            st.pyplot(shap.summary_plot(shap_values, X_sample, show=False))
        except Exception as e:
            st.info(f"SHAP yorumlaması yapılamadı: {e}")
    else:
        st.info("Önce bir model eğitin (GridSearch veya Stacking).")

# ------------------ TAB 10: TAHMIN (INFERENCE) ------------------
with st.expander("🔮 Yeni Veri ile Tahmin"):
    if st.session_state.best_model is not None:
        st.subheader("Manuel Giriş")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.text_input(col, "0")
        if st.button("Tahmin Et"):
            input_df = pd.DataFrame([input_data])
            input_df = handle_missing_values(input_df, num_strategy, cat_strategy)
            pred = st.session_state.best_model.predict(input_df)
            st.success(f"Tahmin: {pred[0]}")
    else:
        st.info("Lütfen önce bir model eğitin.")
