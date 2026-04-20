import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Bilimsel ve makine öğrenmesi
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
                             classification_report, confusion_matrix, r2_score, mean_absolute_error,
                             mean_squared_error, mean_absolute_percentage_error, silhouette_score)
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif, mutual_info_regression

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# Öneri sistemleri
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# Gelişmiş görselleştirme
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Konfigürasyon
st.set_page_config(page_title="Senior Full-Stack Data Science Studio", layout="wide", page_icon="🧠")

# Session state başlangıcı
if 'step' not in st.session_state:
    st.session_state.update({
        'step': 1,  # 1: Veri Yükleme, 2: Genel EDA, 3: Problem Tanımı, 4: Hedef & Gereksiz Değişkenler, 5: Detaylı Preprocessing, 6: Train/Test Split, 7: Pipeline Konfigürasyonu, 8: Model Karşılaştırma, 9: Hiperparametre Tuning, 10: Cross Validation, 11: Stacking/Ensemble, 12: Model Yorumlama
        'df': None,
        'df_original': None,
        'target': None,
        'problem_type': None,  # 'classification', 'regression', 'recommendation', 'clustering'
        'exclude_cols': [],
        'numeric_cols': [],
        'categorical_cols': [],
        'preprocessing_config': {},
        'X_train': None, 'X_test': None, 'y_train': None, 'y_test': None,
        'models': {},
        'best_model': None,
        'best_params': {},
        'cv_results': {},
        'stacking_model': None,
        'feature_importance': None,
        'test_size': 0.2,
        'random_state': 42
    })

# Yardımcı fonksiyonlar
def load_data(file):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file, engine='openpyxl')
    return df

def show_first_rows(df):
    st.subheader("Veri Setinin İlk 10 Gözlemi")
    st.dataframe(df.head(10))

def detect_problem_type(target_col):
    unique_vals = target_col.nunique()
    if unique_vals < 20 or target_col.dtype == 'object' or target_col.dtype == 'category':
        return 'classification'
    else:
        return 'regression'

def split_data(X, y, test_size, random_state):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y if st.session_state.problem_type == 'classification' else None)

def get_model_list(problem_type):
    if problem_type == 'classification':
        return {
            'Lojistik Regresyon': LogisticRegression(max_iter=1000),
            'Karar Ağacı': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier(),
            'XGBoost': xgb.XGBClassifier(),
            'LightGBM': lgb.LGBMClassifier(),
            'CatBoost': cb.CatBoostClassifier(verbose=0),
            'SVM': SVC(probability=True)
        }
    elif problem_type == 'regression':
        return {
            'Lineer Regresyon': LinearRegression(),
            'Karar Ağacı': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': xgb.XGBRegressor(),
            'LightGBM': lgb.LGBMRegressor(),
            'CatBoost': cb.CatBoostRegressor(verbose=0),
            'SVR': SVR()
        }
    elif problem_type == 'clustering':
        return {
            'K-Means': KMeans(),
            'DBSCAN': DBSCAN(),
            'Hiyerarşik': AgglomerativeClustering()
        }
    else:  # recommendation
        return {}

def get_param_grid(model_name, problem_type):
    # Her model için örnek parametre grid'leri (kullanıcı seçebilir)
    grids = {
        'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
        'XGBoost': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'max_depth': [3, 5]},
        'LightGBM': {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1], 'num_leaves': [31, 50]},
        'CatBoost': {'iterations': [50, 100], 'learning_rate': [0.01, 0.1], 'depth': [3, 6]},
        'SVM': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        'Lojistik Regresyon': {'C': [0.1, 1, 10]},
        'Karar Ağacı': {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]},
        'Lineer Regresyon': {},
        'SVR': {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1]}
    }
    return grids.get(model_name, {})

def evaluate_classification(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr') if y_proba is not None and len(np.unique(y_test)) > 2 else None
    return {'Accuracy': acc, 'F1-Score': f1, 'Precision': prec, 'Recall': rec, 'AUC': auc}

def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    return {'R2': r2, 'MAE': mae, 'MSE': mse, 'RMSE': rmse, 'MAPE': mape}

def evaluate_clustering(model, X):
    if hasattr(model, 'labels_'):
        labels = model.labels_
    else:
        labels = model.fit_predict(X)
    if len(set(labels)) > 1:
        sil = silhouette_score(X, labels)
    else:
        sil = -1
    return {'Silhouette Score': sil, 'N_clusters': len(set(labels)) - (1 if -1 in labels else 0)}

# ------------------ ANA UYGULAMA ------------------
def main():
    st.title("🧠 Senior Full-Stack Data Science Studio")
    st.markdown("Otomatik makine öğrenmesi pipeline'ı, veri sızıntısı olmadan, ölçeklenebilir ve güvenilir.")

    # Sidebar'da adım takibi
    with st.sidebar:
        st.subheader("Adımlar")
        step_names = {
            1: "1. Veri Yükleme & İlk Bakış",
            2: "2. Genel EDA",
            3: "3. Problem Tanımı",
            4: "4. Hedef & Gereksiz Değişkenler",
            5: "5. Detaylı Preprocessing",
            6: "6. Train/Test Split",
            7: "7. Pipeline Konfigürasyonu",
            8: "8. Model Karşılaştırma",
            9: "9. Hiperparametre Tuning",
            10: "10. Cross Validation",
            11: "11. Stacking & Ensemble",
            12: "12. Model Yorumlama"
        }
        for step_num, step_name in step_names.items():
            if st.session_state.step == step_num:
                st.markdown(f"**→ {step_name}**")
            else:
                st.markdown(step_name)
        st.divider()
        if st.button("🔄 Sıfırla"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Adım 1: Veri Yükleme & İlk Bakış
    if st.session_state.step == 1:
        st.header("📂 Veri Yükleme & İlk Bakış")
        uploaded_file = st.file_uploader("CSV veya Excel dosyası seçin", type=['csv', 'xlsx'])
        if uploaded_file:
            df = load_data(uploaded_file)
            st.session_state.df = df.copy()
            st.session_state.df_original = df.copy()
            show_first_rows(df)
            st.info(f"Veri seti boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
            if st.button("İleri ➡️"):
                st.session_state.step = 2
                st.rerun()
        else:
            st.info("Lütfen bir dosya yükleyin.")

    # Adım 2: Genel EDA
    elif st.session_state.step == 2:
        st.header("📊 Genel Keşifsel Veri Analizi (EDA)")
        df = st.session_state.df
        # Temel bilgiler
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Toplam Satır", df.shape[0])
            st.metric("Toplam Sütun", df.shape[1])
            st.metric("Eksik Değer Oranı (%)", round(df.isnull().sum().sum() / (df.shape[0]*df.shape[1])*100, 2))
        with col2:
            st.metric("Sayısal Sütunlar", len(df.select_dtypes(include=np.number).columns))
            st.metric("Kategorik Sütunlar", len(df.select_dtypes(include='object').columns))
            st.metric("Benzersiz Değerler (ortalama)", round(df.nunique().mean(), 2))
        
        # Dağılım grafikleri
        st.subheader("Sayısal Değişken Dağılımları")
        numeric_cols = df.select_dtypes(include=np.number).columns
        if len(numeric_cols) > 0:
            selected_num = st.selectbox("Sütun seç", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[selected_num].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            st.info("Sayısal sütun yok.")
        
        st.subheader("Kategorik Değişken Frekansları")
        categorical_cols = df.select_dtypes(include='object').columns
        if len(categorical_cols) > 0:
            selected_cat = st.selectbox("Kategorik sütun seç", categorical_cols)
            st.dataframe(df[selected_cat].value_counts().head(10))
            fig, ax = plt.subplots()
            df[selected_cat].value_counts().head(10).plot(kind='bar', ax=ax)
            st.pyplot(fig)
        else:
            st.info("Kategorik sütun yok.")
        
        if st.button("İleri ➡️"):
            st.session_state.step = 3
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 1
            st.rerun()

    # Adım 3: Problem Tanımı
    elif st.session_state.step == 3:
        st.header("🎯 Problem Tanımı")
        problem_type = st.radio("Problem tipini seçin", ["Sınıflandırma", "Regresyon", "Kümeleme", "Öneri Sistemi"])
        problem_map = {"Sınıflandırma": "classification", "Regresyon": "regression", "Kümeleme": "clustering", "Öneri Sistemi": "recommendation"}
        st.session_state.problem_type = problem_map[problem_type]
        if st.button("İleri ➡️"):
            st.session_state.step = 4
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 2
            st.rerun()

    # Adım 4: Hedef Değişken & Gereksiz Değişkenler
    elif st.session_state.step == 4:
        st.header("🎯 Hedef Değişken ve Gereksiz Değişkenler")
        df = st.session_state.df
        all_cols = df.columns.tolist()
        target = st.selectbox("Hedef değişkeni seçin", all_cols)
        st.session_state.target = target
        exclude = st.multiselect("Çıkarılacak değişkenler (ID, tarih vb.)", [c for c in all_cols if c != target])
        st.session_state.exclude_cols = exclude
        if st.button("İleri ➡️"):
            # Hedef ve çıkarılacakları uygula
            df = df.drop(columns=exclude)
            st.session_state.df = df
            st.session_state.step = 5
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 3
            st.rerun()

    # Adım 5: Detaylı Preprocessing (Tip dönüşümleri, eksik veri, aykırı değer, korelasyon, EDA, feature engineering)
    elif st.session_state.step == 5:
        st.header("🔧 Detaylı Preprocessing")
        df = st.session_state.df
        target = st.session_state.target
        problem_type = st.session_state.problem_type
        
        # Tip dönüşümleri
        st.subheader("Tip Dönüşümleri")
        col_types = {}
        for col in df.columns:
            if col == target:
                continue
            new_type = st.selectbox(f"{col} tipi", ["numeric", "categorical", "datetime"], key=f"type_{col}")
            col_types[col] = new_type
        # Dönüşümleri uygula
        for col, typ in col_types.items():
            if typ == "numeric":
                df[col] = pd.to_numeric(df[col], errors='coerce')
            elif typ == "categorical":
                df[col] = df[col].astype('category')
            elif typ == "datetime":
                df[col] = pd.to_datetime(df[col], errors='coerce')
        st.session_state.df = df
        
        # Değişken sınıflandırma
        st.subheader("Değişken Sınıflandırma")
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include=['category', 'object']).columns.tolist()
        if target in numeric_cols:
            numeric_cols.remove(target)
        if target in categorical_cols:
            categorical_cols.remove(target)
        st.write(f"Sayısal değişkenler: {numeric_cols}")
        st.write(f"Kategorik değişkenler: {categorical_cols}")
        st.session_state.numeric_cols = numeric_cols
        st.session_state.categorical_cols = categorical_cols
        
        # Eksik veri analizi
        st.subheader("Eksik Veri Analizi")
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            st.dataframe(missing)
            # Kullanıcıdan strateji seçimi (daha sonra pipeline'da kullanılacak)
            num_impute_strategy = st.selectbox("Sayısal eksik değer stratejisi", ["median", "mean", "constant"], key="num_impute")
            cat_impute_strategy = st.selectbox("Kategorik eksik değer stratejisi", ["most_frequent", "constant"], key="cat_impute")
            st.session_state.preprocessing_config['num_impute'] = num_impute_strategy
            st.session_state.preprocessing_config['cat_impute'] = cat_impute_strategy
        else:
            st.success("Eksik değer yok.")
        
        # Aykırı değer analizi (sadece sayısal)
        st.subheader("Aykırı Değer Analizi")
        if numeric_cols:
            outlier_method = st.selectbox("Aykırı değer yöntemi", ["IQR", "Z-Score", "Isolation Forest"], key="outlier_method")
            st.session_state.preprocessing_config['outlier_method'] = outlier_method
            # Kullanıcıya aykırı değerleri görselleştir
            selected_num = st.selectbox("Aykırı değer gösterilecek sütun", numeric_cols)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[selected_num], ax=ax)
            st.pyplot(fig)
        else:
            st.info("Sayısal değişken yok, aykırı değer analizi atlanıyor.")
        
        # Korelasyon & İstatistik
        st.subheader("Korelasyon ve Normallik Testi")
        if len(numeric_cols) > 1:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        # Normallik testi (hedef değişken için)
        if target in df.columns and df[target].dtype in [np.number]:
            stat, p = stats.normaltest(df[target].dropna())
            st.write(f"Hedef değişken {target} için normallik testi p-değeri: {p:.4f} -> {'Normal dağılım' if p > 0.05 else 'Normal dağılım değil'}")
        
        # EDA Görselleştirme
        st.subheader("Gelişmiş EDA Görselleştirmeleri")
        plot_type = st.selectbox("Grafik türü", ["Histogram", "Kutu Grafiği", "Scatter", "Isı Haritası"])
        if plot_type == "Histogram":
            col = st.selectbox("Sütun", df.columns)
            fig, ax = plt.subplots()
            sns.histplot(df[col].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        elif plot_type == "Kutu Grafiği":
            col = st.selectbox("Sütun", df.columns)
            fig, ax = plt.subplots()
            sns.boxplot(x=df[col], ax=ax)
            st.pyplot(fig)
        elif plot_type == "Scatter":
            x_col = st.selectbox("X sütunu", numeric_cols)
            y_col = st.selectbox("Y sütunu", numeric_cols)
            fig, ax = plt.subplots()
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            st.pyplot(fig)
        elif plot_type == "Isı Haritası":
            if len(numeric_cols) > 1:
                fig, ax = plt.subplots()
                sns.heatmap(df[numeric_cols].corr(), annot=True, ax=ax)
                st.pyplot(fig)
            else:
                st.info("Yeterli sayısal sütun yok.")
        
        # Feature Engineering (basit örnek)
        st.subheader("Feature Engineering (Opsiyonel)")
        if st.checkbox("Yeni özellikler oluştur"):
            new_feat = st.text_input("Yeni özellik adı")
            expr = st.text_input("İfade (örnek: col1 + col2)")
            if new_feat and expr:
                try:
                    df[new_feat] = eval(expr)
                    st.success(f"{new_feat} eklendi.")
                    st.session_state.df = df
                    # Yeni sütunu uygun şekilde numeric/categorical olarak ekle
                    if df[new_feat].dtype == 'object':
                        st.session_state.categorical_cols.append(new_feat)
                    else:
                        st.session_state.numeric_cols.append(new_feat)
                except Exception as e:
                    st.error(f"Hata: {e}")
        
        if st.button("İleri ➡️"):
            st.session_state.step = 6
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 4
            st.rerun()

    # Adım 6: Train/Test Split
    elif st.session_state.step == 6:
        st.header("📊 Train/Test Split")
        test_size = st.slider("Test seti oranı (%)", 10, 40, 20) / 100
        st.session_state.test_size = test_size
        if st.button("Split'i Uygula"):
            df = st.session_state.df
            target = st.session_state.target
            X = df.drop(columns=[target])
            y = df[target]
            # Veri sızıntısı olmadan split
            X_train, X_test, y_train, y_test = split_data(X, y, test_size, st.session_state.random_state)
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.success(f"Train: {X_train.shape[0]} satır, Test: {X_test.shape[0]} satır")
            st.session_state.step = 7
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 5
            st.rerun()

    # Adım 7: Pipeline Konfigürasyonu (imputation, scaling, encoding, outlier handling)
    elif st.session_state.step == 7:
        st.header("🔧 Pipeline Konfigürasyonu")
        numeric_cols = st.session_state.numeric_cols
        categorical_cols = st.session_state.categorical_cols
        num_impute = st.session_state.preprocessing_config.get('num_impute', 'median')
        cat_impute = st.session_state.preprocessing_config.get('cat_impute', 'most_frequent')
        
        # Scaling seçimi
        scaler_type = st.selectbox("Sayısal ölçeklendirme", ["RobustScaler", "StandardScaler", "MinMaxScaler", "Hiçbiri"])
        scaler_map = {"RobustScaler": RobustScaler(), "StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "Hiçbiri": None}
        scaler = scaler_map[scaler_type]
        
        # Encoding seçimi
        encoder_type = st.selectbox("Kategorik kodlama", ["OneHotEncoder", "LabelEncoder (sadece hedef için)"])
        # Pipeline oluştur
        transformers = []
        if numeric_cols:
            num_pipe = []
            if num_impute:
                num_pipe.append(('impute', SimpleImputer(strategy=num_impute)))
            if scaler:
                num_pipe.append(('scale', scaler))
            if num_pipe:
                transformers.append(('num', Pipeline(num_pipe), numeric_cols))
        if categorical_cols:
            cat_pipe = []
            if cat_impute:
                cat_pipe.append(('impute', SimpleImputer(strategy=cat_impute)))
            if encoder_type == "OneHotEncoder":
                cat_pipe.append(('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
            if cat_pipe:
                transformers.append(('cat', Pipeline(cat_pipe), categorical_cols))
        
        preprocessor = ColumnTransformer(transformers) if transformers else None
        st.session_state.preprocessor = preprocessor
        st.success("Pipeline konfigürasyonu tamamlandı.")
        
        if st.button("İleri ➡️"):
            st.session_state.step = 8
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 6
            st.rerun()

    # Adım 8: Model Karşılaştırma
    elif st.session_state.step == 8:
        st.header("🤖 Model Karşılaştırma")
        problem_type = st.session_state.problem_type
        if problem_type in ['classification', 'regression']:
            models = get_model_list(problem_type)
            selected_models = st.multiselect("Karşılaştırılacak modelleri seçin", list(models.keys()), default=list(models.keys())[:3])
            if st.button("Modelleri Eğit ve Karşılaştır"):
                X_train = st.session_state.X_train
                y_train = st.session_state.y_train
                X_test = st.session_state.X_test
                y_test = st.session_state.y_test
                preprocessor = st.session_state.preprocessor
                results = {}
                for name in selected_models:
                    with st.spinner(f"{name} eğitiliyor..."):
                        model = models[name]
                        if preprocessor:
                            pipe = Pipeline([('prep', preprocessor), ('model', model)])
                        else:
                            pipe = Pipeline([('model', model)])
                        pipe.fit(X_train, y_train)
                        if problem_type == 'classification':
                            metrics = evaluate_classification(pipe, X_test, y_test)
                        else:
                            metrics = evaluate_regression(pipe, X_test, y_test)
                        results[name] = metrics
                        st.session_state.models[name] = pipe
                st.session_state.model_comparison = results
                # Sonuçları göster
                st.subheader("Karşılaştırma Sonuçları")
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)
                # En iyi modeli seç
                if problem_type == 'classification':
                    best_model = max(results, key=lambda x: results[x]['Accuracy'])
                else:
                    best_model = max(results, key=lambda x: results[x]['R2'])
                st.success(f"En iyi model: {best_model}")
                st.session_state.best_model_name = best_model
                st.session_state.best_model = st.session_state.models[best_model]
        elif problem_type == 'clustering':
            st.info("Kümeleme için model karşılaştırması henüz eklenmedi.")
        else:
            st.info("Öneri sistemi için model karşılaştırması farklı şekilde yapılacak.")
        
        if st.button("İleri ➡️"):
            st.session_state.step = 9
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 7
            st.rerun()

    # Adım 9: Hiperparametre Tuning
    elif st.session_state.step == 9:
        st.header("🔧 Hiperparametre Optimizasyonu")
        if 'best_model_name' in st.session_state:
            model_name = st.session_state.best_model_name
            param_grid = get_param_grid(model_name, st.session_state.problem_type)
            if param_grid:
                st.write(f"Seçilen model: {model_name}")
                st.json(param_grid)
                # Kullanıcıya parametreleri düzenleme imkanı
                new_params = {}
                for param, values in param_grid.items():
                    new_params[param] = st.multiselect(f"{param}", values, default=values)
                if st.button("Grid Search Başlat"):
                    model = get_model_list(st.session_state.problem_type)[model_name]
                    preprocessor = st.session_state.preprocessor
                    pipe = Pipeline([('prep', preprocessor), ('model', model)]) if preprocessor else Pipeline([('model', model)])
                    grid = GridSearchCV(pipe, param_grid={'model__' + k: v for k, v in new_params.items()}, cv=5, scoring='accuracy' if st.session_state.problem_type == 'classification' else 'r2', n_jobs=-1)
                    grid.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.best_model = grid.best_estimator_
                    st.session_state.best_params = grid.best_params_
                    st.success(f"En iyi parametreler: {grid.best_params_}")
                    st.write(f"En iyi skor: {grid.best_score_:.4f}")
            else:
                st.info("Bu model için parametre grid'i tanımlı değil, tuning atlanıyor.")
        else:
            st.warning("Önce model karşılaştırması yapmalısınız.")
        
        if st.button("İleri ➡️"):
            st.session_state.step = 10
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 8
            st.rerun()

    # Adım 10: Cross Validation
    elif st.session_state.step == 10:
        st.header("📈 Cross Validation")
        if 'best_model' in st.session_state:
            cv_folds = st.slider("Katman sayısı", 3, 10, 5)
            if st.button("CV Uygula"):
                model = st.session_state.best_model
                X = st.session_state.X_train
                y = st.session_state.y_train
                if st.session_state.problem_type == 'classification':
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=st.session_state.random_state)
                    scoring = 'accuracy'
                else:
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=st.session_state.random_state)
                    scoring = 'r2'
                scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
                st.write(f"Ortalama {scoring}: {scores.mean():.4f} (+/- {scores.std()*2:.4f})")
                st.session_state.cv_results = scores
        else:
            st.warning("Önce bir model eğitmelisiniz.")
        if st.button("İleri ➡️"):
            st.session_state.step = 11
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 9
            st.rerun()

    # Adım 11: Stacking & Ensemble
    elif st.session_state.step == 11:
        st.header("🧩 Stacking & Ensemble")
        if st.session_state.problem_type in ['classification', 'regression']:
            models_list = list(st.session_state.models.values())
            model_names = list(st.session_state.models.keys())
            selected_ensemble = st.multiselect("Ensemble'da kullanılacak modeller", model_names, default=model_names[:2])
            if selected_ensemble:
                estimators = [(name, st.session_state.models[name]) for name in selected_ensemble]
                if st.session_state.problem_type == 'classification':
                    stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
                else:
                    stacking = StackingRegressor(estimators=estimators, final_estimator=LinearRegression())
                if st.button("Stacking Modeli Eğit"):
                    stacking.fit(st.session_state.X_train, st.session_state.y_train)
                    st.session_state.stacking_model = stacking
                    if st.session_state.problem_type == 'classification':
                        metrics = evaluate_classification(stacking, st.session_state.X_test, st.session_state.y_test)
                    else:
                        metrics = evaluate_regression(stacking, st.session_state.X_test, st.session_state.y_test)
                    st.write("Stacking Model Performansı:")
                    st.json(metrics)
            else:
                st.info("En az iki model seçin.")
        else:
            st.info("Kümeleme veya öneri sistemi için ensemble bu sürümde desteklenmiyor.")
        if st.button("İleri ➡️"):
            st.session_state.step = 12
            st.rerun()
        if st.button("⬅️ Geri"):
            st.session_state.step = 10
            st.rerun()

    # Adım 12: Model Yorumlama
    elif st.session_state.step == 12:
        st.header("📖 Model Yorumlama")
        if st.session_state.best_model:
            # Feature importance (varsa)
            model = st.session_state.best_model
            # Pipeline içindeki modeli bul
            if hasattr(model, 'named_steps'):
                real_model = model.named_steps['model']
            else:
                real_model = model
            if hasattr(real_model, 'feature_importances_'):
                # Özellik isimlerini al
                if st.session_state.preprocessor:
                    # Preprocessor ile feature isimlerini almak karmaşık, basitçe sütun isimlerini göster
                    feature_names = st.session_state.X_train.columns.tolist()
                    importances = real_model.feature_importances_
                    # Sadece ilk n kadar
                    if len(importances) != len(feature_names):
                        st.warning("Özellik isimleri uyuşmuyor, sadece önem değerleri gösteriliyor.")
                        st.bar_chart(importances[:20])
                    else:
                        feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
                        st.bar_chart(feat_imp)
                else:
                    feature_names = st.session_state.X_train.columns
                    importances = real_model.feature_importances_
                    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False).head(20)
                    st.bar_chart(feat_imp)
            elif hasattr(real_model, 'coef_'):
                coefs = real_model.coef_[0] if real_model.coef_.ndim > 1 else real_model.coef_
                feature_names = st.session_state.X_train.columns
                coef_imp = pd.Series(coefs, index=feature_names).sort_values(key=abs, ascending=False).head(20)
                st.bar_chart(coef_imp)
            else:
                st.info("Bu model için özellik önemi çıkarılamadı.")
            
            # SHAP değerleri (opsiyonel)
            if st.checkbox("SHAP değerlerini göster (zaman alabilir)"):
                try:
                    import shap
                    explainer = shap.Explainer(real_model, st.session_state.X_train)
                    shap_values = explainer(st.session_state.X_test)
                    st.pyplot(shap.summary_plot(shap_values, st.session_state.X_test, show=False))
                except Exception as e:
                    st.error(f"SHAP hatası: {e}")
        else:
            st.warning("Model bulunamadı.")
        if st.button("🏁 Bitir"):
            st.balloons()
            st.success("Pipeline tamamlandı!")
        if st.button("⬅️ Geri"):
            st.session_state.step = 11
            st.rerun()

if __name__ == "__main__":
    main()
