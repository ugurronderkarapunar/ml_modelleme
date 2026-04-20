import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import warnings
from datetime import datetime
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, classification_report, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
import shap

warnings.filterwarnings('ignore')

st.set_page_config(page_title="AutoML Studio", layout="wide")
st.title("🧠 Senior Full Stack Data Science Studio")
st.markdown("**Veri Yükleme → EDA → Problem Tanımı → Feature Engineering → Modelleme → Ensemble → Yorumlama**")

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None
    st.session_state.target = None
    st.session_state.problem_type = None
    st.session_state.best_model = None
    st.session_state.preprocessor = None
    st.session_state.X_columns = None

# ------------------ YARDIMCI FONKSİYONLAR ------------------
def handle_missing_values(df, num_strategy='median', cat_strategy='most_frequent'):
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

def safe_boxplot(data, x_col, y_col, ax):
    plot_data = data[[x_col, y_col]].dropna()
    if plot_data.empty or plot_data[x_col].nunique() == 0:
        ax.text(0.5, 0.5, "Yeterli veri yok", transform=ax.transAxes, ha='center')
        return
    grouped = plot_data.groupby(x_col)[y_col].count()
    valid_groups = grouped[grouped > 1].index.tolist()
    if len(valid_groups) == 0:
        ax.text(0.5, 0.5, "Her grupta en az 2 gözlem gerekli", transform=ax.transAxes, ha='center')
        return
    plot_data = plot_data[plot_data[x_col].isin(valid_groups)]
    sns.boxplot(data=plot_data, x=x_col, y=y_col, ax=ax)

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
        if len(df[col_sel].dropna()) >= 3:
            stat, p = stats.shapiro(df[col_sel].dropna())
            st.write(f"**Shapiro-Wilk p-value:** {p:.4f} → {'Normal' if p>0.05 else 'Normal değil'}")
    else:
        st.bar_chart(df[col_sel].value_counts().head(10))
    num_df = df.select_dtypes(include=[np.number])
    if not num_df.empty:
        st.subheader("Korelasyon Matrisi")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)

# ------------------ TAB 3: PROBLEM TANIMI ------------------
with st.expander("🎯 Problem Tanımı"):
    st.session_state.problem_type = st.radio("Problem tipi:", ["Sınıflandırma", "Regresyon", "Kümeleme", "Öneri Sistemi"])
    st.session_state.target = st.selectbox("Hedef Değişken (Y)", df.columns)
    drop_cols = st.multiselect("Çıkarılacak değişkenler", [c for c in df.columns if c != st.session_state.target])
    X = df.drop(columns=[st.session_state.target] + drop_cols)
    y = df[st.session_state.target]
    st.session_state.X_columns = X.columns.tolist()

# ------------------ TAB 4: VERİ ÖN İŞLEME ------------------
with st.expander("🛠️ Veri Ön İşleme"):
    st.subheader("Tip Dönüşümleri")
    for col in X.columns:
        if X[col].dtype == 'object':
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
                st.info(f"{col} sayısala çevrildi")
            except:
                pass
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    st.write(f"**Sayısal ({len(numeric_cols)}):** {numeric_cols}")
    st.write(f"**Kategorik ({len(categorical_cols)}):** {categorical_cols}")
    
    st.subheader("Eksik Veri Analizi")
    missing = X.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        st.dataframe(missing)
        num_strategy = st.selectbox("Sayısal eksik stratejisi", ["median", "mean", "zero"])
        cat_strategy = st.selectbox("Kategorik eksik stratejisi", ["most_frequent", "new_category"])
    else:
        num_strategy, cat_strategy = "median", "most_frequent"
        st.success("Eksik veri yok!")
    
    st.subheader("Aykırı Değer Analizi")
    outlier_method = st.selectbox("Aykırı değer temizleme", ["none", "iqr"])
    if outlier_method == "iqr":
        original_len = len(X)
        outlier_mask = pd.Series([False] * len(X), index=X.index)
        for col in numeric_cols:
            outlier_mask |= detect_outliers_iqr(X, col)
        X = X[~outlier_mask]
        y = y[~outlier_mask]
        st.success(f"Aykırı değerler temizlendi: {original_len} → {len(X)} satır")
    
    X = handle_missing_values(X, num_strategy, cat_strategy)
    st.success("Eksik değerler dolduruldu.")
    
    st.subheader("Korelasyon")
    if numeric_cols:
        st.dataframe(X[numeric_cols].corr())
    
    st.subheader("EDA Görselleştirme")
    if categorical_cols and numeric_cols:
        for cat in categorical_cols[:2]:
            if X[cat].nunique() <= 10:
                fig, ax = plt.subplots(figsize=(10, 5))
                safe_boxplot(X, cat, numeric_cols[0], ax)
                st.pyplot(fig)
    
    st.subheader("Feature Engineering")
    if st.checkbox("Polinomik özellikler ekle (derece 2)"):
        poly = PolynomialFeatures(degree=2, include_bias=False)
        num_feat = poly.fit_transform(X[numeric_cols])
        new_cols = poly.get_feature_names_out(numeric_cols)
        X = pd.concat([X, pd.DataFrame(num_feat, columns=new_cols)], axis=1)
        st.success(f"Yeni özellik sayısı: {X.shape[1]}")

# ------------------ TAB 5: TRAIN-TEST SPLIT ------------------
with st.expander("🔀 Train-Test Split"):
    test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100
    cv_folds = st.slider("Cross-validation kat sayısı", 3, 10, 5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    st.write(f"**Eğitim:** {X_train.shape[0]} | **Test:** {X_test.shape[0]}")

# ------------------ TAB 6: MODEL PIPELINE & KARŞILAŞTIRMA ------------------
with st.expander("⚙️ Model Pipeline & Karşılaştırma"):
    num_transformer = Pipeline([('scaler', RobustScaler())])
    cat_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer([
        ('num', num_transformer, numeric_cols),
        ('cat', cat_transformer, categorical_cols)
    ])
    st.session_state.preprocessor = preprocessor
    
    models = {}
    scoring_metric = ''
    if st.session_state.problem_type == "Sınıflandırma":
        models = {
            "Lojistik Regresyon": LogisticRegression(max_iter=1000, random_state=42),
            "Karar Ağacı": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }
        scoring_metric = 'accuracy'
    elif st.session_state.problem_type == "Regresyon":
        models = {
            "Ridge Regresyon": Ridge(random_state=42),
            "Karar Ağacı": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42)
        }
        scoring_metric = 'r2'
    
    if models and st.button("🚀 Model Karşılaştır"):
        results = []
        prog = st.progress(0)
        for i, (name, model) in enumerate(models.items()):
            pipe = Pipeline([('prep', preprocessor), ('model', model)])
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring_metric)
            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)
            test_score = accuracy_score(y_test, y_pred) if st.session_state.problem_type == "Sınıflandırma" else r2_score(y_test, y_pred)
            results.append({"Model": name, f"CV {scoring_metric}": round(cv_scores.mean(), 4), f"Test {scoring_metric}": round(test_score, 4)})
            prog.progress((i+1)/len(models))
        st.dataframe(pd.DataFrame(results))

# ------------------ TAB 7: HİPERPARAMETRE TUNING ------------------
with st.expander("🎛️ Hiperparametre Tuning"):
    if models:
        selected_model = st.selectbox("Tuning yapılacak model", list(models.keys()))
        if st.button("🔍 GridSearch Başlat"):
            base_model = models[selected_model]
            param_grid = {}
            if "Random Forest" in selected_model:
                param_grid = {'model__n_estimators': [50, 100], 'model__max_depth': [5, 10, None]}
            elif "XGBoost" in selected_model:
                param_grid = {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            elif "Gradient Boosting" in selected_model:
                param_grid = {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1]}
            if param_grid:
                pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
                grid = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring=scoring_metric, n_jobs=-1)
                with st.spinner("GridSearch çalışıyor..."):
                    grid.fit(X_train, y_train)
                st.success(f"En iyi parametreler: {grid.best_params_}")
                st.metric(f"En iyi {scoring_metric}", f"{grid.best_score_:.4f}")
                st.session_state.best_model = grid.best_estimator_
            else:
                st.info("Bu model için parametre aralığı tanımlanmamış.")
    else:
        st.info("Önce model karşılaştırmasını çalıştırın.")

# ------------------ TAB 8: STACKING & ENSEMBLE ------------------
with st.expander("🧬 Stacking & Ensemble"):
    if st.session_state.problem_type in ["Sınıflandırma", "Regresyon"]:
        if st.button("🧩 Stacking Modeli Oluştur"):
            if st.session_state.problem_type == "Sınıflandırma":
                estimators = [('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                              ('xgb', XGBClassifier(n_estimators=50, eval_metric='logloss', random_state=42))]
                stack_model = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
            else:
                estimators = [('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                              ('xgb', XGBRegressor(n_estimators=50, random_state=42))]
                stack_model = StackingRegressor(estimators=estimators, final_estimator=Ridge())
            pipe_stack = Pipeline([('prep', preprocessor), ('model', stack_model)])
            pipe_stack.fit(X_train, y_train)
            y_pred = pipe_stack.predict(X_test)
            score = accuracy_score(y_test, y_pred) if st.session_state.problem_type == "Sınıflandırma" else r2_score(y_test, y_pred)
            st.metric("Stacking Performansı", f"{score:.4f}")
            if st.session_state.best_model is None:
                st.session_state.best_model = pipe_stack
    else:
        st.info("Stacking sadece sınıflandırma/regresyon için.")

# ------------------ TAB 9: MODEL YORUMLAMA (SHAP) ------------------
with st.expander("📖 Model Yorumlama (SHAP)"):
    if st.session_state.best_model is not None:
        try:
            pipe = st.session_state.best_model
            X_transformed = pipe.named_steps['prep'].transform(X_train)
            model = pipe.named_steps['model']
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed[:100])
            if hasattr(pipe.named_steps['prep'], 'get_feature_names_out'):
                feature_names = pipe.named_steps['prep'].get_feature_names_out()
            else:
                feature_names = [f"f{i}" for i in range(X_transformed.shape[1])]
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_transformed[:100], feature_names=feature_names, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.info(f"SHAP hatası: {e}")
    else:
        st.info("Önce bir model eğitin (GridSearch veya Stacking).")

# ------------------ TAB 10: TAHMIN (INFERENCE) ------------------
with st.expander("🔮 Yeni Veri ile Tahmin"):
    if st.session_state.best_model is not None:
        option = st.radio("Giriş tipi", ["Manuel", "CSV yükle"])
        if option == "Manuel":
            input_data = {}
            for col in st.session_state.X_columns:
                input_data[col] = st.text_input(col, "0")
            if st.button("Tahmin Et"):
                input_df = pd.DataFrame([input_data])
                input_df = handle_missing_values(input_df, num_strategy, cat_strategy)
                for c in input_df.columns:
                    try:
                        input_df[c] = pd.to_numeric(input_df[c])
                    except:
                        pass
                pred = st.session_state.best_model.predict(input_df)
                st.success(f"Tahmin: {pred[0]}")
        else:
            infer_file = st.file_uploader("CSV dosyası", type=['csv'])
            if infer_file and st.button("Toplu Tahmin"):
                infer_df = pd.read_csv(infer_file)
                infer_df = handle_missing_values(infer_df, num_strategy, cat_strategy)
                preds = st.session_state.best_model.predict(infer_df)
                result_df = infer_df.copy()
                result_df['Tahmin'] = preds
                st.dataframe(result_df.head(100))
                st.download_button("CSV indir", data=result_df.to_csv(index=False), file_name="predictions.csv")
    else:
        st.info("Lütfen önce bir model eğitin.")

st.markdown("---")
st.caption("Senior Full Stack Data Science Studio | AutoML, EDA, Feature Engineering, Model Karşılaştırma, Tuning, Ensemble, SHAP, Inference")
