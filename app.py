import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import io
import warnings
from scipy import stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import RobustScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier, GradientBoostingRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier, XGBRegressor
import shap

warnings.filterwarnings('ignore')
st.set_page_config(page_title="AutoML Studio", layout="wide")

# ------------------ 1. Yardımcı fonksiyonlar ------------------
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

# ------------------ 2. Session state ------------------
if 'df' not in st.session_state:
    st.session_state.df = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = None

st.title("🧠 Senior Full Stack Data Science Studio")
st.markdown("**Adım Adım: Veri Yükleme → EDA → Problem Tanımı → Ön İşleme → Modelleme → Ensemble → Yorumlama**")

# ------------------ 3. VERİ YÜKLEME ------------------
st.header("1️⃣ Veri Yükleme")
uploaded_file = st.file_uploader("CSV veya Excel dosyası seçin", type=['csv', 'xlsx'])
if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.session_state.df = df
        st.success(f"✅ {df.shape[0]} satır, {df.shape[1]} sütun")
        st.subheader("İlk 10 Gözlem")
        st.dataframe(df.head(10))
        st.subheader("Veri Bilgileri")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        st.subheader("Betimsel İstatistikler")
        st.dataframe(df.describe(include='all'))
    except Exception as e:
        st.error(f"Hata: {e}")

if st.session_state.df is None:
    st.stop()
df = st.session_state.df

# ------------------ 4. GENEL EDA ------------------
st.header("2️⃣ Genel Keşifsel Veri Analizi (EDA)")
col_sel = st.selectbox("Değişken seçin", df.columns)
if df[col_sel].dtype in ['int64', 'float64']:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(df[col_sel], kde=True, ax=axes[0])
    axes[0].set_title(f"{col_sel} - Histogram")
    sns.boxplot(y=df[col_sel], ax=axes[1])
    axes[1].set_title(f"{col_sel} - Boxplot")
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

# ------------------ 5. PROBLEM TANIMI ------------------
st.header("3️⃣ Problem Tanımı")
problem_type = st.radio("Problem tipi", ["Sınıflandırma", "Regresyon", "Kümeleme", "Öneri Sistemi"])
target = st.selectbox("Hedef Değişken (Y)", df.columns)
drop_cols = st.multiselect("Çıkarılacak gereksiz değişkenler", [c for c in df.columns if c != target])
X = df.drop(columns=[target] + drop_cols)
y = df[target]
st.session_state.target = target
st.session_state.problem_type = problem_type
st.session_state.X_columns = X.columns.tolist()

# ------------------ 6. VERİ ÖN İŞLEME ------------------
st.header("4️⃣ Veri Ön İşleme")
# Tip dönüşümü
for col in X.columns:
    if X[col].dtype == 'object':
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except:
            pass
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
st.write(f"**Sayısal ({len(numeric_cols)}):** {numeric_cols}")
st.write(f"**Kategorik ({len(categorical_cols)}):** {categorical_cols}")

# Eksik veri analizi
missing = X.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    st.dataframe(missing)
    num_strategy = st.selectbox("Sayısal eksik stratejisi", ["median", "mean", "zero"])
    cat_strategy = st.selectbox("Kategorik eksik stratejisi", ["most_frequent", "new_category"])
else:
    num_strategy, cat_strategy = "median", "most_frequent"
    st.success("Eksik veri yok!")

# Aykırı değer temizleme
if st.checkbox("Aykırı değer temizle (IQR)"):
    original_len = len(X)
    outlier_mask = pd.Series([False]*len(X), index=X.index)
    for col in numeric_cols:
        outlier_mask |= detect_outliers_iqr(X, col)
    X = X[~outlier_mask]
    y = y[~outlier_mask]
    st.success(f"Aykırı değerler temizlendi: {original_len} → {len(X)}")

# Eksik değer doldurma
X = handle_missing_values(X, num_strategy, cat_strategy)
st.success("Eksik değerler dolduruldu.")

# Feature engineering
if st.checkbox("Polinomik özellikler ekle (derece 2)"):
    if numeric_cols:
        poly = PolynomialFeatures(degree=2, include_bias=False)
        num_feat = poly.fit_transform(X[numeric_cols])
        new_cols = poly.get_feature_names_out(numeric_cols)
        X = pd.concat([X, pd.DataFrame(num_feat, columns=new_cols)], axis=1)
        st.success(f"Yeni özellik sayısı: {X.shape[1]}")
        numeric_cols = list(new_cols)  # güncelle

# ------------------ 7. TRAIN-TEST SPLIT ------------------
st.header("5️⃣ Train-Test Split")
test_size = st.slider("Test oranı (%)", 10, 40, 20) / 100
cv_folds = st.slider("Cross-validation kat sayısı", 3, 10, 5)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
st.write(f"**Eğitim:** {X_train.shape[0]} | **Test:** {X_test.shape[0]}")

# ------------------ 8. MODEL PIPELINE & KARŞILAŞTIRMA ------------------
st.header("6️⃣ Model Pipeline & Karşılaştırma")
num_transformer = Pipeline([('scaler', RobustScaler())])
cat_transformer = Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([
    ('num', num_transformer, numeric_cols),
    ('cat', cat_transformer, categorical_cols)
])
st.session_state.preprocessor = preprocessor

models = {}
if problem_type == "Sınıflandırma":
    models = {
        "Lojistik Regresyon": LogisticRegression(max_iter=1000),
        "Karar Ağacı": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(eval_metric='logloss'),
        "Gradient Boosting": GradientBoostingClassifier()
    }
    scoring = 'accuracy'
elif problem_type == "Regresyon":
    models = {
        "Ridge Regresyon": Ridge(),
        "Karar Ağacı": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor(),
        "XGBoost": XGBRegressor(),
        "Gradient Boosting": GradientBoostingRegressor()
    }
    scoring = 'r2'

if models and st.button("Model Karşılaştırmasını Başlat"):
    results = []
    prog = st.progress(0)
    for i, (name, model) in enumerate(models.items()):
        pipe = Pipeline([('prep', preprocessor), ('model', model)])
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=cv_folds, scoring=scoring)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        test_score = accuracy_score(y_test, y_pred) if problem_type=="Sınıflandırma" else r2_score(y_test, y_pred)
        results.append({"Model": name, f"CV {scoring}": round(cv_scores.mean(),4), f"Test {scoring}": round(test_score,4)})
        prog.progress((i+1)/len(models))
    st.dataframe(pd.DataFrame(results))

# ------------------ 9. HİPERPARAMETRE TUNING ------------------
st.header("7️⃣ Hiperparametre Tuning")
if models:
    selected_model = st.selectbox("Tuning yapılacak model", list(models.keys()))
    if st.button("GridSearch Başlat"):
        base_model = models[selected_model]
        param_grid = {}
        if "Random Forest" in selected_model:
            param_grid = {'model__n_estimators': [50,100], 'model__max_depth': [5,10,None]}
        elif "XGBoost" in selected_model:
            param_grid = {'model__n_estimators': [50,100], 'model__learning_rate': [0.01,0.1]}
        elif "Gradient Boosting" in selected_model:
            param_grid = {'model__n_estimators': [50,100], 'model__learning_rate': [0.01,0.1]}
        if param_grid:
            pipe = Pipeline([('prep', preprocessor), ('model', base_model)])
            grid = GridSearchCV(pipe, param_grid, cv=cv_folds, scoring=scoring, n_jobs=-1)
            with st.spinner("GridSearch çalışıyor..."):
                grid.fit(X_train, y_train)
            st.success(f"En iyi parametreler: {grid.best_params_}")
            st.metric(f"En iyi {scoring}", f"{grid.best_score_:.4f}")
            st.session_state.best_model = grid.best_estimator_
        else:
            st.info("Bu model için ön tanımlı parametre yok.")

# ------------------ 10. STACKING ENSEMBLE ------------------
st.header("8️⃣ Stacking Ensemble")
if problem_type in ["Sınıflandırma", "Regresyon"] and st.button("Stacking Modeli Oluştur"):
    if problem_type == "Sınıflandırma":
        estimators = [('rf', RandomForestClassifier(n_estimators=50)),
                      ('xgb', XGBClassifier(n_estimators=50, eval_metric='logloss'))]
        stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    else:
        estimators = [('rf', RandomForestRegressor(n_estimators=50)),
                      ('xgb', XGBRegressor(n_estimators=50))]
        stack = StackingRegressor(estimators=estimators, final_estimator=Ridge())
    pipe_stack = Pipeline([('prep', preprocessor), ('model', stack)])
    pipe_stack.fit(X_train, y_train)
    y_pred = pipe_stack.predict(X_test)
    score = accuracy_score(y_test, y_pred) if problem_type=="Sınıflandırma" else r2_score(y_test, y_pred)
    st.metric("Stacking Performansı", f"{score:.4f}")
    if st.session_state.best_model is None:
        st.session_state.best_model = pipe_stack

# ------------------ 11. MODEL YORUMLAMA (SHAP) ------------------
st.header("9️⃣ Model Yorumlama (SHAP)")
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
        st.info(f"SHAP çalıştırılamadı: {e}")
else:
    st.info("Önce bir model eğitin (GridSearch veya Stacking).")

# ------------------ 12. YENİ VERİ İLE TAHMİN ------------------
st.header("🔟 Yeni Veri ile Tahmin")
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
        infer_file = st.file_uploader("CSV dosyası yükleyin", type=['csv'])
        if infer_file and st.button("Toplu Tahmin Yap"):
            infer_df = pd.read_csv(infer_file)
            infer_df = handle_missing_values(infer_df, num_strategy, cat_strategy)
            preds = st.session_state.best_model.predict(infer_df)
            result_df = infer_df.copy()
            result_df['Tahmin'] = preds
            st.dataframe(result_df.head(100))
            st.download_button("CSV indir", data=result_df.to_csv(index=False), file_name="predictions.csv")
else:
    st.info("Lütfen önce bir model eğitin (GridSearch veya Stacking).")
