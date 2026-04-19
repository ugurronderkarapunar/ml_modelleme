import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report, mean_absolute_percentage_error, r2_score, confusion_matrix
import xgboost as xgb

# ------------------ CORE ANALYTICS FUNCTIONS ------------------

def get_statistical_summary(df, col):
    """Gelişmiş deskriptif istatistik ve dağılım analizi."""
    ser = df[col].dropna()
    if ser.empty: return {}
    
    k2, p = stats.normaltest(ser) if len(ser) >= 8 else (0, 0)
    
    return {
        "Skewness": round(ser.skew(), 3),
        "Kurtosis": round(ser.kurtosis(), 3),
        "Normal Dağılım mı?": "Evet" if p > 0.05 else "Hayır",
        "Ortalama": round(ser.mean(), 2),
        "Medyan": round(ser.median(), 2),
        "Standart Sapma": round(ser.std(), 2)
    }

def build_advanced_pipeline(X, num_strategy='median', scale_type='Robust'):
    """
    Dinamik Feature Engineering Hattı.
    HATA ÇÖZÜMÜ: handle_unknown='ignore' ile eğitimde olmayan 
    kategorik verilerin tahmini sırasında çökme engellenir.
    """
    
    # Sütunları içeriklerine göre ayır
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Sayısal Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=num_strategy)),
        ('scaler', RobustScaler() if scale_type == 'Robust' else StandardScaler())
    ])

    # Kategorik Pipeline
    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Birleştirici
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, numeric_features),
            ('cat', cat_transformer, categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor

# ------------------ STREAMLIT UI CONFIG ------------------

st.set_page_config(page_title="Pro-Level AutoML", layout="wide", page_icon="🧪")

# Session state başlatma
if 'df' not in st.session_state:
    st.session_state.update({
        'df': None, 
        'model': None, 
        'X_cols': None,
        'target': None,
        'task': "Classification"
    })

st.title("🧪 Senior Data Science Studio")
st.markdown("Veri analizinden model yayınına (deployment) kadar tam döngü.")

# ------------------ SIDEBAR ------------------

with st.sidebar:
    st.header("📂 Veri Yükleme")
    uploaded_file = st.file_uploader("Dosya Seçin", type=['csv', 'xlsx'])
    
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success("Veri Yüklendi!")
        except Exception as e:
            st.error(f"Hata: {e}")

    if st.session_state.df is not None:
        st.divider()
        st.session_state.target = st.selectbox("🎯 Hedef Değişken (Y)", st.session_state.df.columns)
        st.session_state.task = st.radio("Görev Tipi", ["Classification", "Regression"])

# ------------------ MAIN APP TABS ------------------

if st.session_state.df is not None:
    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["📊 Keşifsel Analiz (EDA)", "🤖 Eğitim & XAI", "🔮 Tahmin (Inference)"])

    # --- TAB 1: EDA ---
    with tab1:
        st.subheader("Veri Profilleme")
        col_sel = st.selectbox("İncelemek istediğiniz sütun:", df.columns)
        c1, c2 = st.columns([1, 2])
        
        if df[col_sel].dtype in ['int64', 'float64']:
            stats_res = get_statistical_summary(df, col_sel)
            c1.json(stats_res)
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df[col_sel], kde=True, color="#4A90E2", ax=ax)
            plt.title(f"{col_sel} Dağılım Grafiği")
            c2.pyplot(fig)
        else:
            c1.write("### Kategorik Frekanslar")
            c1.dataframe(df[col_sel].value_counts())
            fig, ax = plt.subplots()
            df[col_sel].value_counts().head(10).plot(kind='pie', autopct='%1.1f%%', ax=ax)
            c2.pyplot(fig)

    # --- TAB 2: TRAINING & INTERPRETABILITY ---
    with tab2:
        st.header("Model Pipeline Konfigürasyonu")
        cols = st.columns(3)
        n_est = cols[0].slider("Ağaç Sayısı (n_estimators)", 50, 500, 100)
        test_size = cols[1].slider("Test Veri Oranı (%)", 10, 40, 20) / 100
        
        if st.button("🚀 Pipeline'ı Eğit"):
            X = df.drop(columns=[st.session_state.target])
            y = df[st.session_state.target]
            st.session_state.X_cols = X.columns.tolist()
            
            # Pipeline İnşası
            prepro = build_advanced_pipeline(X)
            
            if st.session_state.task == "Classification":
                model_obj = RandomForestClassifier(n_estimators=n_est, class_weight='balanced', random_state=42)
            else:
                model_obj = xgb.XGBRegressor(n_estimators=n_est, learning_rate=0.05, random_state=42)

            full_pipe = Pipeline([('preprocessor', prepro), ('model', model_obj)])
            
            # Train-Test Split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            with st.spinner("Model optimize ediliyor..."):
                full_pipe.fit(X_train, y_train)
                st.session_state.model = full_pipe
            
            st.success("Eğitim Tamamlandı!")
            
            # Metrikler
            st.divider()
            y_pred = full_pipe.predict(X_test)
            m1, m2 = st.columns(2)
            
            if st.session_state.task == "Classification":
                m1.text("Sınıflandırma Raporu")
                m1.code(classification_report(y_test, y_pred))
                
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', ax=ax)
                m2.write("Hata Matrisi (Confusion Matrix)")
                m2.pyplot(fig)
            else:
                m1.metric("Açıklanan Varyans (R²)", f"{r2_score(y_test, y_pred):.4f}")
                m2.metric("Ortalama Hata (MAPE)", f"{mean_absolute_percentage_error(y_test, y_pred):.2%}")

            # Feature Importance
            st.subheader("💡 Model Hangi Değişkenlere Güveniyor?")
            try:
                # OHE sonrası oluşan sütun isimlerini yakala
                ohe_cols = full_pipe.named_steps['preprocessor'].transformers_[1][1].named_steps['encoder'].get_feature_names_out()
                num_cols = full_pipe.named_steps['preprocessor'].transformers_[0][2]
                all_features = list(num_cols) + list(ohe_cols)
                
                importances = full_pipe.named_steps['model'].feature_importances_
                feat_df = pd.Series(importances, index=all_features).sort_values(ascending=False).head(10)
                
                fig, ax = plt.subplots()
                feat_df.plot(kind='barh', ax=ax, color="#2ECC71")
                st.pyplot(fig)
            except:
                st.info("Öznitelik önemi bu model için çıkarılamadı.")

            # Download Model
            buffer = io.BytesIO()
            pickle.dump(full_pipe, buffer)
            st.download_button("💾 Modeli Dışa Aktar (.pkl)", data=buffer.getvalue(), file_name="auto_model.pkl")

    # --- TAB 3: PRO-LEVEL INFERENCE ---
    with tab3:
        if st.session_state.model:
            st.header("🔮 Akıllı Tahmin Arayüzü")
            X_orig = df.drop(columns=[st.session_state.target])
            
            with st.form("inference_form"):
                inputs = {}
                cols = st.columns(3)
                for i, c_name in enumerate(X_orig.columns):
                    with cols[i % 3]:
                        if X_orig[c_name].dtype in ['int64', 'float64']:
                            inputs[c_name] = st.number_input(c_name, value=float(X_orig[c_name].median()))
                        else:
                            inputs[c_name] = st.selectbox(c_name, X_orig[c_name].unique())
                
                submit = st.form_submit_button("Tahmin Et")
            
            if submit:
                input_df = pd.DataFrame([inputs])
                # Numeric zorlaması (Tahmin hatalarını engellemek için)
                for col in X_orig.select_dtypes(include=[np.number]).columns:
                    input_df[col] = pd.to_numeric(input_df[col])
                
                prediction = st.session_state.model.predict(input_df)
                
                st.balloons()
                res_c1, res_c2 = st.columns(2)
                res_c1.metric("Tahmin Sonucu", f"{prediction[0]:.2f}" if st.session_state.task == "Regression" else str(prediction[0]))
                
                if st.session_state.task == "Classification":
                    probs = st.session_state.model.predict_proba(input_df)
                    res_c2.write("Sınıf Olasılıkları")
                    res_c2.bar_chart(probs[0])
        else:
            st.warning("Eğitilmiş bir model bulunamadı. Lütfen 'Model Training' sekmesine gidin.")
else:
    st.info("Başlamak için yan menüden veri seti yükleyiniz.")
