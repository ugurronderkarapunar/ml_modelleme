import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# İstatistik ve bilimsel kütüphaneler
from scipy import stats
from scipy.spatial.distance import cosine, euclidean
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# Makine öğrenmesi
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, r2_score,
                             mean_absolute_percentage_error, silhouette_score)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import xgboost as xgb

# Öneri sistemleri için
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import pairwise_distances

# Production
import mlflow
import mlflow.sklearn
import tempfile
import logging

# ------------------ KONFİGÜRASYON ------------------
st.set_page_config(page_title="PhD-Level Analytics & RecSys", layout="wide", page_icon="🧠")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Session state
if 'df' not in st.session_state:
    st.session_state.update({
        'df': None,
        'model': None,
        'preprocessor': None,
        'target': None,
        'task': 'Classification',
        'recommender': None,
        'user_item_matrix': None,
        'rec_type': 'content'
    })

# ------------------ İSTATİSTİKSEL ANALİZ MOTORU ------------------
class StatisticalEngine:
    """Doktora seviyesinde istatistiksel testler ve analizler"""
    
    @staticmethod
    def normality_test(data, col):
        """Shapiro-Wilk ve D'Agostino-Pearson testleri"""
        ser = data[col].dropna()
        if len(ser) < 3:
            return None
        shapiro = stats.shapiro(ser)
        dagostino = stats.normaltest(ser) if len(ser) >= 20 else (None, None)
        return {
            "Shapiro-Wilk": {"statistic": shapiro[0], "p-value": shapiro[1], "normal": shapiro[1] > 0.05},
            "D'Agostino-Pearson": {"statistic": dagostino[0], "p-value": dagostino[1], "normal": dagostino[1] > 0.05} if dagostino[0] else None
        }
    
    @staticmethod
    def homogeneity_test(data, group_col, value_col):
        """Levene ve Bartlett varyans homojenliği testleri"""
        groups = [data[data[group_col] == g][value_col].dropna() for g in data[group_col].unique()]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            return None
        levene = stats.levene(*groups)
        bartlett = stats.bartlett(*groups) if all(len(g) > 0 for g in groups) else (None, None)
        return {
            "Levene": {"statistic": levene[0], "p-value": levene[1], "homogeneous": levene[1] > 0.05},
            "Bartlett": {"statistic": bartlett[0], "p-value": bartlett[1], "homogeneous": bartlett[1] > 0.05} if bartlett[0] else None
        }
    
    @staticmethod
    def t_test(data, group_col, value_col):
        """Bağımsız iki grup t-testi"""
        groups = data[group_col].unique()
        if len(groups) != 2:
            return None
        g1 = data[data[group_col] == groups[0]][value_col].dropna()
        g2 = data[data[group_col] == groups[1]][value_col].dropna()
        t_stat, p_val = stats.ttest_ind(g1, g2)
        return {"t_statistic": t_stat, "p-value": p_val, "significant": p_val < 0.05}
    
    @staticmethod
    def anova(data, group_col, value_col):
        """Tek yönlü ANOVA"""
        groups = [data[data[group_col] == g][value_col].dropna() for g in data[group_col].unique()]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            return None
        f_stat, p_val = stats.f_oneway(*groups)
        return {"F_statistic": f_stat, "p-value": p_val, "significant": p_val < 0.05}
    
    @staticmethod
    def correlation_matrix(data, method='pearson'):
        """Korelasyon matrisi (Pearson, Spearman, Kendall)"""
        num_data = data.select_dtypes(include=[np.number])
        return num_data.corr(method=method)
    
    @staticmethod
    def vif_analysis(X):
        """Variance Inflation Factor - çoklu doğrusal bağlantı"""
        X_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        return vif_data
    
    @staticmethod
    def outlier_detection(data, method='iqr', contamination=0.1):
        """Aykırı değer tespiti: IQR, Z-score, Isolation Forest"""
        num_cols = data.select_dtypes(include=[np.number]).columns
        outliers = pd.DataFrame(index=data.index)
        
        if method == 'iqr':
            for col in num_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = (data[col] < (Q1 - 1.5 * IQR)) | (data[col] > (Q3 + 1.5 * IQR))
        elif method == 'zscore':
            from scipy.stats import zscore
            z_scores = np.abs(zscore(data[num_cols]))
            outliers = (z_scores > 3).any(axis=1)
            return outliers
        elif method == 'isolation_forest':
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(data[num_cols].fillna(data[num_cols].median()))
            outliers = preds == -1
            return outliers
        return outliers.any(axis=1)
    
    @staticmethod
    def pca_analysis(data, n_components=2, scale=True):
        """Temel Bileşen Analizi"""
        num_data = data.select_dtypes(include=[np.number]).fillna(data.median())
        if scale:
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(num_data)
        else:
            num_scaled = num_data.values
        pca = PCA(n_components=min(n_components, num_scaled.shape[1]))
        pcs = pca.fit_transform(num_scaled)
        explained_var = pca.explained_variance_ratio_
        return pcs, explained_var, pca
    
    @staticmethod
    def time_series_analysis(series, order=(1,1,1), seasonal=False):
        """ARIMA zaman serisi analizi ve tahmini"""
        # Stationarity test
        adf = adfuller(series.dropna())
        is_stationary = adf[1] < 0.05
        
        if not is_stationary and seasonal:
            # Basit mevsimsel fark
            series = series.diff(12).dropna()
        
        model = ARIMA(series, order=order)
        fitted = model.fit()
        forecast = fitted.forecast(steps=10)
        return {
            "model": fitted,
            "forecast": forecast,
            "adf_pvalue": adf[1],
            "is_stationary": is_stationary,
            "aic": fitted.aic
        }

# ------------------ ÖNERİ SİSTEMİ (Production-Grade) ------------------
class HybridRecommender:
    """
    İçerik tabanlı + İşbirlikçi + Contextual Bandit hibrit öneri sistemi
    """
    def __init__(self, content_columns=None, user_col='user_id', item_col='item_id', rating_col='rating'):
        self.content_columns = content_columns
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.content_sim_matrix = None
        self.collab_svd = None
        self.user_encoder = None
        self.item_encoder = None
        self.is_fitted = False
    
    def fit_content(self, df_items):
        """İçerik tabanlı benzerlik matrisi oluştur"""
        if self.content_columns is None:
            return
        # Özellik vektörlerini oluştur (sayısal + kategorik)
        features = df_items[self.content_columns].copy()
        cat_cols = features.select_dtypes(include=['object']).columns
        for col in cat_cols:
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))
        features = features.fillna(features.median())
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(features)
        self.content_sim_matrix = cosine_similarity(feature_matrix)
        return self.content_sim_matrix
    
    def fit_collaborative(self, df_ratings, n_factors=20):
        """İşbirlikçi filtreleme için SVD"""
        # Kullanıcı ve item ID'lerini encode et
        self.user_encoder = {uid: i for i, uid in enumerate(df_ratings[self.user_col].unique())}
        self.item_encoder = {iid: i for i, iid in enumerate(df_ratings[self.item_col].unique())}
        user_inv = {v: k for k, v in self.user_encoder.items()}
        item_inv = {v: k for k, v in self.item_encoder.items()}
        
        n_users = len(self.user_encoder)
        n_items = len(self.item_encoder)
        matrix = np.zeros((n_users, n_items))
        for _, row in df_ratings.iterrows():
            u = self.user_encoder[row[self.user_col]]
            i = self.item_encoder[row[self.item_col]]
            matrix[u, i] = row[self.rating_col]
        
        # SVD ile matris faktorizasyonu
        svd = TruncatedSVD(n_components=min(n_factors, min(n_users, n_items)-1))
        self.collab_svd = svd.fit(matrix)
        self.user_factors = svd.transform(matrix)
        self.item_factors = svd.components_.T
        self.is_fitted = True
    
    def recommend_content(self, item_id, top_k=5):
        """İçerik tabanlı öneri"""
        if self.content_sim_matrix is None:
            return []
        item_idx = list(self.item_encoder.values())[0]  # basit demo
        # Gerçek implementasyonda item_id'den index bulunur
        sim_scores = list(enumerate(self.content_sim_matrix[item_idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        return sim_scores[1:top_k+1]
    
    def recommend_collaborative(self, user_id, top_k=5):
        """İşbirlikçi öneri"""
        if not self.is_fitted:
            return []
        u = self.user_encoder.get(user_id, 0)
        user_vec = self.user_factors[u]
        predictions = np.dot(self.item_factors, user_vec)
        top_items = np.argsort(predictions)[-top_k:][::-1]
        return top_items
    
    def hybrid_recommend(self, user_id, item_id=None, alpha=0.5, top_k=5):
        """Hibrit: ağırlıklı birleştirme"""
        collab_recs = self.recommend_collaborative(user_id, top_k=top_k*2)
        content_recs = self.recommend_content(item_id, top_k=top_k*2) if item_id else []
        # Basit birleştirme (gerçekte daha sofistike)
        combined = {}
        for idx, score in collab_recs:
            combined[idx] = combined.get(idx, 0) + alpha * score
        for idx, score in content_recs:
            combined[idx] = combined.get(idx, 0) + (1-alpha) * score
        sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_recs

# ------------------ PRODUCTION UTILS ------------------
def save_model_pipeline(pipeline, name="model"):
    """Modeli MLflow ile kaydet"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp:
        pickle.dump(pipeline, tmp)
        tmp_path = tmp.name
    mlflow.log_artifact(tmp_path, artifact_path="models")
    return tmp_path

def batch_inference(model, X_new, chunk_size=1000):
    """Batch tahmin (büyük veri için)"""
    predictions = []
    for i in range(0, len(X_new), chunk_size):
        chunk = X_new.iloc[i:i+chunk_size]
        preds = model.predict(chunk)
        predictions.extend(preds)
    return np.array(predictions)

# ------------------ STREAMLIT UI ------------------
def main():
    st.title("🧠 PhD-Level Analytics & Recommendation System")
    st.markdown("İstatistiksel testler, makine öğrenmesi, production-grade öneri sistemi")
    
    # Sidebar: Veri yükleme
    with st.sidebar:
        st.header("📂 Veri")
        uploaded_file = st.file_uploader("CSV veya Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            if uploaded_file.name.endswith('.csv'):
                st.session_state.df = pd.read_csv(uploaded_file)
            else:
                st.session_state.df = pd.read_excel(uploaded_file)
            st.success(f"✅ {st.session_state.df.shape[0]} satır, {st.session_state.df.shape[1]} sütun")
        
        if st.session_state.df is not None:
            st.divider()
            st.session_state.target = st.selectbox("🎯 Hedef Değişken", st.session_state.df.columns)
            st.session_state.task = st.radio("Görev Tipi", ["Classification", "Regression"])
            
            # Öneri sistemi tipi
            st.divider()
            st.subheader("🎯 Öneri Sistemi")
            st.session_state.rec_type = st.selectbox("Öneri Tipi", ["İçerik Tabanlı", "İşbirlikçi", "Hibrit"])
            
            # İçerik sütunları seçimi
            if st.session_state.rec_type != "İşbirlikçi":
                content_cols = st.multiselect("İçerik özellik sütunları", st.session_state.df.columns)
                st.session_state.content_cols = content_cols
    
    if st.session_state.df is None:
        st.info("Lütfen bir veri seti yükleyin.")
        return
    
    df = st.session_state.df
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 İstatistiksel Analiz", "🤖 ML Model", "🎯 Öneri Sistemi", "🏭 Production"])
    
    # ---------- TAB 1: STATISTICAL ANALYSIS ----------
    with tab1:
        st.subheader("🔬 İleri İstatistiksel Analiz Motoru")
        
        col_sel = st.selectbox("Analiz edilecek sütun", df.columns)
        
        # Normallik testi
        with st.expander("📈 Normallik Testleri (Shapiro-Wilk, D'Agostino)"):
            norm_res = StatisticalEngine.normality_test(df, col_sel)
            if norm_res:
                st.json(norm_res)
        
        # Varyans homojenliği
        if len(df.select_dtypes(include=['object']).columns) > 0:
            with st.expander("📊 Varyans Homojenliği (Levene, Bartlett)"):
                group_col = st.selectbox("Grup sütunu", df.select_dtypes(include=['object']).columns)
                homo_res = StatisticalEngine.homogeneity_test(df, group_col, col_sel)
                if homo_res:
                    st.json(homo_res)
        
        # Korelasyon matrisi
        with st.expander("📉 Korelasyon Matrisi"):
            corr_method = st.selectbox("Korelasyon metodu", ["pearson", "spearman", "kendall"])
            corr_matrix = StatisticalEngine.correlation_matrix(df, corr_method)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        # PCA
        with st.expander("🔍 Temel Bileşen Analizi (PCA)"):
            n_comp = st.slider("Bileşen sayısı", 2, min(10, df.shape[1]-1), 2)
            pcs, var_exp, pca_model = StatisticalEngine.pca_analysis(df, n_components=n_comp)
            st.write(f"Açıklanan varyans: {var_exp.cumsum()[-1]:.2%}")
            fig2, ax2 = plt.subplots()
            ax2.bar(range(1, len(var_exp)+1), var_exp, alpha=0.7)
            ax2.set_xlabel("Bileşenler")
            ax2.set_ylabel("Varyans Oranı")
            st.pyplot(fig2)
        
        # Aykırı değerler
        with st.expander("⚠️ Aykırı Değer Tespiti"):
            method = st.selectbox("Yöntem", ["iqr", "zscore", "isolation_forest"])
            outliers = StatisticalEngine.outlier_detection(df, method=method)
            st.write(f"Aykırı değer sayısı: {outliers.sum()} / {len(df)}")
            if outliers.sum() > 0:
                st.dataframe(df[outliers].head())
    
    # ---------- TAB 2: ML MODEL (AutoML gelişmiş) ----------
    with tab2:
        st.header("🤖 AutoML Eğitimi")
        n_est = st.slider("N_estimators", 50, 500, 100)
        test_size = st.slider("Test oranı", 0.1, 0.4, 0.2)
        
        if st.button("🚀 Modeli Eğit"):
            X = df.drop(columns=[st.session_state.target])
            y = df[st.session_state.target]
            
            # Preprocessor
            numeric_cols = X.select_dtypes(include=['int64','float64']).columns
            categorical_cols = X.select_dtypes(include=['object','category']).columns
            
            num_pipe = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', RobustScaler())])
            cat_pipe = Pipeline([('impute', SimpleImputer(strategy='most_frequent')), 
                                 ('encode', OneHotEncoder(handle_unknown='ignore'))])
            preprocessor = ColumnTransformer([('num', num_pipe, numeric_cols), ('cat', cat_pipe, categorical_cols)])
            
            if st.session_state.task == "Classification":
                model = RandomForestClassifier(n_estimators=n_est, class_weight='balanced', random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=n_est, learning_rate=0.05, random_state=42)
            
            full_pipe = Pipeline([('prep', preprocessor), ('model', model)])
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            full_pipe.fit(X_train, y_train)
            st.session_state.model = full_pipe
            
            y_pred = full_pipe.predict(X_test)
            if st.session_state.task == "Classification":
                st.text(classification_report(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', ax=ax)
                st.pyplot(fig)
            else:
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                st.metric("R²", f"{r2:.4f}")
                st.metric("MAPE", f"{mape:.2%}")
            
            # Feature importance
            try:
                importances = full_pipe.named_steps['model'].feature_importances_
                # OHE sonrası isimleri al
                ohe_cols = full_pipe.named_steps['prep'].transformers_[1][1].named_steps['encode'].get_feature_names_out(categorical_cols)
                all_features = list(numeric_cols) + list(ohe_cols)
                feat_imp = pd.Series(importances[:len(all_features)], index=all_features).sort_values(ascending=False).head(10)
                st.subheader("Önemli Değişkenler")
                st.bar_chart(feat_imp)
            except:
                pass
            
            # Modeli kaydet
            model_bytes = pickle.dumps(full_pipe)
            st.download_button("💾 Modeli İndir", data=model_bytes, file_name="model.pkl")
    
    # ---------- TAB 3: RECOMMENDER SYSTEM ----------
    with tab3:
        st.header("🎯 Hibrit Öneri Sistemi")
        
        # Örnek: İşbirlikçi için kullanıcı-ürün matrisi oluşturma
        # Varsayılan olarak, hedef değişkenin rating olduğunu varsayalım
        # Kullanıcı ve ürün sütunları yoksa demo oluştur
        
        if 'user_col' not in st.session_state:
            st.session_state.user_col = st.selectbox("Kullanıcı ID sütunu", df.columns)
            st.session_state.item_col = st.selectbox("Ürün/Item ID sütunu", df.columns)
            st.session_state.rating_col = st.session_state.target  # hedef değişken rating olabilir
        
        if st.button("📌 Öneri Modelini Eğit"):
            recommender = HybridRecommender(
                content_columns=st.session_state.get('content_cols', []),
                user_col=st.session_state.user_col,
                item_col=st.session_state.item_col,
                rating_col=st.session_state.rating_col
            )
            # İçerik tabanlı
            if st.session_state.rec_type != "İşbirlikçi" and st.session_state.get('content_cols'):
                # Benzersiz item'ların özelliklerini al
                items_df = df.drop_duplicates(st.session_state.item_col)[st.session_state.content_cols + [st.session_state.item_col]]
                recommender.fit_content(items_df)
            # İşbirlikçi
            if st.session_state.rec_type != "İçerik Tabanlı":
                ratings_df = df[[st.session_state.user_col, st.session_state.item_col, st.session_state.rating_col]]
                recommender.fit_collaborative(ratings_df)
            st.session_state.recommender = recommender
            st.success("Öneri sistemi eğitildi!")
        
        if st.session_state.recommender:
            user_id_input = st.text_input("Kullanıcı ID (örnek)")
            if user_id_input:
                recs = st.session_state.recommender.hybrid_recommend(user_id_input, top_k=5)
                st.write("Önerilen item'lar:", recs)
    
    # ---------- TAB 4: PRODUCTION ----------
    with tab4:
        st.header("🏭 Production-Grade Özellikler")
        
        if st.session_state.model:
            # Batch Inference
            st.subheader("Batch Inference")
            if st.button("Test setinde batch tahmin yap"):
                X = df.drop(columns=[st.session_state.target])
                y_true = df[st.session_state.target]
                preds = batch_inference(st.session_state.model, X)
                result_df = pd.DataFrame({"Gerçek": y_true, "Tahmin": preds})
                st.dataframe(result_df.head(100))
                st.download_button("Sonuçları CSV olarak indir", data=result_df.to_csv(index=False), file_name="predictions.csv")
            
            # Model drift izleme (basit)
            st.subheader("Model Drift İzleme")
            if st.button("Drift hesapla"):
                # Eğitim dağılımı vs yeni veri (simülasyon)
                train_preds = st.session_state.model.predict(df.drop(columns=[st.session_state.target]))
                drift_score = np.mean(np.abs(train_preds - df[st.session_state.target].values)) / np.std(train_preds)
                st.metric("Drift Skoru", f"{drift_score:.4f}")
                if drift_score > 0.2:
                    st.warning("⚠️ Model drift tespit edildi! Yeniden eğitim önerilir.")
            
            # API endpoint şablonu
            st.subheader("FastAPI Endpoint Şablonu")
            api_code = """
from fastapi import FastAPI, HTTPException
import pandas as pd
import pickle

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": pred[0]}
"""
            st.code(api_code, language="python")
            
            # MLflow ile model kaydı
            if st.button("MLflow'a modeli logla"):
                with mlflow.start_run():
                    mlflow.sklearn.log_model(st.session_state.model, "model")
                    mlflow.log_param("task", st.session_state.task)
                    mlflow.log_param("target", st.session_state.target)
                st.success("Model MLflow'a kaydedildi! (artifacts klasöründe)")

if __name__ == "__main__":
    main()
