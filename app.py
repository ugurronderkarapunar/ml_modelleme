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

# Temel bilimsel kütüphaneler
from scipy import stats
from scipy.spatial.distance import cosine, euclidean

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
from sklearn.metrics.pairwise import cosine_similarity

import xgboost as xgb

# Öneri sistemleri için
from sklearn.decomposition import TruncatedSVD

# ---------- OPSIYONEL KÜTÜPHANELER ----------
try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ------------------ KONFİGÜRASYON ------------------
st.set_page_config(page_title="PhD-Level Analytics & RecSys", layout="wide", page_icon="🧠")

# Session state
if 'df' not in st.session_state:
    st.session_state.update({
        'df': None,
        'model': None,
        'target': None,
        'task': 'Classification',
        'recommender': None,
        'rec_type': 'content',
        'content_cols': []
    })

# ------------------ İSTATİSTİKSEL ANALİZ MOTORU (TAM DÜZELTİLMİŞ) ------------------
class StatisticalEngine:
    @staticmethod
    def normality_test(data, col):
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
        groups = data[group_col].unique()
        if len(groups) != 2:
            return None
        g1 = data[data[group_col] == groups[0]][value_col].dropna()
        g2 = data[data[group_col] == groups[1]][value_col].dropna()
        t_stat, p_val = stats.ttest_ind(g1, g2)
        return {"t_statistic": t_stat, "p-value": p_val, "significant": p_val < 0.05}
    
    @staticmethod
    def anova(data, group_col, value_col):
        groups = [data[data[group_col] == g][value_col].dropna() for g in data[group_col].unique()]
        groups = [g for g in groups if len(g) > 1]
        if len(groups) < 2:
            return None
        f_stat, p_val = stats.f_oneway(*groups)
        return {"F_statistic": f_stat, "p-value": p_val, "significant": p_val < 0.05}
    
    @staticmethod
    def correlation_matrix(data, method='pearson'):
        num_data = data.select_dtypes(include=[np.number])
        return num_data.corr(method=method)
    
    @staticmethod
    def vif_analysis(X):
        if not STATSMODELS_AVAILABLE:
            return None
        X_const = sm.add_constant(X)
        vif_data = pd.DataFrame()
        vif_data["feature"] = X_const.columns
        vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        return vif_data
    
    @staticmethod
    def outlier_detection(data, method='iqr', contamination=0.1):
        # Sadece sayısal sütunlar
        num_data = data.select_dtypes(include=[np.number]).copy()
        if num_data.empty:
            return pd.Series([False] * len(data))
        # NaN'leri medyan ile doldur
        num_data = num_data.fillna(num_data.median())
        
        if method == 'iqr':
            outliers = pd.DataFrame(index=data.index)
            for col in num_data.columns:
                Q1 = num_data[col].quantile(0.25)
                Q3 = num_data[col].quantile(0.75)
                IQR = Q3 - Q1
                outliers[col] = (num_data[col] < (Q1 - 1.5 * IQR)) | (num_data[col] > (Q3 + 1.5 * IQR))
            return outliers.any(axis=1)
        elif method == 'zscore':
            from scipy.stats import zscore
            z_scores = np.abs(zscore(num_data))
            return (z_scores > 3).any(axis=1)
        elif method == 'isolation_forest':
            iso = IsolationForest(contamination=contamination, random_state=42)
            preds = iso.fit_predict(num_data)
            return preds == -1
        return pd.Series([False] * len(data))
    
    @staticmethod
    def pca_analysis(data, n_components=2, scale=True):
        # Sadece sayısal sütunlar
        num_data = data.select_dtypes(include=[np.number]).copy()
        if num_data.empty:
            return np.array([]), np.array([]), None
        # NaN'leri medyan ile doldur
        num_data = num_data.fillna(num_data.median())
        
        if scale:
            scaler = StandardScaler()
            num_scaled = scaler.fit_transform(num_data)
        else:
            num_scaled = num_data.values
        
        n_comp = min(n_components, num_scaled.shape[1])
        pca = PCA(n_components=n_comp)
        pcs = pca.fit_transform(num_scaled)
        explained_var = pca.explained_variance_ratio_
        return pcs, explained_var, pca

# ------------------ HİBRİT ÖNERİ SİSTEMİ ------------------
class HybridRecommender:
    def __init__(self, content_columns=None, user_col='user_id', item_col='item_id', rating_col='rating'):
        self.content_columns = content_columns if content_columns else []
        self.user_col = user_col
        self.item_col = item_col
        self.rating_col = rating_col
        self.content_sim_matrix = None
        self.user_factors = None
        self.item_factors = None
        self.user_encoder = None
        self.item_encoder = None
        self.is_fitted = False
    
    def fit_content(self, df_items):
        if len(self.content_columns) == 0:
            return
        features = df_items[self.content_columns].copy()
        cat_cols = features.select_dtypes(include=['object', 'category']).columns
        for col in cat_cols:
            features[col] = LabelEncoder().fit_transform(features[col].astype(str))
        features = features.fillna(features.median())
        scaler = StandardScaler()
        feature_matrix = scaler.fit_transform(features)
        self.content_sim_matrix = cosine_similarity(feature_matrix)
    
    def fit_collaborative(self, df_ratings, n_factors=20):
        if df_ratings.empty:
            return
        self.user_encoder = {uid: i for i, uid in enumerate(df_ratings[self.user_col].unique())}
        self.item_encoder = {iid: i for i, iid in enumerate(df_ratings[self.item_col].unique())}
        n_users = len(self.user_encoder)
        n_items = len(self.item_encoder)
        matrix = np.zeros((n_users, n_items))
        for _, row in df_ratings.iterrows():
            u = self.user_encoder[row[self.user_col]]
            i = self.item_encoder[row[self.item_col]]
            matrix[u, i] = row[self.rating_col]
        n_comp = min(n_factors, n_users-1, n_items-1)
        if n_comp < 1:
            return
        svd = TruncatedSVD(n_components=n_comp)
        self.user_factors = svd.fit_transform(matrix)
        self.item_factors = svd.components_.T
        self.is_fitted = True
    
    def recommend_collaborative(self, user_id, top_k=5):
        if not self.is_fitted or self.user_factors is None:
            return []
        u = self.user_encoder.get(user_id, None)
        if u is None:
            return []
        user_vec = self.user_factors[u]
        predictions = np.dot(self.item_factors, user_vec)
        top_items = np.argsort(predictions)[-top_k:][::-1]
        inv_item_encoder = {v: k for k, v in self.item_encoder.items()}
        return [(inv_item_encoder[i], predictions[i]) for i in top_items if i in inv_item_encoder]
    
    def hybrid_recommend(self, user_id, item_id=None, alpha=0.5, top_k=5):
        collab_recs = self.recommend_collaborative(user_id, top_k=top_k*2)
        content_recs = []
        if item_id:
            pass
        combined = {}
        for item, score in collab_recs:
            combined[item] = combined.get(item, 0) + alpha * score
        sorted_recs = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return sorted_recs

# ------------------ ANA UYGULAMA ------------------
def main():
    st.title("🧠 PhD-Level Analytics & Recommendation System")
    st.markdown("İstatistiksel testler, makine öğrenmesi, production-grade öneri sistemi")
    
    if not STATSMODELS_AVAILABLE:
        st.info("📌 Not: 'statsmodels' yüklü değil. Bazı analizler devre dışı.")
    if not MLFLOW_AVAILABLE:
        st.info("📌 Not: 'mlflow' yüklü değil. Model loglama devre dışı.")
    
    # Sidebar
    with st.sidebar:
        st.header("📂 Veri Yükleme")
        uploaded_file = st.file_uploader("CSV veya Excel", type=['csv', 'xlsx'])
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file, engine='openpyxl')
                # NaN'leri temizle (tüm sayısal sütunlarda medyan ile doldur)
                for col in df.columns:
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown", inplace=True)
                st.session_state.df = df
                st.success(f"✅ {st.session_state.df.shape[0]} satır, {st.session_state.df.shape[1]} sütun")
            except Exception as e:
                st.error(f"Hata: {e}")
        
        if st.session_state.df is not None:
            st.divider()
            st.session_state.target = st.selectbox("🎯 Hedef Değişken", st.session_state.df.columns)
            st.session_state.task = st.radio("Görev Tipi", ["Classification", "Regression"])
            
            st.divider()
            st.subheader("🎯 Öneri Sistemi")
            st.session_state.rec_type = st.selectbox("Öneri Tipi", ["İçerik Tabanlı", "İşbirlikçi", "Hibrit"])
            if st.session_state.rec_type != "İşbirlikçi":
                content_cols = st.multiselect("İçerik özellik sütunları", st.session_state.df.columns)
                st.session_state.content_cols = content_cols
    
    if st.session_state.df is None:
        st.info("Lütfen bir veri seti yükleyin.")
        return
    
    df = st.session_state.df
    tabs = st.tabs(["📊 İstatistiksel Analiz", "🤖 ML Model", "🎯 Öneri Sistemi", "🏭 Production"])
    
    # ---------- TAB 1 ----------
    with tabs[0]:
        st.subheader("🔬 İleri İstatistiksel Analiz")
        col_sel = st.selectbox("Analiz edilecek sütun", df.columns)
        
        with st.expander("📈 Normallik Testleri"):
            norm_res = StatisticalEngine.normality_test(df, col_sel)
            if norm_res:
                st.json(norm_res)
        
        if len(df.select_dtypes(include=['object']).columns) > 0:
            with st.expander("📊 Varyans Homojenliği"):
                group_col = st.selectbox("Grup sütunu", df.select_dtypes(include=['object']).columns)
                homo_res = StatisticalEngine.homogeneity_test(df, group_col, col_sel)
                if homo_res:
                    st.json(homo_res)
        
        with st.expander("📉 Korelasyon Matrisi"):
            corr_method = st.selectbox("Metod", ["pearson", "spearman", "kendall"])
            corr_matrix = StatisticalEngine.correlation_matrix(df, corr_method)
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        
        with st.expander("🔍 PCA"):
            n_comp = st.slider("Bileşen sayısı", 2, min(10, df.shape[1]-1), 2)
            pcs, var_exp, _ = StatisticalEngine.pca_analysis(df, n_components=n_comp)
            if len(pcs) > 0:
                st.write(f"Açıklanan varyans: {var_exp.cumsum()[-1]:.2%}")
                fig2, ax2 = plt.subplots()
                ax2.bar(range(1, len(var_exp)+1), var_exp, alpha=0.7)
                st.pyplot(fig2)
            else:
                st.info("PCA için yeterli sayısal sütun yok.")
        
        with st.expander("⚠️ Aykırı Değerler"):
            method = st.selectbox("Yöntem", ["iqr", "zscore", "isolation_forest"])
            outliers = StatisticalEngine.outlier_detection(df, method=method)
            st.write(f"Aykırı değer sayısı: {outliers.sum()} / {len(df)}")
            if outliers.sum() > 0:
                st.dataframe(df[outliers].head())
    
    # ---------- TAB 2 ----------
    with tabs[1]:
        st.header("🤖 AutoML Eğitimi")
        n_est = st.slider("N_estimators", 50, 500, 100)
        test_size = st.slider("Test oranı", 0.1, 0.4, 0.2)
        
        if st.button("🚀 Modeli Eğit"):
            X = df.drop(columns=[st.session_state.target])
            y = df[st.session_state.target]
            
            # Sayısal/kategorik ayrımı
            for col in X.columns:
                if X[col].dtype == 'object':
                    try:
                        X[col] = pd.to_numeric(X[col], errors='coerce')
                    except:
                        pass
            
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
            
            for col in X.columns:
                if col not in numeric_cols and col not in categorical_cols:
                    categorical_cols.append(col)
            
            transformers = []
            if numeric_cols:
                num_pipe = Pipeline([
                    ('impute', SimpleImputer(strategy='median')),
                    ('scale', RobustScaler())
                ])
                transformers.append(('num', num_pipe, numeric_cols))
            if categorical_cols:
                cat_pipe = Pipeline([
                    ('impute', SimpleImputer(strategy='most_frequent')),
                    ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
                ])
                transformers.append(('cat', cat_pipe, categorical_cols))
            
            if not transformers:
                st.error("Hiçbir sütun işlenemedi.")
                return
            
            preprocessor = ColumnTransformer(transformers)
            
            if st.session_state.task == "Classification":
                model = RandomForestClassifier(n_estimators=n_est, class_weight='balanced', random_state=42)
            else:
                model = xgb.XGBRegressor(n_estimators=n_est, learning_rate=0.05, random_state=42)
            
            full_pipe = Pipeline([('prep', preprocessor), ('model', model)])
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            with st.spinner("Eğitiliyor..."):
                full_pipe.fit(X_train, y_train)
                st.session_state.model = full_pipe
            
            y_pred = full_pipe.predict(X_test)
            if st.session_state.task == "Classification":
                st.text(classification_report(y_test, y_pred))
                fig, ax = plt.subplots()
                sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', ax=ax)
                st.pyplot(fig)
            else:
                r2 = r2_score(y_test, y_pred)
                mape = mean_absolute_percentage_error(y_test, y_pred)
                col1, col2 = st.columns(2)
                col1.metric("R²", f"{r2:.4f}")
                col2.metric("MAPE", f"{mape:.2%}")
            
            try:
                if hasattr(full_pipe.named_steps['model'], 'feature_importances_'):
                    importances = full_pipe.named_steps['model'].feature_importances_
                    if categorical_cols and 'cat' in full_pipe.named_steps['prep'].named_transformers_:
                        ohe = full_pipe.named_steps['prep'].named_transformers_['cat'].named_steps['encode']
                        ohe_cols = ohe.get_feature_names_out(categorical_cols)
                        all_features = numeric_cols + list(ohe_cols)
                    else:
                        all_features = numeric_cols
                    feat_imp = pd.Series(importances[:len(all_features)], index=all_features).sort_values(ascending=False).head(10)
                    st.subheader("🔑 En Önemli Değişkenler")
                    st.bar_chart(feat_imp)
            except Exception:
                pass
            
            model_bytes = pickle.dumps(full_pipe)
            st.download_button("💾 Modeli İndir (.pkl)", data=model_bytes, file_name="model.pkl")
    
    # ---------- TAB 3 ----------
    with tabs[2]:
        st.header("🎯 Hibrit Öneri Sistemi")
        user_col = st.selectbox("Kullanıcı ID sütunu", df.columns, key="user_col")
        item_col = st.selectbox("Ürün/Item ID sütunu", df.columns, key="item_col")
        
        if st.button("📌 Öneri Modelini Eğit"):
            recommender = HybridRecommender(
                content_columns=st.session_state.content_cols,
                user_col=user_col,
                item_col=item_col,
                rating_col=st.session_state.target
            )
            if st.session_state.rec_type != "İşbirlikçi" and st.session_state.content_cols:
                items_df = df.drop_duplicates(item_col)[[item_col] + st.session_state.content_cols]
                recommender.fit_content(items_df)
            if st.session_state.rec_type != "İçerik Tabanlı":
                ratings_df = df[[user_col, item_col, st.session_state.target]].dropna()
                if len(ratings_df) > 5:
                    recommender.fit_collaborative(ratings_df)
                else:
                    st.warning("İşbirlikçi için yeterli veri yok.")
            st.session_state.recommender = recommender
            st.success("Öneri sistemi eğitildi!")
        
        if st.session_state.recommender:
            user_id_input = st.text_input("Kullanıcı ID girin")
            if user_id_input:
                recs = st.session_state.recommender.hybrid_recommend(user_id_input, top_k=5)
                if recs:
                    for item, score in recs:
                        st.write(f"- {item} (skor: {score:.2f})")
                else:
                    st.info("Öneri yok.")
    
    # ---------- TAB 4 ----------
    with tabs[3]:
        st.header("🏭 Production Özellikleri")
        if st.session_state.model:
            st.subheader("Batch Inference")
            if st.button("Tahmin yap"):
                X = df.drop(columns=[st.session_state.target])
                preds = st.session_state.model.predict(X)
                result_df = pd.DataFrame({"Gerçek": df[st.session_state.target], "Tahmin": preds})
                st.dataframe(result_df.head(100))
                st.download_button("CSV indir", data=result_df.to_csv(index=False), file_name="predictions.csv")
            
            st.subheader("FastAPI Şablonu")
            st.code("""
from fastapi import FastAPI
import pandas as pd
import pickle

app = FastAPI()
model = pickle.load(open("model.pkl", "rb"))

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    pred = model.predict(df)
    return {"prediction": pred[0]}
""", language="python")
        else:
            st.info("Önce bir model eğitin.")

if __name__ == "__main__":
    main()
