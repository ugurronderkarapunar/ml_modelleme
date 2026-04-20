import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
from scipy import stats

st.set_page_config(page_title="Veri Yükleme & EDA", layout="wide")
st.title("📊 Veri Yükleme ve Keşifsel Veri Analizi (EDA)")

# Session state
if 'df' not in st.session_state:
    st.session_state.df = None

# ------------------ 1. VERİ YÜKLEME ------------------
st.header("1️⃣ Veri Yükleme")
uploaded_file = st.file_uploader("CSV veya Excel dosyası seçin", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
        st.session_state.df = df
        st.success(f"✅ Dosya başarıyla yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
        
        # İlk 10 gözlem
        st.subheader("İlk 10 Gözlem")
        st.dataframe(df.head(10))
        
        # Veri bilgileri (info)
        st.subheader("Veri Seti Bilgileri")
        buffer = io.StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        # Betimsel istatistikler
        st.subheader("Betimsel İstatistikler")
        st.dataframe(df.describe(include='all'))
        
    except Exception as e:
        st.error(f"Hata: {e}")

# ------------------ 2. GENEL EDA ------------------
if st.session_state.df is not None:
    st.header("2️⃣ Genel Keşifsel Veri Analizi (EDA)")
    df = st.session_state.df
    
    # Değişken seçimi
    col_sel = st.selectbox("Analiz edilecek değişkeni seçin", df.columns)
    
    # Değişken tipine göre görselleştirme
    if df[col_sel].dtype in ['int64', 'float64']:
        # Sayısal değişken: histogram + boxplot
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col_sel], kde=True, ax=axes[0])
        axes[0].set_title(f"{col_sel} - Histogram")
        sns.boxplot(y=df[col_sel], ax=axes[1])
        axes[1].set_title(f"{col_sel} - Boxplot")
        st.pyplot(fig)
        
        # Normallik testi
        if len(df[col_sel].dropna()) >= 3:
            stat, p = stats.shapiro(df[col_sel].dropna())
            st.write(f"**Shapiro-Wilk normallik testi:** p-value = {p:.4f} → {'Normal dağılım gösteriyor' if p > 0.05 else 'Normal dağılım göstermiyor'}")
        
        # Aykırı değer özeti (IQR yöntemi)
        Q1 = df[col_sel].quantile(0.25)
        Q3 = df[col_sel].quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((df[col_sel] < (Q1 - 1.5*IQR)) | (df[col_sel] > (Q3 + 1.5*IQR))).sum()
        st.write(f"**Aykırı değer sayısı (IQR):** {outliers}")
        
    else:
        # Kategorik değişken: frekans tablosu ve barchart
        st.write("**Frekans tablosu (ilk 10 kategori):**")
        freq = df[col_sel].value_counts().head(10)
        st.dataframe(freq)
        st.bar_chart(freq)
    
    # Tüm sayısal değişkenler için korelasyon matrisi
    numeric_df = df.select_dtypes(include=[np.number])
    if not numeric_df.empty:
        st.subheader("Korelasyon Matrisi (Sayısal Değişkenler)")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
        st.pyplot(fig)
    else:
        st.info("Sayısal değişken olmadığı için korelasyon matrisi gösterilemiyor.")
else:
    st.info("Lütfen önce bir veri dosyası yükleyin.")
