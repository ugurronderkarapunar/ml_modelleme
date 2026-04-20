import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(page_title="Smart Budget AI", layout="wide", page_icon="💰")
# Gelişmiş CSS - app.py'nin en başına (st.set_page_config'ten sonra)
st.markdown("""
<style>
    /* Ana arka plan */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    /* Ana container */
    .main .block-container {
        background: rgba(255,255,255,0.95);
        border-radius: 30px;
        padding: 2rem;
        margin-top: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
    /* Kartlar */
    .custom-card {
        background: white;
        border-radius: 20px;
        padding: 1.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.05);
        transition: transform 0.3s ease;
        margin-bottom: 1rem;
    }
    .custom-card:hover {
        transform: translateY(-5px);
    }
    /* Metrik kartları */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 20px;
        padding: 1rem;
        text-align: center;
    }
    /* Butonlar */
    .stButton > button {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 10px 20px rgba(0,0,0,0.2);
    }
    /* Progress bar özelleştirme */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #FFC107, #F44336);
        border-radius: 20px;
    }
    /* Başlık */
    h1, h2, h3 {
        font-family: 'Segoe UI', sans-serif;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Sidebar */
    .css-1d391kg {
        background: rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)
# Başlık
st.title("💰 Smart Budget AI")
st.markdown("**Proaktif • AI Destekli • Minimalist**")
st.markdown("---")

# Sidebar - Kullanıcı bilgileri
with st.sidebar:
    st.header("👤 Profil")
    monthly_income = st.number_input("Aylık Gelir (₺)", min_value=0, value=15000, step=1000)
    st.markdown("---")
    st.header("🎯 Hedefler (Zarf Sistemi)")
    st.caption("Her kategoriye ayırmak istediğiniz bütçeyi girin")
    
    # Varsayılan kategoriler
    categories = ["🏠 Kira/Fatura", "🍔 Yeme-İçme", "🚗 Ulaşım", "🛍️ Alışveriş", "🎉 Eğlence", "💊 Sağlık", "📚 Eğitim", "💸 Diğer"]
    
    budgets = {}
    for cat in categories:
        budgets[cat] = st.number_input(cat, min_value=0, value=2000, step=100, key=f"budget_{cat}")
    
    total_budget = sum(budgets.values())
    if total_budget > monthly_income:
        st.error(f"⚠️ Toplam bütçe ({total_budget}₺) geliri aşıyor! ({monthly_income}₺)")
    else:
        st.success(f"✅ Kalan: {monthly_income - total_budget}₺ (Tasarruf/Yatırım)")

# Ana ekran - Harcama girişi ve analiz
st.header("📝 Bugünkü Harcamalarınız")

# Harcama giriş formu
col1, col2, col3 = st.columns(3)
with col1:
    amount = st.number_input("Tutar (₺)", min_value=0.0, step=10.0, key="amount")
with col2:
    category = st.selectbox("Kategori", categories, key="category")
with col3:
    note = st.text_input("Not (opsiyonel)", placeholder="Örn: Market alışverişi", key="note")

if st.button("➕ Harcama Ekle", type="primary"):
    if amount > 0:
        # Session state'te harcamaları tut
        if 'expenses' not in st.session_state:
            st.session_state.expenses = pd.DataFrame(columns=["Tarih", "Kategori", "Tutar", "Not"])
        
        new_expense = pd.DataFrame({
            "Tarih": [datetime.now().strftime("%Y-%m-%d %H:%M")],
            "Kategori": [category],
            "Tutar": [amount],
            "Not": [note]
        })
        st.session_state.expenses = pd.concat([st.session_state.expenses, new_expense], ignore_index=True)
        st.success("✅ Harcama eklendi!")
        st.rerun()
    else:
        st.error("Lütfen geçerli bir tutar girin.")

# Harcamaları göster
if 'expenses' in st.session_state and len(st.session_state.expenses) > 0:
    st.markdown("---")
    st.subheader("📋 Son Harcamalar")
    
    # Harcama tablosu
    st.dataframe(st.session_state.expenses.sort_values("Tarih", ascending=False), use_container_width=True)
    
    if st.button("🗑️ Tüm Harcamaları Temizle"):
        st.session_state.expenses = pd.DataFrame(columns=["Tarih", "Kategori", "Tutar", "Not"])
        st.rerun()
    
    # ========== 1. ZARF SİSTEMİ (Gerçek Zamanlı Takip) ==========
    st.markdown("---")
    st.header("📊 Zarf Sistemi - Gerçek Zamanlı Bütçe Takibi")
    
    # Bu ayın harcamaları
    current_month = datetime.now().month
    current_year = datetime.now().year
    df_expenses = st.session_state.expenses
    df_expenses["Tarih"] = pd.to_datetime(df_expenses["Tarih"])
    monthly_expenses = df_expenses[df_expenses["Tarih"].dt.month == current_month]
    
    # Kategori bazlı harcama toplamları
    category_spending = monthly_expenses.groupby("Kategori")["Tutar"].sum().to_dict()
    
    # Bütçe ilerlemesini göster
    cols = st.columns(2)
    col_idx = 0
    for cat in categories:
        spent = category_spending.get(cat, 0)
        budget = budgets[cat]
        percent = min(100, int((spent / budget) * 100)) if budget > 0 else 0
        color = "green" if percent < 80 else ("orange" if percent < 100 else "red")
        
        with cols[col_idx % 2]:
            st.markdown(f"**{cat}**")
            st.progress(percent/100, text=f"🔄 {spent}₺ / {budget}₺ (%{percent})")
            if percent > 100:
                st.warning(f"⚠️ Bütçe aşıldı! Aşan: {spent - budget}₺")
            elif percent > 80:
                st.info(f"⚠️ Bütçenin %{percent}'ü kullanıldı, dikkat!")
        col_idx += 1
    
    # ========== 2. AI TAHMİNİ VE ANORMAL HARCAMA TESPİTİ ==========
    st.markdown("---")
    st.header("🤖 AI Analizleri")
    
    # 2a. Anormal harcama tespiti (IQR yöntemi)
    st.subheader("🔍 Anormal Harcama Tespiti")
    if len(monthly_expenses) > 0:
        Q1 = monthly_expenses["Tutar"].quantile(0.25)
        Q3 = monthly_expenses["Tutar"].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        anomalies = monthly_expenses[(monthly_expenses["Tutar"] < lower) | (monthly_expenses["Tutar"] > upper)]
        
        if len(anomalies) > 0:
            st.warning(f"⚠️ {len(anomalies)} anormal harcama tespit edildi:")
            st.dataframe(anomalies[["Tarih", "Kategori", "Tutar", "Not"]], use_container_width=True)
        else:
            st.success("✅ Anormal harcama yok.")
    else:
        st.info("Henüz harcama yok.")
    
    # 2b. Gelecek harcama tahmini (basit linear regression)
    st.subheader("📈 Yarın İçin Harcama Tahmini")
    if len(df_expenses) >= 3:
        # Günlük toplam harcamalar
        df_expenses["Gün"] = df_expenses["Tarih"].dt.date
        daily_totals = df_expenses.groupby("Gün")["Tutar"].sum().reset_index()
        daily_totals = daily_totals.sort_values("Gün")
        daily_totals["GünIndex"] = range(len(daily_totals))
        
        # Model eğitimi
        X = daily_totals["GünIndex"].values.reshape(-1, 1)
        y = daily_totals["Tutar"].values
        model = LinearRegression()
        model.fit(X, y)
        
        # Yarın tahmini
        next_day_index = len(daily_totals)
        predicted = model.predict([[next_day_index]])[0]
        predicted = max(0, predicted)  # Negatif olmasın
        
        st.metric("📅 Tahmini Yarınki Harcama", f"{predicted:.0f} ₺")
        
        # Grafik
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(daily_totals["Gün"], daily_totals["Tutar"], 'o-', label="Gerçekleşen")
        ax.axhline(y=predicted, color='r', linestyle='--', label=f"Tahmini: {predicted:.0f}₺")
        ax.set_xlabel("Gün")
        ax.set_ylabel("Toplam Harcama (₺)")
        ax.set_title("Günlük Harcama Trendi")
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
    else:
        st.info("Tahmin için en az 3 günlük veri gerekli.")
    
    # 2c. AI Bütçe Önerisi
    st.subheader("💡 AI Bütçe Önerisi")
    if len(monthly_expenses) > 0:
        total_spent = monthly_expenses["Tutar"].sum()
        remaining_budget = monthly_income - total_budget
        days_left = (datetime(current_year, current_month+1, 1) - datetime.now()).days if current_month < 12 else 30
        days_left = max(1, days_left)
        daily_avg = total_spent / (datetime.now().day)
        projected = total_spent + (daily_avg * days_left)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("🔄 Bugüne Kadar Harcanan", f"{total_spent:.0f}₺")
        col2.metric("📅 Kalan Gün", days_left)
        col3.metric("📊 Projeksiyon (Ay Sonu)", f"{projected:.0f}₺")
        
        if projected > monthly_income:
            st.error(f"⚠️ Mevcut hızla ay sonunda {projected - monthly_income:.0f}₺ açık vereceksiniz! Günlük harcamanızı {daily_avg:.0f}₺'den {max(0, (monthly_income - total_spent)/days_left):.0f}₺'ye düşürmelisiniz.")
        else:
            st.success(f"✅ Mevcut hızla ay sonunda {monthly_income - projected:.0f}₺ tasarruf edebilirsiniz.")
    else:
        st.info("Henüz bu ay harcama yok.")
    
    # 2d. AI Hedef Yönlendirme (Fazlayı hedefe ekle)
    if monthly_income - total_budget > 0:
        st.subheader("🎯 AI Önerisi: Fazla Bütçenizi Hedefleyin")
        leftover = monthly_income - total_budget
        st.info(f"📌 Aylık gelirinizden zarf bütçelerini ayırdıktan sonra **{leftover}₺** fazlanız var. Bunu birikim, borç kapatma veya yatırım hedefine yönlendirmenizi öneririm.")
    else:
        st.warning("⚠️ Bütçeler gelirinizi aşıyor. Lütfen bütçeleri kısın.")
    
else:
    st.info("👈 Henüz harcama eklenmedi. Yukarıdan harcama ekleyerek başlayın.")

# Footer
st.markdown("---")
st.caption("💰 Smart Budget AI | Proaktif, AI Destekli, Minimalist")
