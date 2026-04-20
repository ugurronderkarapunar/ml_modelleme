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

# ------------------------------
# SAYFA YAPILANDIRMASI
# ------------------------------
st.set_page_config(
    page_title="SmartBudget AI - Kişisel Finans Asistanınız",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# ÖZEL CSS (MODERN TASARIM)
# ------------------------------
st.markdown("""
<style>
    /* Ana arka plan */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Kartlar */
    .custom-card {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease;
    }
    .custom-card:hover {
        transform: translateY(-5px);
    }
    
    /* Başlık */
    .main-title {
        text-align: center;
        color: white;
        font-size: 3rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        margin-bottom: 0;
    }
    .sub-title {
        text-align: center;
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* Butonlar */
    .stButton>button {
        background: linear-gradient(90deg, #00c6ff, #0072ff);
        color: white;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Inputlar */
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select {
        border-radius: 30px;
        border: 1px solid #ddd;
        padding: 0.5rem 1rem;
    }
    
    /* Metrik kartları */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------
# SESSION STATE BAŞLAT
# ------------------------------
if 'expenses' not in st.session_state:
    st.session_state.expenses = pd.DataFrame(columns=["Tarih", "Kategori", "Tutar", "Not"])
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'view' not in st.session_state:
    st.session_state.view = "dashboard"

# ------------------------------
# SIDEBAR (GELİŞMİŞ)
# ------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag.png", width=80)
    st.markdown("## 💰 SmartBudget AI")
    st.markdown("---")
    
    # Dark mode toggle
    dark_mode = st.toggle("🌙 Dark Mode", st.session_state.dark_mode)
    if dark_mode != st.session_state.dark_mode:
        st.session_state.dark_mode = dark_mode
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 👤 Profil")
    monthly_income = st.number_input("💰 Aylık Gelir (₺)", min_value=0, value=15000, step=500)
    
    st.markdown("---")
    st.markdown("### 🎯 Hedefler (Zarf Sistemi)")
    categories = ["🏠 Kira/Fatura", "🍔 Yeme-İçme", "🚗 Ulaşım", "🛍️ Alışveriş", 
                  "🎉 Eğlence", "💊 Sağlık", "📚 Eğitim", "💸 Diğer"]
    
    budgets = {}
    for cat in categories:
        budgets[cat] = st.number_input(cat, min_value=0, value=2000, step=100, key=f"budget_{cat}")
    
    total_budget = sum(budgets.values())
    remaining = monthly_income - total_budget
    
    if total_budget > monthly_income:
        st.error(f"⚠️ Toplam bütçe ({total_budget}₺) geliri aşıyor! ({monthly_income}₺)")
    else:
        st.success(f"✅ Tasarruf: {remaining}₺")
    
    st.markdown("---")
    st.markdown("### 📱 Menü")
    view = st.radio("", ["🏠 Dashboard", "📊 Raporlar", "📝 Harcamalar", "🤖 AI Asistan"])
    st.session_state.view = view.split()[1].lower() if len(view.split())>1 else "dashboard"

# ------------------------------
# ANA EKRAN - DASHBOARD
# ------------------------------
st.markdown('<div class="main-title">💰 SmartBudget AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Yapay Zeka Destekli Kişisel Finans Asistanınız</div>', unsafe_allow_html=True)

# Harcama ekleme formu (card içinde)
with st.container():
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns([2,2,2,1])
    with col1:
        amount = st.number_input("💵 Tutar (₺)", min_value=0.0, step=10.0, key="amount")
    with col2:
        category = st.selectbox("📂 Kategori", categories, key="category")
    with col3:
        note = st.text_input("📝 Not", placeholder="Örn: Migros alışverişi", key="note")
    with col4:
        if st.button("➕ Ekle", use_container_width=True):
            if amount > 0:
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
                st.error("Tutar giriniz")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# DASHBOARD GÖRÜNÜMÜ
# ------------------------------
if st.session_state.view == "dashboard":
    if len(st.session_state.expenses) > 0:
        df = st.session_state.expenses.copy()
        df["Tarih"] = pd.to_datetime(df["Tarih"])
        current_month = datetime.now().month
        monthly = df[df["Tarih"].dt.month == current_month]
        total_spent = monthly["Tutar"].sum()
        
        # Metrik kartları (üst kısım)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{total_spent:,.0f} ₺</div>
                <div class="metric-label">Bu Ay Harcanan</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            remaining_budget = monthly_income - total_spent
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{remaining_budget:,.0f} ₺</div>
                <div class="metric-label">Kalan Bütçe</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            avg_daily = total_spent / max(1, datetime.now().day)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{avg_daily:.0f} ₺</div>
                <div class="metric-label">Günlük Ortalama</div>
            </div>
            """, unsafe_allow_html=True)
        with col4:
            days_left = (datetime(datetime.now().year, datetime.now().month+1, 1) - datetime.now()).days if datetime.now().month < 12 else 30
            days_left = max(1, days_left)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{days_left} gün</div>
                <div class="metric-label">Ay Sonuna Kalan</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Zarf sistemi ilerlemesi (progress bars)
        st.subheader("📊 Kategori Bazlı Bütçe Takibi")
        cols = st.columns(2)
        for i, cat in enumerate(categories):
            spent = monthly[monthly["Kategori"] == cat]["Tutar"].sum() if not monthly.empty else 0
            budget = budgets[cat]
            percent = min(100, int((spent / budget) * 100)) if budget > 0 else 0
            color = "#4CAF50" if percent < 80 else ("#FF9800" if percent < 100 else "#F44336")
            with cols[i % 2]:
                st.markdown(f"**{cat}**")
                st.progress(percent/100, text=f"{spent:.0f}₺ / {budget}₺ (%{percent})")
                if percent > 100:
                    st.warning(f"⚠️ Aşan: {spent - budget:.0f}₺")
        
        # Grafikler (Plotly interaktif)
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🍕 Kategori Dağılımı")
            if not monthly.empty:
                pie_data = monthly.groupby("Kategori")["Tutar"].sum().reset_index()
                fig = px.pie(pie_data, values="Tutar", names="Kategori", hole=0.4, color_discrete_sequence=px.colors.sequential.Blues_r)
                fig.update_layout(showlegend=True, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Bu ay henüz harcama yok")
        with col2:
            st.subheader("📈 Günlük Harcama Trendi")
            if len(df) >= 3:
                daily = df.groupby(df["Tarih"].dt.date)["Tutar"].sum().reset_index()
                fig = px.line(daily, x="Tarih", y="Tutar", markers=True, title="Günlük Harcama")
                fig.update_layout(margin=dict(t=30, b=0, l=0, r=0))
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Yeterli veri yok")
    else:
        st.info("👈 Henüz harcama eklenmedi. Yukarıdan ekleyerek başlayın.")

# ------------------------------
# RAPORLAR GÖRÜNÜMÜ
# ------------------------------
elif st.session_state.view == "raporlar":
    st.subheader("📊 Detaylı Raporlar")
    if len(st.session_state.expenses) > 0:
        df = st.session_state.expenses.copy()
        df["Tarih"] = pd.to_datetime(df["Tarih"])
        # Aylık filtre
        months = df["Tarih"].dt.strftime("%Y-%m").unique()
        selected_month = st.selectbox("Ay seç", months)
        filtered = df[df["Tarih"].dt.strftime("%Y-%m") == selected_month]
        
        st.dataframe(filtered, use_container_width=True)
        
        # Excel export
        if st.button("📥 Raporu İndir (Excel)"):
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                filtered.to_excel(writer, sheet_name="Harcamalar", index=False)
            st.download_button("İndir", data=output.getvalue(), file_name=f"rapor_{selected_month}.xlsx")
    else:
        st.info("Veri yok")

# ------------------------------
# HARCAMALAR GÖRÜNÜMÜ
# ------------------------------
elif st.session_state.view == "harcamalar":
    st.subheader("📝 Tüm Harcamalar")
    if len(st.session_state.expenses) > 0:
        st.dataframe(st.session_state.expenses.sort_values("Tarih", ascending=False), use_container_width=True)
        if st.button("🗑️ Tümünü Sil", type="secondary"):
            st.session_state.expenses = pd.DataFrame(columns=["Tarih", "Kategori", "Tutar", "Not"])
            st.rerun()
    else:
        st.info("Henüz harcama yok")

# ------------------------------
# AI ASİSTAN GÖRÜNÜMÜ
# ------------------------------
elif st.session_state.view == "asistan":
    st.subheader("🤖 AI Finans Asistanı")
    if len(st.session_state.expenses) > 0:
        df = st.session_state.expenses.copy()
        df["Tarih"] = pd.to_datetime(df["Tarih"])
        monthly = df[df["Tarih"].dt.month == datetime.now().month]
        total_spent = monthly["Tutar"].sum()
        
        # Anomaly detection
        Q1 = monthly["Tutar"].quantile(0.25)
        Q3 = monthly["Tutar"].quantile(0.75)
        IQR = Q3 - Q1
        anomalies = monthly[(monthly["Tutar"] > Q3 + 1.5*IQR) | (monthly["Tutar"] < Q1 - 1.5*IQR)]
        
        st.markdown("### 🔍 Anormal Harcamalar")
        if len(anomalies) > 0:
            st.warning(f"{len(anomalies)} anormal harcama tespit edildi:")
            st.dataframe(anomalies[["Tarih","Kategori","Tutar","Not"]])
        else:
            st.success("Anormal harcama yok")
        
        st.markdown("### 📈 Gelecek Tahmini")
        if len(df) >= 3:
            daily = df.groupby(df["Tarih"].dt.date)["Tutar"].sum().reset_index()
            daily["GünIndex"] = range(len(daily))
            model = LinearRegression()
            model.fit(daily[["GünIndex"]], daily["Tutar"])
            next_pred = model.predict([[len(daily)]])[0]
            st.metric("Yarınki Tahmini Harcama", f"{max(0,next_pred):.0f} ₺")
        
        st.markdown("### 💡 Tasarruf Önerisi")
        daily_avg = total_spent / max(1, datetime.now().day)
        days_left = (datetime(datetime.now().year, datetime.now().month+1, 1) - datetime.now()).days
        projected = total_spent + daily_avg * days_left
        if projected > monthly_income:
            st.error(f"⚠️ Mevcut hızla ay sonunda {projected - monthly_income:.0f}₺ açık vereceksiniz. Günlük harcamanızı {daily_avg:.0f}₺'den {max(0,(monthly_income - total_spent)/days_left):.0f}₺'ye düşürmelisiniz.")
        else:
            st.success(f"✅ Ay sonunda {monthly_income - projected:.0f}₺ tasarruf edebilirsiniz. Tebrikler!")
    else:
        st.info("Yeterli veri yok, önce harcama ekleyin.")
