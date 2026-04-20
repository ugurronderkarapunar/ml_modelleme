import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(page_title="Smart Budget AI", layout="wide", page_icon="💰")

# Özel CSS - Minimalist ve modern
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 8px 24px;
        font-weight: bold;
    }
    .stButton > button:hover { background-color: #45a049; }
    div[data-testid="stMetric"] { background-color: white; padding: 15px; border-radius: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Session state
if 'transactions' not in st.session_state:
    # Örnek veri
    dates = [datetime.now() - timedelta(days=i) for i in range(30, 0, -1)]
    categories = ['Yemek', 'Alışveriş', 'Fatura', 'Eğlence', 'Ulaşım', 'Sağlık']
    st.session_state.transactions = pd.DataFrame({
        'Tarih': dates,
        'Kategori': [random.choice(categories) for _ in range(30)],
        'Tutar': [random.randint(10, 200) for _ in range(30)],
        'Açıklama': ['Harcama']*30
    })
    # Aylık bütçe zarfı (her kategori için)
    st.session_state.budget_envelopes = {cat: 500 for cat in categories}
    st.session_state.monthly_income = 3000
    st.session_state.goals = []  # [{'hedef': 'Tatil', 'hedef_tutar': 2000, 'biriken': 0}]

st.title("💰 Smart Budget AI")
st.caption("Proaktif, AI destekli, sade ve etkili bütçe yönetimi")

# Sidebar - Minimal
with st.sidebar:
    st.header("⚡ Hızlı İşlemler")
    st.markdown("---")
    # Hızlı harcama ekleme
    with st.form("quick_expense"):
        st.subheader("➕ Hızlı Harcama")
        cat = st.selectbox("Kategori", list(st.session_state.budget_envelopes.keys()))
        amount = st.number_input("Tutar (₺)", min_value=1, step=10)
        submitted = st.form_submit_button("Ekle")
        if submitted:
            new_row = pd.DataFrame({
                'Tarih': [datetime.now()],
                'Kategori': [cat],
                'Tutar': [amount],
                'Açıklama': ['Hızlı giriş']
            })
            st.session_state.transactions = pd.concat([st.session_state.transactions, new_row], ignore_index=True)
            st.success("Harcama eklendi!")
            st.rerun()
    st.markdown("---")
    # Bütçe düzenleme
    st.subheader("📌 Zarf Bütçeleri")
    for cat in st.session_state.budget_envelopes.keys():
        new_budget = st.number_input(f"{cat}", value=int(st.session_state.budget_envelopes[cat]), key=f"budget_{cat}")
        st.session_state.budget_envelopes[cat] = new_budget
    st.session_state.monthly_income = st.number_input("Aylık Gelir", value=int(st.session_state.monthly_income))

# Ana Ekran - Üç sütun: KPI'lar, Hedefler, Uyarılar
col1, col2, col3 = st.columns(3)

# 1. KPI'lar (Gerçek zamanlı takip)
current_month = datetime.now().month
current_year = datetime.now().year
monthly_expenses = st.session_state.transactions[
    (pd.to_datetime(st.session_state.transactions['Tarih']).dt.month == current_month) &
    (pd.to_datetime(st.session_state.transactions['Tarih']).dt.year == current_year)
]['Tutar'].sum()
remaining = st.session_state.monthly_income - monthly_expenses

with col1:
    st.metric("💰 Bu Ay Harcama", f"{monthly_expenses:.0f} ₺", delta=f"Kalan {remaining:.0f} ₺")
    st.metric("📊 Zarf Kullanımı", f"{len([c for c in st.session_state.budget_envelopes if monthly_category_spending(c) > st.session_state.budget_envelopes[c]])} kategori aştı")

# Helper function
def monthly_category_spending(category):
    return st.session_state.transactions[
        (pd.to_datetime(st.session_state.transactions['Tarih']).dt.month == current_month) &
        (pd.to_datetime(st.session_state.transactions['Tarih']).dt.year == current_year) &
        (st.session_state.transactions['Kategori'] == category)
    ]['Tutar'].sum()

# 2. Hedef Yönetimi (Zarf sistemi ile esnek)
with col2:
    st.subheader("🎯 Akıllı Hedefler")
    # Hedef ekleme
    with st.expander("➕ Yeni Hedef", expanded=False):
        goal_name = st.text_input("Hedef adı (Örn: Yeni Telefon)")
        goal_amount = st.number_input("Hedef tutar", min_value=100, step=100)
        if st.button("Hedef Oluştur"):
            st.session_state.goals.append({'hedef': goal_name, 'hedef_tutar': goal_amount, 'biriken': 0})
            st.rerun()
    # Hedef listesi ve ilerleme
    for idx, goal in enumerate(st.session_state.goals):
        # Öneri: Aylık harcama fazlasını hedefe yönlendir (AI)
        surplus = max(0, st.session_state.monthly_income - monthly_expenses)
        suggestion = f"Bu ay {surplus} ₺ fazlanız var, hedefe ekleyebilirsiniz."
        col_a, col_b = st.columns([3,1])
        with col_a:
            st.write(f"**{goal['hedef']}** - {goal['biriken']}/{goal['hedef_tutar']} ₺")
            progress = goal['biriken'] / goal['hedef_tutar'] if goal['hedef_tutar']>0 else 0
            st.progress(progress)
        with col_b:
            if st.button("➕", key=f"add_{idx}"):
                to_add = st.number_input("Ne kadar eklemek istersiniz?", min_value=1, max_value=surplus, value=min(100, surplus))
                goal['biriken'] += to_add
                st.rerun()
        st.caption(f"💡 {suggestion}")
    if not st.session_state.goals:
        st.info("Yukarıdan bir hedef ekleyin, AI size fazla bütçeyi hedefe yönlendirmeyi önerecek.")

# 3. AI Destekli Uyarılar ve Tahminler
with col3:
    st.subheader("🤖 AI Asistan")
    # Anormal harcama tespiti (basit IQR)
    amounts = st.session_state.transactions['Tutar']
    Q1 = amounts.quantile(0.25)
    Q3 = amounts.quantile(0.75)
    IQR = Q3 - Q1
    outliers = amounts[(amounts < Q1 - 1.5*IQR) | (amounts > Q3 + 1.5*IQR)]
    if len(outliers) > 0:
        st.warning(f"⚠️ {len(outliers)} anormal harcama tespit edildi! En yüksek: {outliers.max():.0f} ₺")
    else:
        st.success("✅ Anormal harcama yok.")
    
    # Gelecek tahmini (basit linear regression)
    if len(st.session_state.transactions) > 7:
        df_temp = st.session_state.transactions.copy()
        df_temp['Gün'] = (pd.to_datetime(df_temp['Tarih']) - pd.to_datetime(df_temp['Tarih']).min()).dt.days
        X = df_temp['Gün'].values.reshape(-1,1)
        y = df_temp['Tutar'].values
        model = LinearRegression()
        model.fit(X, y)
        next_day = X.max() + 1
        pred = model.predict([[next_day]])[0]
        st.info(f"📈 Yarınki tahmini harcama: {pred:.0f} ₺")
        # Öneri
        if pred > amounts.mean() * 1.2:
            st.warning("Dikkat! Yarın yüksek harcama bekleniyor, bütçeni gözden geçir.")
    
    # Zarf aşım uyarıları
    over_budget = []
    for cat, budget in st.session_state.budget_envelopes.items():
        spent = monthly_category_spending(cat)
        if spent > budget:
            over_budget.append(f"{cat}: {spent}/{budget} ₺")
    if over_budget:
        with st.expander("🚨 Zarf Aşımları", expanded=True):
            for msg in over_budget:
                st.write(f"❌ {msg}")

# Ana grafik - Zarf kullanımı (gerçek zamanlı görsel)
st.markdown("---")
st.subheader("📊 Zarf Kullanım Durumu (Bu Ay)")

# Kategoriler ve harcamalar
cats = list(st.session_state.budget_envelopes.keys())
spent_list = [monthly_category_spending(cat) for cat in cats]
budget_list = [st.session_state.budget_envelopes[cat] for cat in cats]

fig = go.Figure()
fig.add_trace(go.Bar(x=cats, y=spent_list, name='Harcanan', marker_color='#4CAF50'))
fig.add_trace(go.Bar(x=cats, y=budget_list, name='Bütçe', marker_color='#FFA726', opacity=0.6))
fig.update_layout(barmode='group', title="Kategorilere Göre Bütçe vs Gerçekleşen", xaxis_title="Kategori", yaxis_title="Tutar (₺)")
st.plotly_chart(fig, use_container_width=True)

# Zaman serisi grafiği (son 30 gün)
st.subheader("📈 Harcama Trendi")
df_time = st.session_state.transactions.copy()
df_time['Tarih'] = pd.to_datetime(df_time['Tarih'])
df_daily = df_time.groupby('Tarih')['Tutar'].sum().reset_index()
fig2 = px.line(df_daily, x='Tarih', y='Tutar', title="Günlük Toplam Harcama")
st.plotly_chart(fig2, use_container_width=True)

# Footer
st.markdown("---")
st.caption("💰 Smart Budget AI | Proaktif hedef yönetimi + AI destekli anormallik tespiti ve tahmin | Sade ve etkili")
