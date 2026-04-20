import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(page_title="AutoML Pipeline", layout="wide", page_icon="🤖")

# Session state başlatma
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'variable_types' not in st.session_state:
    st.session_state.variable_types = {}

# Başlık
st.title("🤖 Otomatik Veri Bilimi Pipeline")
st.markdown("---")

# Sidebar - İlerleme göstergesi
with st.sidebar:
    st.header("📊 İlerleme")
    steps = [
        "1️⃣ Veri Yükleme",
        "2️⃣ Genel EDA",
        "3️⃣ Değişken Tipleri",
        "4️⃣ Değişken Sınıflandırma",
        "5️⃣ Veri Bölme",
        "6️⃣ Feature Engineering",
        "7️⃣ Aykırı/Eksik Değer",
        "8️⃣ Encoding/Scaling",
        "9️⃣ Modelleme",
        "🔟 Hiperparametre",
        "1️⃣1️⃣ Model Yorumlama",
        "1️⃣2️⃣ Canlı Deneme"
    ]
    
    for i, step in enumerate(steps, 1):
        if i < st.session_state.step:
            st.success(step)
        elif i == st.session_state.step:
            st.info(f"**{step}** ⬅️")
        else:
            st.text(step)
    
    st.markdown("---")
    if st.button("🔄 Baştan Başla"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ADIM 1: VERİ YÜKLEME VE PROBLEM TİPİ SEÇİMİ
if st.session_state.step == 1:
    st.header("1️⃣ Veri Yükleme ve Problem Tipi Seçimi")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "CSV veya Excel dosyası yükleyin",
            type=['csv', 'xlsx', 'xls'],
            help="Veri setinizi yükleyin"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Veri yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
                
                # İlk 10 değişkeni göster
                st.subheader("İlk 10 Değişken")
                st.dataframe(df.iloc[:, :min(10, df.shape[1])].head())
                
                st.session_state.data = df
                
            except Exception as e:
                st.error(f"Hata: {str(e)}")
    
    with col2:
        st.subheader("Problem Tipi")
        problem_type = st.selectbox(
            "Lütfen problem tipini seçin",
            ["Seçiniz", "Regresyon", "Sınıflandırma", "Kümeleme"],
            help="Çözmek istediğiniz problem tipini belirleyin"
        )
        
        if problem_type != "Seçiniz":
            st.session_state.problem_type = problem_type
            
            # Problem tipi hakkında bilgi
            info = {
                "Regresyon": "Sürekli sayısal bir değeri tahmin etmek için kullanılır (örn: fiyat, satış miktarı).",
                "Sınıflandırma": "Kategorik bir sınıfı tahmin etmek için kullanılır (örn: evet/hayır, müşteri tipi).",
                "Kümeleme": "Verileri benzerliklerine göre gruplara ayırmak için kullanılır."
            }
            st.info(f"ℹ️ {info[problem_type]}")
    
    if st.session_state.data is not None and st.session_state.problem_type:
        if st.button("İlerle ➡️", type="primary"):
            st.session_state.step = 2
            st.rerun()

# ADIM 2: GENEL EDA
elif st.session_state.step == 2:
    st.header("2️⃣ Genel Keşifsel Veri Analizi (EDA)")
    
    df = st.session_state.data
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Temel İstatistikler", "❓ Eksik Değerler", "🔥 Korelasyon", "📈 Dağılımlar"])
    
    with tab1:
        st.subheader("Temel İstatistikler")
        st.dataframe(df.describe(include='all').T)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Toplam Satır", df.shape[0])
        col2.metric("Toplam Sütun", df.shape[1])
        col3.metric("Sayısal Sütun", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("Kategorik Sütun", len(df.select_dtypes(include=['object']).columns))
    
    with tab2:
        st.subheader("Eksik Değer Analizi")
        
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(2)
        missing_df = pd.DataFrame({
            'Değişken': missing.index,
            'Eksik Sayı': missing.values,
            'Eksik Yüzde (%)': missing_pct.values
        }).query('`Eksik Sayı` > 0')
        
        if len(missing_df) > 0:
            st.warning(f"⚠️ {len(missing_df)} değişkende eksik değer bulundu")
            st.dataframe(missing_df, use_container_width=True)
            
            # Görsel
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_df.set_index('Değişken')['Eksik Yüzde (%)'].plot(kind='barh', ax=ax)
            ax.set_xlabel('Eksik Değer Yüzdesi (%)')
            ax.set_title('Değişkenlere Göre Eksik Değer Oranları')
            st.pyplot(fig)
            
            # MCAR testi (Little's test yerine basitleştirilmiş)
            st.info("💡 **MCAR (Missing Completely At Random) Değerlendirmesi**\n\n"
                   "Eksik değerler rastgele görünüyorsa MCAR, belirli bir paternle eksikse MAR/MNAR olabilir. "
                   "İlerleyen adımlarda uygun doldurma yöntemi seçilecek.")
        else:
            st.success("✅ Veri setinde eksik değer bulunmamaktadır!")
    
    with tab3:
        st.subheader("Korelasyon Analizi")
        
        # Sayısal değişkenler için Pearson
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("**Sayısal Değişkenler - Pearson Korelasyonu**")
            corr = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Pearson Korelasyon Isı Haritası')
            st.pyplot(fig)
            
            # Yüksek korelasyonlar
            high_corr = []
            for i in range(len(corr.columns)):
                for j in range(i+1, len(corr.columns)):
                    if abs(corr.iloc[i, j]) > 0.7:
                        high_corr.append({
                            'Değişken 1': corr.columns[i],
                            'Değişken 2': corr.columns[j],
                            'Korelasyon': corr.iloc[i, j]
                        })
            
            if high_corr:
                st.warning(f"⚠️ Yüksek korelasyonlu değişken çiftleri (|r| > 0.7):")
                st.dataframe(pd.DataFrame(high_corr))
        else:
            st.info("Korelasyon analizi için en az 2 sayısal değişken gereklidir.")
        
        # Kategorik değişkenler için Cramer's V
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 1:
            st.markdown("**Kategorik Değişkenler - Cramer's V**")
            
            def cramers_v(x, y):
                confusion_matrix = pd.crosstab(x, y)
                chi2 = chi2_contingency(confusion_matrix)[0]
                n = confusion_matrix.sum().sum()
                min_dim = min(confusion_matrix.shape) - 1
                return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
            
            cramers_matrix = pd.DataFrame(
                [[cramers_v(df[col1], df[col2]) if col1 != col2 else 1.0 
                  for col2 in cat_cols[:5]] for col1 in cat_cols[:5]],
                index=cat_cols[:5],
                columns=cat_cols[:5]
            )
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cramers_matrix, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
            ax.set_title("Cramer's V - Kategorik İlişki Gücü")
            st.pyplot(fig)
    
    with tab4:
        st.subheader("Değişken Dağılımları")
        
        # Sayısal değişkenler için
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Sayısal değişken seçin", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig, ax = plt.subplots(figsize=(8, 4))
                df[selected_col].hist(bins=30, edgecolor='black', ax=ax)
                ax.set_title(f'{selected_col} - Histogram')
                ax.set_xlabel(selected_col)
                ax.set_ylabel('Frekans')
                st.pyplot(fig)
            
            with col2:
                # Box plot
                fig, ax = plt.subplots(figsize=(8, 4))
                df.boxplot(column=selected_col, ax=ax)
                ax.set_title(f'{selected_col} - Box Plot')
                st.pyplot(fig)
            
            # Normallik testi
            if len(df[selected_col].dropna()) > 3:
                stat, p_value = stats.shapiro(df[selected_col].dropna()[:5000])  # Shapiro max 5000 örnek
                
                if p_value > 0.05:
                    st.success(f"✅ **Shapiro-Wilk Testi**: p-value = {p_value:.4f} > 0.05 → Normal dağılıma yakın")
                else:
                    st.warning(f"⚠️ **Shapiro-Wilk Testi**: p-value = {p_value:.4f} < 0.05 → Normal dağılımdan sapma var")
            
            # Q-Q Plot
            fig, ax = plt.subplots(figsize=(8, 4))
            stats.probplot(df[selected_col].dropna(), dist="norm", plot=ax)
            ax.set_title(f'{selected_col} - Q-Q Plot')
            st.pyplot(fig)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Geri"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("İlerle ➡️", type="primary"):
            st.session_state.step = 3
            st.rerun()

# ADIM 3: DEĞİŞKEN TİPLERİNİ GÖSTER VE ÖNER
elif st.session_state.step == 3:
    st.header("3️⃣ Değişken Tiplerini Göster ve Öner")
    
    df = st.session_state.data
    
    # Mevcut tipler
    current_types = df.dtypes.to_dict()
    
    # Otomatik öneriler
    suggestions = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_count = df[col].nunique()
            if unique_count == 2:
                suggestions[col] = "binary"
            elif unique_count <= 20:
                suggestions[col] = "categorical"
            else:
                suggestions[col] = "categorical (yüksek kardinalite)"
        elif df[col].dtype in ['int64', 'float64']:
            unique_count = df[col].nunique()
            if unique_count == 2:
                suggestions[col] = "binary (sayısal)"
            elif unique_count <= 10:
                suggestions[col] = "categorical olabilir (sayısal)"
            else:
                suggestions[col] = "numerical"
    
    st.info("💡 **Otomatik Tip Önerileri**: Aşağıda her değişken için önerilen tip gösterilmektedir. "
           "Onaylayabilir veya manuel olarak değiştirebilirsiniz.")
    
    # Tablo oluştur
    type_data = []
    for col in df.columns:
        type_data.append({
            'Değişken': col,
            'Mevcut Tip': str(current_types[col]),
            'Benzersiz Değer Sayısı': df[col].nunique(),
            'Önerilen Tip': suggestions[col],
            'Örnek Değerler': str(df[col].dropna().iloc[:3].tolist())
        })
    
    type_df = pd.DataFrame(type_data)
    st.dataframe(type_df, use_container_width=True, height=400)
    
    st.markdown("---")
    st.subheader("Tip Dönüşümleri")
    
    # Kullanıcı seçimi
    if 'confirmed_types' not in st.session_state:
        st.session_state.confirmed_types = suggestions.copy()
    
    # Her değişken için seçim
    cols_per_row = 3
    columns = list(df.columns)
    
    with st.expander("🔧 Değişken Tiplerini Düzenle", expanded=False):
        for i in range(0, len(columns), cols_per_row):
            cols = st.columns(cols_per_row)
            for j, col in enumerate(columns[i:i+cols_per_row]):
                with cols[j]:
                    st.session_state.confirmed_types[col] = st.selectbox(
                        col,
                        ["numerical", "categorical", "categorical (yüksek kardinalite)", "binary", "binary (sayısal)"],
                        index=["numerical", "categorical", "categorical (yüksek kardinalite)", "binary", "binary (sayısal)"].index(
                            suggestions[col]
                        ) if suggestions[col] in ["numerical", "categorical", "categorical (yüksek kardinalite)", "binary", "binary (sayısal)"] else 0,
                        key=f"type_{col}"
                    )
    
    if st.button("✅ Tipleri Onayla ve İlerle", type="primary"):
        st.session_state.variable_types = st.session_state.confirmed_types
        st.success("Değişken tipleri kaydedildi!")
        st.session_state.step = 4
        st.rerun()
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 2
        st.rerun()

# ADIM 4: DEĞİŞKENLERİ SINIFLANDIRMA VE İNTERAKTİF GÖSTERİM
elif st.session_state.step == 4:
    st.header("4️⃣ Değişken Sınıflandırması")
    
    df = st.session_state.data
    var_types = st.session_state.variable_types
    
    # Kategorilere ayır
    numerical = [col for col, typ in var_types.items() if typ == "numerical"]
    categorical = [col for col, typ in var_types.items() if typ == "categorical"]
    high_card = [col for col, typ in var_types.items() if typ == "categorical (yüksek kardinalite)"]
    binary = [col for col, typ in var_types.items() if "binary" in typ]
    
    # Özet kartlar
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Sayısal", len(numerical))
    col2.metric("📂 Kategorik", len(categorical))
    col3.metric("🔺 Yüksek Kardinalite", len(high_card))
    col4.metric("⚡ Binary", len(binary))
    
    # Detaylı tablo
    st.markdown("---")
    
    tab1, tab2, tab3, tab4 = st.tabs(["🔢 Sayısal", "📂 Kategorik", "🔺 Yüksek Kardinalite", "⚡ Binary"])
    
    with tab1:
        if numerical:
            num_data = []
            for col in numerical:
                num_data.append({
                    'Değişken': col,
                    'Min': df[col].min(),
                    'Max': df[col].max(),
                    'Ortalama': df[col].mean(),
                    'Std': df[col].std()
                })
            st.dataframe(pd.DataFrame(num_data), use_container_width=True)
        else:
            st.info("Sayısal değişken bulunmamaktadır.")
    
    with tab2:
        if categorical:
            cat_data = []
            for col in categorical:
                cat_data.append({
                    'Değişken': col,
                    'Benzersiz Değer': df[col].nunique(),
                    'En Sık Değer': df[col].mode()[0] if len(df[col].mode()) > 0 else None,
                    'Frekans': df[col].value_counts().iloc[0] if len(df[col].value_counts()) > 0 else 0
                })
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True)
        else:
            st.info("Kategorik değişken bulunmamaktadır.")
    
    with tab3:
        if high_card:
            hc_data = []
            for col in high_card:
                hc_data.append({
                    'Değişken': col,
                    'Benzersiz Değer': df[col].nunique(),
                    'Kardinalite Oranı (%)': (df[col].nunique() / len(df) * 100).round(2)
                })
            st.dataframe(pd.DataFrame(hc_data), use_container_width=True)
            st.warning("⚠️ Yüksek kardinaliteli değişkenler için özel encoding yöntemleri (Target Encoding, vb.) gerekebilir.")
        else:
            st.info("Yüksek kardinaliteli değişken bulunmamaktadır.")
    
    with tab4:
        if binary:
            bin_data = []
            for col in binary:
                bin_data.append({
                    'Değişken': col,
                    'Değer 1': df[col].unique()[0],
                    'Değer 2': df[col].unique()[1] if df[col].nunique() > 1 else None,
                    'Dağılım': f"{(df[col].value_counts().iloc[0] / len(df) * 100).round(1)}% / {(df[col].value_counts().iloc[1] / len(df) * 100).round(1)}%" if df[col].nunique() > 1 else "N/A"
                })
            st.dataframe(pd.DataFrame(bin_data), use_container_width=True)
        else:
            st.info("Binary değişken bulunmamaktadır.")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Geri"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("İlerle ➡️", type="primary"):
            st.session_state.step = 5
            st.rerun()

# ADIM 5: VERİ BÖLME VE HEDEF BELİRLEME
elif st.session_state.step == 5:
    st.header("5️⃣ Veri Bölme ve Hedef Değişken Belirleme")
    
    df = st.session_state.data
    problem_type = st.session_state.problem_type
    
    st.info(f"💡 **Problem Tipi**: {problem_type}")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Hedef Değişken")
        
        if problem_type == "Kümeleme":
            st.warning("⚠️ Kümeleme için hedef değişken gerekmez. Tüm değişkenler kullanılacaktır.")
            target = None
        else:
            target = st.selectbox(
                "Hedef değişkeni seçin",
                ["Seçiniz"] + list(df.columns),
                help="Tahmin etmek istediğiniz değişkeni seçin"
            )
            
            if target != "Seçiniz":
                st.success(f"✅ Hedef: **{target}**")
                
                # Hedef değişken analizi
                if problem_type == "Regresyon":
                    st.metric("Benzersiz Değer", df[target].nunique())
                    st.metric("Ortalama", f"{df[target].mean():.2f}")
                    st.metric("Std", f"{df[target].std():.2f}")
                else:  # Sınıflandırma
                    st.write("**Sınıf Dağılımı:**")
                    class_dist = df[target].value_counts()
                    st.bar_chart(class_dist)
                    
                    # Dengesizlik kontrolü
                    if len(class_dist) > 1:
                        ratio = class_dist.max() / class_dist.min()
                        if ratio > 3:
                            st.warning(f"⚠️ Veri dengesiz (oran: {ratio:.1f}:1). SMOTE veya class weighting gerekebilir.")
    
    with col2:
        st.subheader("Ayarlar")
        
        # Test set boyutu
        test_size = st.slider(
            "Test seti oranı",
            min_value=0.1,
            max_value=0.5,
            value=0.2,
            step=0.05,
            help="Veri setinin ne kadarı test için ayrılsın?"
        )
        
        # Random state
        random_state = st.number_input(
            "Random State",
            min_value=0,
            value=42,
            help="Sonuçların tekrarlanabilir olması için sabit bir değer"
        )
        
        # Çıkarılacak değişkenler
        drop_cols = st.multiselect(
            "Çıkarılacak değişkenler (opsiyonel)",
            df.columns.tolist(),
            help="Analizden çıkarmak istediğiniz değişkenleri seçin (ID, tarih vb.)"
        )
    
    st.markdown("---")
    
    # Özet bilgi
    st.subheader("📊 Özet")
    
    if problem_type != "Kümeleme" and target != "Seçiniz":
        features = [col for col in df.columns if col != target and col not in drop_cols]
    else:
        features = [col for col in df.columns if col not in drop_cols]
    
    train_count = int(len(df) * (1 - test_size))
    test_count = len(df) - train_count
    
    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
    summary_col1.metric("Toplam Özellik", len(features))
    summary_col2.metric("Train Örnekleri", train_count)
    summary_col3.metric("Test Örnekleri", test_count)
    summary_col4.metric("Çıkarılan Değişken", len(drop_cols))
    
    # Veri bölme butonu
    st.markdown("---")
    
    if problem_type == "Kümeleme":
        ready = True
    else:
        ready = target != "Seçiniz"
    
    if ready:
        if st.button("✅ Veriyi Böl ve Devam Et", type="primary", use_container_width=True):
            from sklearn.model_selection import train_test_split
            
            # Veriyi kaydet
            st.session_state.target = target if problem_type != "Kümeleme" else None
            st.session_state.features = features
            st.session_state.drop_cols = drop_cols
            st.session_state.test_size = test_size
            st.session_state.random_state = random_state
            
            # Veriyi böl
            if problem_type != "Kümeleme":
                X = df[features]
                y = df[target]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
                st.session_state.y_train = y_train
                st.session_state.y_test = y_test
            else:
                # Kümeleme için tüm veri kullanılacak
                X = df[features]
                X_train, X_test = train_test_split(
                    X, test_size=test_size, random_state=random_state
                )
                st.session_state.X_train = X_train
                st.session_state.X_test = X_test
            
            st.success("✅ Veri başarıyla bölündü!")
            st.balloons()
            
            # Devam et (sonraki adımlarda implement edilecek)
            st.info("🎉 MVP tamamlandı! Sonraki adımlar (Feature Engineering, Modelleme vb.) eklenecek.")
            
            # Şimdilik burada dur
            # st.session_state.step = 6
            # st.rerun()
    else:
        st.error("❌ Lütfen hedef değişkeni seçin.")
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 4
        st.rerun()

# Footer
st.markdown("---")
st.caption("🤖 AutoML Pipeline | Developed with Streamlit")
