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
if 'pipeline_fitted' not in st.session_state:
    st.session_state.pipeline_fitted = False

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

# ============================================================================
# ADIM 1: VERİ YÜKLEME VE PROBLEM TİPİ SEÇİMİ
# ============================================================================
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

# ============================================================================
# ADIM 2: GENEL EDA
# ============================================================================
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
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_df.set_index('Değişken')['Eksik Yüzde (%)'].plot(kind='barh', ax=ax)
            ax.set_xlabel('Eksik Değer Yüzdesi (%)')
            ax.set_title('Değişkenlere Göre Eksik Değer Oranları')
            st.pyplot(fig)
            st.info("💡 **MCAR değerlendirmesi:** Eksik değerler rastgele görünüyorsa MCAR, belirli bir paternle eksikse MAR/MNAR olabilir.")
        else:
            st.success("✅ Veri setinde eksik değer bulunmamaktadır!")
    
    with tab3:
        st.subheader("Korelasyon Analizi")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            st.markdown("**Sayısal Değişkenler - Pearson Korelasyonu**")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Pearson Korelasyon Isı Haritası')
            st.pyplot(fig)
        else:
            st.info("Korelasyon analizi için en az 2 sayısal değişken gereklidir.")
    
    with tab4:
        st.subheader("Değişken Dağılımları")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            selected_col = st.selectbox("Sayısal değişken seçin", numeric_cols)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[selected_col].hist(bins=30, edgecolor='black', ax=ax)
                ax.set_title(f'{selected_col} - Histogram')
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                df.boxplot(column=selected_col, ax=ax)
                ax.set_title(f'{selected_col} - Box Plot')
                st.pyplot(fig)
            if len(df[selected_col].dropna()) > 3:
                stat, p_value = stats.shapiro(df[selected_col].dropna()[:5000])
                if p_value > 0.05:
                    st.success(f"✅ Shapiro-Wilk p-value = {p_value:.4f} → Normal dağılıma yakın")
                else:
                    st.warning(f"⚠️ Shapiro-Wilk p-value = {p_value:.4f} → Normal dağılımdan sapma var")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("⬅️ Geri"):
            st.session_state.step = 1
            st.rerun()
    with col2:
        if st.button("İlerle ➡️", type="primary"):
            st.session_state.step = 3
            st.rerun()

# ============================================================================
# ADIM 3: DEĞİŞKEN TİPLERİNİ GÖSTER VE ÖNER
# ============================================================================
elif st.session_state.step == 3:
    st.header("3️⃣ Değişken Tiplerini Göster ve Öner")
    df = st.session_state.data
    
    current_types = df.dtypes.to_dict()
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
    
    type_data = []
    for col in df.columns:
        type_data.append({
            'Değişken': col,
            'Mevcut Tip': str(current_types[col]),
            'Benzersiz Değer Sayısı': df[col].nunique(),
            'Önerilen Tip': suggestions[col],
            'Örnek Değerler': str(df[col].dropna().iloc[:3].tolist())
        })
    st.dataframe(pd.DataFrame(type_data), use_container_width=True, height=400)
    
    st.markdown("---")
    st.subheader("Tip Dönüşümleri")
    if 'confirmed_types' not in st.session_state:
        st.session_state.confirmed_types = suggestions.copy()
    
    with st.expander("🔧 Değişken Tiplerini Düzenle", expanded=False):
        for col in df.columns:
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

# ============================================================================
# ADIM 4: DEĞİŞKENLERİ SINIFLANDIRMA
# ============================================================================
elif st.session_state.step == 4:
    st.header("4️⃣ Değişken Sınıflandırması")
    df = st.session_state.data
    var_types = st.session_state.variable_types
    
    numerical = [col for col, typ in var_types.items() if typ == "numerical"]
    categorical = [col for col, typ in var_types.items() if typ == "categorical"]
    high_card = [col for col, typ in var_types.items() if typ == "categorical (yüksek kardinalite)"]
    binary = [col for col, typ in var_types.items() if "binary" in typ]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🔢 Sayısal", len(numerical))
    col2.metric("📂 Kategorik", len(categorical))
    col3.metric("🔺 Yüksek Kardinalite", len(high_card))
    col4.metric("⚡ Binary", len(binary))
    
    st.markdown("---")
    tab1, tab2, tab3, tab4 = st.tabs(["🔢 Sayısal", "📂 Kategorik", "🔺 Yüksek Kardinalite", "⚡ Binary"])
    with tab1:
        if numerical:
            num_data = [{'Değişken': col, 'Min': df[col].min(), 'Max': df[col].max(), 'Ortalama': df[col].mean(), 'Std': df[col].std()} for col in numerical]
            st.dataframe(pd.DataFrame(num_data), use_container_width=True)
        else:
            st.info("Sayısal değişken yok.")
    with tab2:
        if categorical:
            cat_data = [{'Değişken': col, 'Benzersiz Değer': df[col].nunique(), 'En Sık Değer': df[col].mode()[0] if len(df[col].mode())>0 else None} for col in categorical]
            st.dataframe(pd.DataFrame(cat_data), use_container_width=True)
        else:
            st.info("Kategorik değişken yok.")
    with tab3:
        if high_card:
            hc_data = [{'Değişken': col, 'Benzersiz Değer': df[col].nunique(), 'Kardinalite Oranı (%)': (df[col].nunique()/len(df)*100).round(2)} for col in high_card]
            st.dataframe(pd.DataFrame(hc_data), use_container_width=True)
            st.warning("⚠️ Yüksek kardinaliteli değişkenler için Target Encoding önerilir.")
        else:
            st.info("Yüksek kardinalite yok.")
    with tab4:
        if binary:
            bin_data = [{'Değişken': col, 'Değerler': df[col].unique().tolist()} for col in binary]
            st.dataframe(pd.DataFrame(bin_data), use_container_width=True)
        else:
            st.info("Binary değişken yok.")
    
    col1, col2 = st.columns([1,1])
    with col1:
        if st.button("⬅️ Geri"):
            st.session_state.step = 3
            st.rerun()
    with col2:
        if st.button("İlerle ➡️", type="primary"):
            st.session_state.step = 5
            st.rerun()

# ============================================================================
# ADIM 5: VERİ BÖLME VE HEDEF BELİRLEME
# ============================================================================
elif st.session_state.step == 5:
    st.header("5️⃣ Veri Bölme ve Hedef Değişken Belirleme")
    df = st.session_state.data
    problem_type = st.session_state.problem_type
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Hedef Değişken")
        if problem_type == "Kümeleme":
            st.warning("Kümeleme için hedef değişken gerekmez.")
            target = None
        else:
            target = st.selectbox("Hedef değişkeni seçin", ["Seçiniz"] + list(df.columns))
            if target != "Seçiniz":
                st.success(f"Hedef: {target}")
                if problem_type == "Sınıflandırma":
                    st.bar_chart(df[target].value_counts())
    with col2:
        st.subheader("Ayarlar")
        test_size = st.slider("Test seti oranı", 0.1, 0.5, 0.2, 0.05)
        random_state = st.number_input("Random State", 0, 100, 42)
        drop_cols = st.multiselect("Çıkarılacak değişkenler", df.columns.tolist())
    
    st.markdown("---")
    if problem_type == "Kümeleme":
        ready = True
    else:
        ready = target != "Seçiniz"
    
    if ready:
        if st.button("✅ Veriyi Böl ve Devam Et", type="primary", use_container_width=True):
            from sklearn.model_selection import train_test_split
            features = [col for col in df.columns if col != target and col not in drop_cols] if problem_type != "Kümeleme" else [col for col in df.columns if col not in drop_cols]
            st.session_state.target = target if problem_type != "Kümeleme" else None
            st.session_state.features = features
            st.session_state.test_size = test_size
            st.session_state.random_state = random_state
            
            if problem_type != "Kümeleme":
                X = df[features]
                y = df[target]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.session_state.y_train, st.session_state.y_test = y_train, y_test
            else:
                X = df[features]
                X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
            
            st.success("Veri başarıyla bölündü!")
            st.session_state.step = 6
            st.rerun()
    else:
        st.error("Lütfen hedef değişkeni seçin.")
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 4
        st.rerun()

# ============================================================================
# ADIM 6: FEATURE ENGINEERING
# ============================================================================
elif st.session_state.step == 6:
    st.header("6️⃣ Feature Engineering (Özellik Mühendisliği)")
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    
    st.write(f"**Başlangıç değişken sayısı:** {X_train.shape[1]}")
    from sklearn.preprocessing import PolynomialFeatures
    
    col1, col2 = st.columns(2)
    with col1:
        poly_degree = st.slider("Polynomial derecesi", 1, 3, 2)
    with col2:
        include_bias = st.checkbox("Bias terimi ekle", False)
    
    if st.button("🚀 Feature Engineering Uygula"):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=include_bias, interaction_only=False)
            X_train_poly = poly.fit_transform(X_train[numeric_cols])
            X_test_poly = poly.transform(X_test[numeric_cols])
            feature_names = poly.get_feature_names_out(numeric_cols)
            X_train_poly_df = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
            X_test_poly_df = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)
            
            cat_cols = X_train.select_dtypes(exclude=[np.number]).columns
            for col in cat_cols:
                X_train_poly_df[col] = X_train[col].values
                X_test_poly_df[col] = X_test[col].values
            
            st.session_state.X_train = X_train_poly_df
            st.session_state.X_test = X_test_poly_df
            new_count = X_train_poly_df.shape[1]
            st.success(f"Yeni değişken sayısı: {new_count}")
            if new_count > 2 * X_train.shape[1]:
                st.warning("Değişken sayısı 2 katı aştı, düşük varyanslılar siliniyor.")
                variances = X_train_poly_df.var()
                low_var = variances.nsmallest(int(new_count/2)).index
                st.session_state.X_train = X_train_poly_df.drop(columns=low_var)
                st.session_state.X_test = X_test_poly_df.drop(columns=low_var)
            st.session_state.step = 7
            st.rerun()
        else:
            st.error("Sayısal değişken olmadan feature engineering yapılamaz.")
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 5
        st.rerun()

# ============================================================================
# ADIM 7: AYKIRI VE EKSİK DEĞER YÖNETİMİ
# ============================================================================
elif st.session_state.step == 7:
    st.header("7️⃣ Aykırı ve Eksik Değer Yönetimi")
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    
    missing_train = X_train.isnull().sum()
    missing_train = missing_train[missing_train > 0]
    
    if len(missing_train) > 0:
        st.subheader("Eksik Değerler")
        st.dataframe(pd.DataFrame({'Değişken': missing_train.index, 'Eksik Sayı': missing_train.values, 'Eksik %': (missing_train/len(X_train)*100).round(2)}))
        method = st.selectbox("Eksik değer doldurma yöntemi", ["Medyan (sayısal) / Mod (kategorik)", "Ortalama", "KNN Imputer", "Sil (listwise)"])
        if st.button("Uygula"):
            from sklearn.impute import SimpleImputer, KNNImputer
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            categorical_cols = X_train.select_dtypes(exclude=[np.number]).columns
            if method == "Sil (listwise)":
                X_train = X_train.dropna()
                X_test = X_test.dropna()
            else:
                if method == "Medyan (sayısal) / Mod (kategorik)":
                    num_imp = SimpleImputer(strategy='median')
                    cat_imp = SimpleImputer(strategy='most_frequent')
                elif method == "Ortalama":
                    num_imp = SimpleImputer(strategy='mean')
                    cat_imp = SimpleImputer(strategy='most_frequent')
                else:
                    num_imp = KNNImputer(n_neighbors=5)
                    cat_imp = SimpleImputer(strategy='most_frequent')
                if len(numeric_cols)>0:
                    X_train[numeric_cols] = num_imp.fit_transform(X_train[numeric_cols])
                    X_test[numeric_cols] = num_imp.transform(X_test[numeric_cols])
                if len(categorical_cols)>0:
                    X_train[categorical_cols] = cat_imp.fit_transform(X_train[categorical_cols])
                    X_test[categorical_cols] = cat_imp.transform(X_test[categorical_cols])
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.success("Eksik değerler işlendi.")
            st.session_state.step = 8
            st.rerun()
    else:
        st.success("Eksik değer yok.")
        if st.button("Devam Et ➡️"):
            st.session_state.step = 8
            st.rerun()
    
    # Aykırı değer (opsiyonel)
    with st.expander("Aykırı Değer Analizi (Opsiyonel)"):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols)>0:
            col = st.selectbox("Değişken seç", numeric_cols)
            fig, ax = plt.subplots()
            X_train.boxplot(column=col, ax=ax)
            st.pyplot(fig)
            outlier_method = st.radio("Strateji", ["Hiçbir şey yapma", "Winsorizing", "Sil (IQR)"])
            if outlier_method != "Hiçbir şey yapma":
                Q1 = X_train[col].quantile(0.25)
                Q3 = X_train[col].quantile(0.75)
                IQR = Q3 - Q1
                lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
                if outlier_method == "Winsorizing":
                    X_train[col] = X_train[col].clip(lower, upper)
                    X_test[col] = X_test[col].clip(lower, upper)
                else:
                    mask = (X_train[col] >= lower) & (X_train[col] <= upper)
                    X_train = X_train[mask]
                    if st.session_state.y_train is not None:
                        st.session_state.y_train = st.session_state.y_train[mask]
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.success("Aykırı değer işlemi tamamlandı.")
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 6
        st.rerun()

# ============================================================================
# ADIM 8: ENCODING VE SCALING
# ============================================================================
elif st.session_state.step == 8:
    st.header("8️⃣ Encoding ve Scaling İşlemleri")
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        encoding_method = st.selectbox("Encoding yöntemi", ["One-Hot Encoding", "Label Encoding", "Target Encoding"])
        if st.button("Encoding Uygula"):
            if encoding_method == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder
                for col in categorical_cols:
                    le = LabelEncoder()
                    X_train[col] = le.fit_transform(X_train[col])
                    X_test[col] = le.transform(X_test[col])
            elif encoding_method == "One-Hot Encoding":
                X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
                X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
                missing_cols = set(X_train.columns) - set(X_test.columns)
                for col in missing_cols:
                    X_test[col] = 0
                X_test = X_test[X_train.columns]
            else:  # Target Encoding - sadece denetimli
                if st.session_state.problem_type != "Kümeleme" and st.session_state.y_train is not None:
                    from sklearn.preprocessing import LabelEncoder
                    for col in categorical_cols:
                        le = LabelEncoder()
                        X_train[col] = le.fit_transform(X_train[col])
                        X_test[col] = le.transform(X_test[col])
                    st.info("Target Encoding simülasyonu için LabelEncoder kullanıldı.")
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.success("Encoding tamamlandı.")
            st.rerun()
    
    # Scaling
    st.subheader("Scaling")
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        scaling_method = st.selectbox("Scaling yöntemi", ["StandardScaler", "MinMaxScaler", "RobustScaler"])
        if st.button("Scaling Uygula"):
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            scaler = {"StandardScaler": StandardScaler(), "MinMaxScaler": MinMaxScaler(), "RobustScaler": RobustScaler()}[scaling_method]
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.scaler = scaler
            st.success("Scaling tamamlandı.")
            st.session_state.step = 9
            st.rerun()
    else:
        st.session_state.step = 9
        st.rerun()
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 7
        st.rerun()

# ============================================================================
# ADIM 9: MODEL KARŞILAŞTIRMA (İLK MODELLER)
# ============================================================================
elif st.session_state.step == 9:
    st.header("9️⃣ Model Karşılaştırması (Hiperparametre optimizasyonu yok)")
    problem = st.session_state.problem_type
    
    if problem == "Regresyon":
        from sklearn.linear_model import LinearRegression, Ridge
        from sklearn.ensemble import RandomForestRegressor
        from xgboost import XGBRegressor
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge": Ridge(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": XGBRegressor(n_estimators=100, random_state=42)
        }
        metrics_names = ["R2", "RMSE", "MAE"]
    elif problem == "Sınıflandırma":
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.svm import SVC
        models = {
            "Lojistik Regresyon": LogisticRegression(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(random_state=42)
        }
        metrics_names = ["Accuracy", "F1", "AUC"]
    else:  # Kümeleme
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.mixture import GaussianMixture
        models = {
            "K-Means": KMeans(n_clusters=3, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "DBSCAN": DBSCAN(),
            "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42)
        }
        metrics_names = ["Silhouette Score", "Davies-Bouldin"]
    
    selected_models = st.multiselect("Kıyaslanacak modelleri seçin", list(models.keys()), default=list(models.keys()))
    if st.button("Modelleri Çalıştır"):
        results = []
        for name in selected_models:
            model = models[name]
            if problem != "Kümeleme":
                model.fit(st.session_state.X_train, st.session_state.y_train)
                y_pred = model.predict(st.session_state.X_test)
                if problem == "Regresyon":
                    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                    r2 = r2_score(st.session_state.y_test, y_pred)
                    rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
                    mae = mean_absolute_error(st.session_state.y_test, y_pred)
                    results.append({"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
                    fig, ax = plt.subplots()
                    ax.scatter(st.session_state.y_test, y_pred, alpha=0.5)
                    ax.plot([y_pred.min(), y_pred.max()], [y_pred.min(), y_pred.max()], 'r--')
                    ax.set_xlabel("Gerçek")
                    ax.set_ylabel("Tahmin")
                    ax.set_title(f"{name} - Gerçek vs Tahmin")
                    st.pyplot(fig)
                else:
                    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                    acc = accuracy_score(st.session_state.y_test, y_pred)
                    f1 = f1_score(st.session_state.y_test, y_pred, average='weighted')
                    auc = roc_auc_score(st.session_state.y_test, model.predict_proba(st.session_state.X_test)[:,1]) if hasattr(model, "predict_proba") else 0
                    results.append({"Model": name, "Accuracy": acc, "F1": f1, "AUC": auc})
            else:
                # Kümeleme
                clusters = model.fit_predict(st.session_state.X_train)
                if hasattr(model, "labels_"):
                    clusters = model.labels_
                from sklearn.metrics import silhouette_score, davies_bouldin_score
                sil = silhouette_score(st.session_state.X_train, clusters) if len(set(clusters))>1 else -1
                db = davies_bouldin_score(st.session_state.X_train, clusters) if len(set(clusters))>1 else -1
                results.append({"Model": name, "Silhouette": sil, "Davies-Bouldin": db})
        
        st.session_state.model_results = pd.DataFrame(results)
        st.dataframe(st.session_state.model_results)
        best_model_name = st.session_state.model_results.iloc[0]["Model"]  # ilk sıradaki
        st.session_state.best_model_name = best_model_name
        st.session_state.best_model = models[best_model_name]
        st.success(f"En iyi model (ilk): {best_model_name}")
        if st.button("Hiperparametre Optimizasyonuna Geç"):
            st.session_state.step = 10
            st.rerun()
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 8
        st.rerun()

# ============================================================================
# ADIM 10: HİPERPARAMETRE OPTİMİZASYONU
# ============================================================================
elif st.session_state.step == 10:
    st.header("🔟 Hiperparametre Optimizasyonu")
    from sklearn.model_selection import GridSearchCV
    best_model = st.session_state.best_model
    problem = st.session_state.problem_type
    
    param_grids = {
        "RandomForestRegressor": {"n_estimators": [50,100], "max_depth": [None,10]},
        "XGBRegressor": {"n_estimators": [50,100], "learning_rate": [0.01,0.1]},
        "RandomForestClassifier": {"n_estimators": [50,100], "max_depth": [None,10]},
        "XGBClassifier": {"n_estimators": [50,100], "learning_rate": [0.01,0.1]},
        "SVC": {"C": [0.1,1,10], "kernel": ["linear","rbf"]},
        "Ridge": {"alpha": [0.1,1,10]},
        "LinearRegression": {},  # Hiperparametre yok
    }
    model_class = best_model.__class__.__name__
    param_grid = param_grids.get(model_class, {})
    
    if param_grid:
        st.write("Seçilen model için hiperparametreler:", param_grid)
        time_est = st.slider("Tahmini süre (dakika)", 0.5, 10.0, 2.0)
        if st.button("Grid Search Başlat"):
            if problem != "Kümeleme":
                gs = GridSearchCV(best_model, param_grid, cv=3, scoring='r2' if problem=="Regresyon" else 'accuracy')
                gs.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.best_model = gs.best_estimator_
                st.success(f"En iyi parametreler: {gs.best_params_}")
                st.session_state.step = 11
                st.rerun()
            else:
                st.warning("Kümeleme için hiperparametre optimizasyonu bu sürümde manuel yapılmalıdır.")
    else:
        st.info("Bu modelin hiperparametre seçeneği yok veya tanımlanmamış.")
        if st.button("Model Yorumlamaya Geç"):
            st.session_state.step = 11
            st.rerun()
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 9
        st.rerun()

# ============================================================================
# ADIM 11: MODEL YORUMLAMA
# ============================================================================
elif st.session_state.step == 11:
    st.header("1️⃣1️⃣ Model Yorumlama")
    model = st.session_state.best_model
    X_train = st.session_state.X_train
    problem = st.session_state.problem_type
    
    if problem != "Kümeleme":
        # Feature importance
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feat_imp = pd.DataFrame({"Feature": X_train.columns, "Importance": importances}).sort_values("Importance", ascending=False).head(10)
            st.subheader("En Önemli 10 Değişken")
            st.bar_chart(feat_imp.set_index("Feature"))
        elif hasattr(model, "coef_"):
            coef = model.coef_.flatten() if len(model.coef_.shape)>1 else model.coef_
            feat_imp = pd.DataFrame({"Feature": X_train.columns, "Coefficient": coef}).sort_values("Coefficient", key=abs, ascending=False).head(10)
            st.subheader("En Büyük Katsayılar")
            st.dataframe(feat_imp)
        else:
            st.info("Bu model için değişken önemi gösterilemiyor.")
        
        # SHAP (isteğe bağlı)
        try:
            import shap
            explainer = shap.TreeExplainer(model) if "tree" in str(type(model)).lower() else shap.Explainer(model, X_train)
            shap_values = explainer.shap_values(X_train.iloc[:100])
            st.subheader("SHAP Özet Grafiği")
            fig, ax = plt.subplots()
            shap.summary_plot(shap_values, X_train.iloc[:100], show=False)
            st.pyplot(plt.gcf())
        except:
            st.info("SHAP analizi yapılamadı.")
        
        if st.button("Canlı Deneme Bölümüne Geç"):
            st.session_state.step = 12
            st.rerun()
    else:
        st.info("Kümeleme modeli yorumlaması için küme profilleri gösteriliyor.")
        clusters = model.fit_predict(X_train)
        X_train['Cluster'] = clusters
        st.dataframe(X_train.groupby('Cluster').mean())
        if st.button("Canlı Deneme"):
            st.session_state.step = 12
            st.rerun()
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 10
        st.rerun()

# ============================================================================
# ADIM 12: CANLI DENEME
# ============================================================================
elif st.session_state.step == 12:
    st.header("1️⃣2️⃣ Canlı Deneme (Gerçek Dünya Tahmini)")
    model = st.session_state.best_model
    X_train = st.session_state.X_train
    feature_names = X_train.columns.tolist()
    
    st.write("Lütfen aşağıdaki 5 değişken için değer girin (gerçek dünya verisi)")
    # Kullanıcının 5 değişken seçmesine izin ver
    selected_features = st.multiselect("En fazla 5 değişken seçin", feature_names, default=feature_names[:min(5, len(feature_names))])
    if len(selected_features) > 5:
        st.warning("Lütfen en fazla 5 değişken seçin.")
    else:
        user_input = {}
        for feat in selected_features:
            if X_train[feat].dtype in ['float64','int64']:
                val = st.number_input(f"{feat} (sayısal)", value=float(X_train[feat].mean()))
            else:
                options = X_train[feat].dropna().unique().tolist()
                val = st.selectbox(f"{feat} (kategorik)", options)
            user_input[feat] = val
        
        if st.button("Tahmin Yap"):
            # Eksik değişkenleri ortalama ile doldur (basit)
            input_df = pd.DataFrame([user_input])
            for col in feature_names:
                if col not in input_df.columns:
                    if X_train[col].dtype in ['float64','int64']:
                        input_df[col] = X_train[col].mean()
                    else:
                        input_df[col] = X_train[col].mode()[0]
            input_df = input_df[feature_names]  # sıralama
            prediction = model.predict(input_df)[0]
            st.success(f"Tahmin sonucu: {prediction}")
            
            # Uyum grafiği (gerçek dünya verisi yoksa tahmin dağılımı)
            if st.session_state.problem_type != "Kümeleme":
                y_pred_all = model.predict(st.session_state.X_test)
                fig, ax = plt.subplots()
                ax.hist(y_pred_all, bins=20, alpha=0.7, label="Test tahminleri")
                ax.axvline(prediction, color='red', linestyle='--', label="Kullanıcı tahmini")
                ax.set_xlabel("Tahmin değeri")
                ax.set_ylabel("Frekans")
                ax.legend()
                st.pyplot(fig)
    
    if st.button("⬅️ Geri"):
        st.session_state.step = 11
        st.rerun()

# Footer
st.markdown("---")
st.caption("🤖 AutoML Pipeline | 12 adımda tam otomatik veri bilimi")
