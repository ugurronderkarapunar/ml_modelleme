import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency
import missingno as msno
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="AutoML Pipeline", layout="wide", page_icon="🤖")

# Session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'data' not in st.session_state:
    st.session_state.data = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None
if 'variable_types' not in st.session_state:
    st.session_state.variable_types = {}
if 'confirmed_types' not in st.session_state:
    st.session_state.confirmed_types = {}

st.title("🤖 Otomatik Veri Bilimi Pipeline")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("📊 İlerleme")
    steps = ["1️⃣ Veri Yükleme","2️⃣ Genel EDA","3️⃣ Değişken Tipleri","4️⃣ Değişken Sınıflandırma",
             "5️⃣ Veri Bölme","6️⃣ Feature Engineering","7️⃣ Aykırı/Eksik Değer","8️⃣ Encoding/Scaling",
             "9️⃣ Modelleme","🔟 Hiperparametre","1️⃣1️⃣ Model Yorumlama","1️⃣2️⃣ Canlı Deneme"]
    for i, step in enumerate(steps,1):
        if i < st.session_state.step: st.success(step)
        elif i == st.session_state.step: st.info(f"**{step}** ⬅️")
        else: st.text(step)
    if st.button("🔄 Baştan Başla", key="reset_all"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# ========================= ADIM 1 =========================
if st.session_state.step == 1:
    st.header("1️⃣ Veri Yükleme ve Problem Tipi")
    col1, col2 = st.columns([2,1])
    with col1:
        uploaded_file = st.file_uploader("CSV/Excel yükleyin", type=['csv','xlsx','xls'])
        if uploaded_file:
            try:
                df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                st.success(f"✅ {df.shape[0]} satır, {df.shape[1]} sütun")
                st.dataframe(df.iloc[:,:min(10,df.shape[1])].head())
                st.session_state.data = df
            except Exception as e: st.error(f"Hata: {e}")
    with col2:
        problem_type = st.selectbox("Problem Tipi", ["Seçiniz","Regresyon","Sınıflandırma","Kümeleme"])
        if problem_type != "Seçiniz":
            st.session_state.problem_type = problem_type
            st.info({"Regresyon":"Sürekli sayısal tahmin","Sınıflandırma":"Kategorik sınıf","Kümeleme":"Gruplama"}[problem_type])
    if st.session_state.data is not None and st.session_state.problem_type:
        if st.button("İlerle ➡️", key="go2"): st.session_state.step = 2; st.rerun()

# ========================= ADIM 2 ========================= (MCAR/MAR güvenli)
elif st.session_state.step == 2:
    st.header("2️⃣ EDA")
    df = st.session_state.data
    tabs = st.tabs(["📊 Temel İstatistikler","❓ Eksik Değer & MCAR","🔥 Korelasyon","📈 Dağılımlar","🔍 Patern"])
    with tabs[0]:
        st.dataframe(df.describe(include='all').T)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Satır",df.shape[0]); c2.metric("Sütun",df.shape[1]); c3.metric("Sayısal",len(df.select_dtypes(include=[np.number]).columns)); c4.metric("Kategorik",len(df.select_dtypes(include=['object']).columns))
    with tabs[1]:
        missing = df.isnull().sum()
        missing_pct = (missing/len(df)*100).round(2)
        missing_df = pd.DataFrame({'Değişken':missing.index,'Eksik Sayı':missing.values,'Eksik %':missing_pct.values}).query('`Eksik Sayı`>0')
        if len(missing_df)>0:
            st.warning(f"{len(missing_df)} değişkende eksik var")
            st.dataframe(missing_df)
            fig,ax = plt.subplots(figsize=(10,4))
            missing_df.set_index('Değişken')['Eksik %'].plot(kind='barh',ax=ax)
            ax.set_xlabel('Eksik %'); st.pyplot(fig)
            # Güvenli MCAR
            missing_indicator = df.isnull().astype(int)
            cols_with_missing = [c for c in missing_indicator.columns if missing_indicator[c].sum()>0]
            valid_cols = [c for c in cols_with_missing if missing_indicator[c].nunique()>1]
            if len(valid_cols)>=2:
                try:
                    chi2,p,_,expected = chi2_contingency(missing_indicator[valid_cols].T)
                    if np.any(expected==0): st.warning("Beklenen frekans sıfır, test güvenilir değil")
                    else:
                        st.write(f"**MCAR testi:** χ²={chi2:.2f}, p={p:.4f}")
                        if p>0.05: st.success("✅ MCAR olabilir (basit doldurma yeterli)")
                        else: st.warning("⚠️ MAR/MNAR olabilir (dikkatli doldurma gerekli)")
                except Exception as e: st.warning(f"Test yapılamadı: {e}")
            else: st.info("Yeterli değişken yok, test atlandı.")
        else: st.success("Eksik değer yok")
    with tabs[2]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols)>1:
            corr = df[num_cols].corr()
            fig,ax = plt.subplots(figsize=(10,8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=ax)
            st.pyplot(fig)
        else: st.info("En az 2 sayısal değişken gerekli")
    with tabs[3]:
        num_cols = df.select_dtypes(include=[np.number]).columns
        if len(num_cols)>0:
            col = st.selectbox("Değişken seç", num_cols)
            fig,ax = plt.subplots(figsize=(8,4)); df[col].hist(bins=30, ax=ax); st.pyplot(fig)
            fig,ax = plt.subplots(figsize=(8,4)); df.boxplot(column=col, ax=ax); st.pyplot(fig)
            if len(df[col].dropna())>3:
                _,p = stats.shapiro(df[col].dropna()[:5000])
                st.success(f"Shapiro-Wilk p={p:.4f}") if p>0.05 else st.warning(f"Normal değil p={p:.4f}")
    with tabs[4]:
        if df.isnull().sum().sum()>0:
            fig,ax = plt.subplots(figsize=(12,4)); msno.matrix(df, ax=ax); st.pyplot(fig)
            fig,ax = plt.subplots(figsize=(10,6)); msno.heatmap(df, ax=ax); st.pyplot(fig)
        else: st.info("Eksik yok")
    col1,col2 = st.columns(2)
    with col1:
        if st.button("⬅️ Geri"): st.session_state.step=1; st.rerun()
    with col2:
        if st.button("İlerle ➡️", type="primary"): st.session_state.step=3; st.rerun()

# ========================= ADIM 3 ========================= (tip önerisi)
elif st.session_state.step == 3:
    st.header("3️⃣ Değişken Tipleri ve Öneri")
    df = st.session_state.data
    suggestions = {}
    for col in df.columns:
        if df[col].dtype == 'object':
            uniq = df[col].nunique()
            suggestions[col] = "binary" if uniq==2 else ("categorical" if uniq<=20 else "categorical (yüksek kardinalite)")
        elif df[col].dtype in ['int64','float64']:
            uniq = df[col].nunique()
            suggestions[col] = "binary (sayısal)" if uniq==2 else ("categorical olabilir (sayısal)" if uniq<=10 else "numerical")
        else: suggestions[col] = "other"
    type_data = [{'Değişken':col,'Mevcut Tip':str(df[col].dtype),'Benzersiz':df[col].nunique(),'Öneri':suggestions[col]} for col in df.columns]
    st.dataframe(pd.DataFrame(type_data), use_container_width=True)
    if 'confirmed_types' not in st.session_state or set(st.session_state.confirmed_types.keys())!=set(df.columns):
        st.session_state.confirmed_types = suggestions.copy()
    with st.expander("🔧 Düzenle", expanded=False):
        for col in df.columns:
            st.session_state.confirmed_types[col] = st.selectbox(col, ["numerical","categorical","categorical (yüksek kardinalite)","binary","binary (sayısal)","other"],
                index=["numerical","categorical","categorical (yüksek kardinalite)","binary","binary (sayısal)","other"].index(st.session_state.confirmed_types.get(col,suggestions[col])),
                key=f"type_{col}")
    if st.button("✅ Onayla ve İlerle", type="primary"): st.session_state.variable_types = st.session_state.confirmed_types; st.session_state.step=4; st.rerun()
    if st.button("⬅️ Geri"): st.session_state.step=2; st.rerun()

# ========================= ADIM 4 ========================= (sınıflandırma)
elif st.session_state.step == 4:
    st.header("4️⃣ Değişken Sınıflandırma")
    df = st.session_state.data
    var_types = st.session_state.variable_types
    numerical = [c for c,t in var_types.items() if t=="numerical"]
    categorical = [c for c,t in var_types.items() if t=="categorical"]
    high_card = [c for c,t in var_types.items() if t=="categorical (yüksek kardinalite)"]
    binary = [c for c,t in var_types.items() if "binary" in t]
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("🔢 Sayısal",len(numerical)); c2.metric("📂 Kategorik",len(categorical)); c3.metric("🔺 Yüksek Kardinalite",len(high_card)); c4.metric("⚡ Binary",len(binary))
    tabs = st.tabs(["🔢 Sayısal","📂 Kategorik","🔺 Yüksek","⚡ Binary"])
    with tabs[0]:
        if numerical: st.dataframe(pd.DataFrame([{'Değişken':c,'Min':df[c].min(),'Max':df[c].max()} for c in numerical]))
        else: st.info("Yok")
    with tabs[1]:
        if categorical: st.dataframe(pd.DataFrame([{'Değişken':c,'Benzersiz':df[c].nunique()} for c in categorical]))
        else: st.info("Yok")
    with tabs[2]:
        if high_card: st.dataframe(pd.DataFrame([{'Değişken':c,'Kardinalite':df[c].nunique(),'Oran %':(df[c].nunique()/len(df)*100).round(2)} for c in high_card]))
        else: st.info("Yok")
    with tabs[3]:
        if binary: st.dataframe(pd.DataFrame([{'Değişken':c,'Değerler':df[c].unique().tolist()} for c in binary]))
        else: st.info("Yok")
    if st.button("⬅️ Geri"): st.session_state.step=3; st.rerun()
    if st.button("İlerle ➡️", type="primary"): st.session_state.step=5; st.rerun()

# ========================= ADIM 5 ========================= (bölme)
elif st.session_state.step == 5:
    st.header("5️⃣ Veri Bölme ve Hedef")
    df = st.session_state.data
    problem = st.session_state.problem_type
    col1,col2 = st.columns(2)
    with col1:
        if problem=="Kümeleme": target=None; st.warning("Kümelemede hedef yok")
        else:
            target = st.selectbox("Hedef değişken", ["Seçiniz"]+list(df.columns))
            if target!="Seçiniz": st.bar_chart(df[target].value_counts()) if problem=="Sınıflandırma" else st.metric("Benzersiz",df[target].nunique())
    with col2:
        test_size = st.slider("Test oranı",0.1,0.5,0.2,0.05)
        random_state = st.number_input("Random state",0,100,42)
        drop_cols = st.multiselect("Çıkarılacaklar", df.columns.tolist())
    ready = True if problem=="Kümeleme" else (target!="Seçiniz")
    if ready:
        if st.button("✅ Böl ve Devam", type="primary", use_container_width=True):
            from sklearn.model_selection import train_test_split
            features = [c for c in df.columns if c!=target and c not in drop_cols] if problem!="Kümeleme" else [c for c in df.columns if c not in drop_cols]
            st.session_state.features = features
            if problem!="Kümeleme":
                X,y = df[features], df[target]
                X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=random_state)
                st.session_state.X_train,st.session_state.X_test = X_train,X_test
                st.session_state.y_train,st.session_state.y_test = y_train,y_test
            else:
                X = df[features]
                X_train,X_test = train_test_split(X,test_size=test_size,random_state=random_state)
                st.session_state.X_train,st.session_state.X_test = X_train,X_test
            st.success("Bölündü!"); st.session_state.step=6; st.rerun()
    else: st.error("Hedef seçin")
    if st.button("⬅️ Geri"): st.session_state.step=4; st.rerun()

# ========================= ADIM 6 ========================= (feature eng)
elif st.session_state.step == 6:
    st.header("6️⃣ Feature Engineering")
    X_train, X_test = st.session_state.X_train, st.session_state.X_test
    st.write(f"Başlangıç: {X_train.shape[1]} değişken")
    from sklearn.preprocessing import PolynomialFeatures
    poly_degree = st.slider("Polynomial derece",1,3,2)
    include_bias = st.checkbox("Bias ekle",False)
    if st.button("Uygula", key="fe"):
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric_cols)>0:
            poly = PolynomialFeatures(degree=poly_degree, include_bias=include_bias)
            X_train_poly = poly.fit_transform(X_train[numeric_cols])
            X_test_poly = poly.transform(X_test[numeric_cols])
            feature_names = poly.get_feature_names_out(numeric_cols)
            X_train_new = pd.DataFrame(X_train_poly, columns=feature_names, index=X_train.index)
            X_test_new = pd.DataFrame(X_test_poly, columns=feature_names, index=X_test.index)
            for col in X_train.select_dtypes(exclude=[np.number]).columns:
                X_train_new[col] = X_train[col].values
                X_test_new[col] = X_test[col].values
            if X_train_new.shape[1] > 2*X_train.shape[1]:
                st.warning("2 kat aşıldı, düşük varyanslılar siliniyor")
                var = X_train_new.var()
                low_var = var.nsmallest(int(X_train_new.shape[1]/2)).index
                X_train_new = X_train_new.drop(columns=low_var)
                X_test_new = X_test_new.drop(columns=low_var)
            st.session_state.X_train, st.session_state.X_test = X_train_new, X_test_new
            st.success(f"Yeni sayı: {X_train_new.shape[1]}")
            st.session_state.step=7; st.rerun()
        else: st.error("Sayısal değişken yok")
    if st.button("⬅️ Geri"): st.session_state.step=5; st.rerun()

# ========================= ADIM 7 ========================= (eksik/aykırı)
elif st.session_state.step == 7:
    st.header("7️⃣ Eksik ve Aykırı Değerler")
    X_train, X_test = st.session_state.X_train, st.session_state.X_test
    missing = X_train.isnull().sum()
    missing = missing[missing>0]
    if len(missing)>0:
        st.subheader("Eksik Değerler")
        st.dataframe(pd.DataFrame({'Değişken':missing.index,'Eksik Sayı':missing.values,'%':(missing/len(X_train)*100).round(2)}))
        method = st.selectbox("Doldurma yöntemi", ["Medyan/Mod","Ortalama","KNN Imputer","Sil"])
        if st.button("Uygula", key="miss"):
            from sklearn.impute import SimpleImputer, KNNImputer
            numeric = X_train.select_dtypes(include=[np.number]).columns
            categorical = X_train.select_dtypes(exclude=[np.number]).columns
            if method=="Sil":
                X_train = X_train.dropna()
                X_test = X_test.dropna()
            else:
                if method=="Medyan/Mod":
                    num_imp = SimpleImputer(strategy='median')
                    cat_imp = SimpleImputer(strategy='most_frequent')
                elif method=="Ortalama":
                    num_imp = SimpleImputer(strategy='mean')
                    cat_imp = SimpleImputer(strategy='most_frequent')
                else:
                    num_imp = KNNImputer(n_neighbors=5)
                    cat_imp = SimpleImputer(strategy='most_frequent')
                if len(numeric)>0:
                    X_train[numeric] = num_imp.fit_transform(X_train[numeric])
                    X_test[numeric] = num_imp.transform(X_test[numeric])
                if len(categorical)>0:
                    X_train[categorical] = cat_imp.fit_transform(X_train[categorical])
                    X_test[categorical] = cat_imp.transform(X_test[categorical])
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.success("Eksikler işlendi"); st.session_state.step=8; st.rerun()
    else:
        st.success("Eksik yok")
        if st.button("Devam ➡️"): st.session_state.step=8; st.rerun()
    # Aykırı opsiyonel
    with st.expander("Aykırı Değer (Opsiyonel)"):
        numeric = X_train.select_dtypes(include=[np.number]).columns
        if len(numeric)>0:
            col = st.selectbox("Değişken", numeric)
            fig,ax = plt.subplots()
            X_train.boxplot(column=col, ax=ax); st.pyplot(fig)
            method = st.radio("Strateji",["Hiçbir şey","Winsorizing","Sil (IQR)"])
            if method!="Hiçbir şey":
                Q1,Q3 = X_train[col].quantile(0.25), X_train[col].quantile(0.75)
                IQR = Q3-Q1
                lower, upper = Q1-1.5*IQR, Q3+1.5*IQR
                if method=="Winsorizing":
                    X_train[col] = X_train[col].clip(lower,upper)
                    X_test[col] = X_test[col].clip(lower,upper)
                else:
                    mask = (X_train[col]>=lower)&(X_train[col]<=upper)
                    X_train = X_train[mask]
                    if hasattr(st.session_state,'y_train') and st.session_state.y_train is not None:
                        st.session_state.y_train = st.session_state.y_train[mask]
                st.session_state.X_train, st.session_state.X_test = X_train, X_test
                st.success("Aykırı işlendi")
    if st.button("⬅️ Geri"): st.session_state.step=6; st.rerun()

# ========================= ADIM 8 ========================= (encoding + scaling - ZORUNLU)
elif st.session_state.step == 8:
    st.header("8️⃣ Encoding ve Scaling")
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    
    # Kategorik değişken varsa encoding zorunlu
    cat_cols = X_train.select_dtypes(include=['object']).columns
    if len(cat_cols)>0:
        st.warning(f"⚠️ {len(cat_cols)} kategorik değişken var. Encoding yapmalısınız!")
        method = st.selectbox("Encoding yöntemi", ["One-Hot Encoding", "Label Encoding"])
        if st.button("Encoding Uygula", key="enc"):
            if method == "Label Encoding":
                from sklearn.preprocessing import LabelEncoder
                for col in cat_cols:
                    le = LabelEncoder()
                    X_train[col] = le.fit_transform(X_train[col].astype(str))
                    X_test[col] = le.transform(X_test[col].astype(str))
            else:
                X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
                X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
                # Sütun uyumluluğu
                missing_cols = set(X_train.columns) - set(X_test.columns)
                for col in missing_cols:
                    X_test[col] = 0
                X_test = X_test[X_train.columns]
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.success("Encoding tamamlandı")
            st.rerun()
    else:
        st.success("✅ Kategorik değişken yok, encoding gerekmez.")
    
    # Scaling
    st.subheader("Scaling")
    num_cols = X_train.select_dtypes(include=[np.number]).columns
    if len(num_cols)>0:
        method = st.selectbox("Scaling", ["StandardScaler","MinMaxScaler","RobustScaler"])
        if st.button("Scaling Uygula", key="scale"):
            from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
            scaler = {"StandardScaler":StandardScaler(),"MinMaxScaler":MinMaxScaler(),"RobustScaler":RobustScaler()}[method]
            X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
            X_test[num_cols] = scaler.transform(X_test[num_cols])
            st.session_state.X_train, st.session_state.X_test = X_train, X_test
            st.session_state.scaler = scaler
            st.success("Scaling tamamlandı")
            st.session_state.step = 9
            st.rerun()
    else:
        st.info("Sayısal değişken yok, scaling atlandı")
        if st.button("Scaling'siz Devam Et"): st.session_state.step = 9; st.rerun()
    
    if st.button("⬅️ Geri"): st.session_state.step=7; st.rerun()

# ========================= ADIM 9 ========================= (modelleme, hata yönetimli)
elif st.session_state.step == 9:
    st.header("9️⃣ Model Karşılaştırma")
    problem = st.session_state.problem_type
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    
    # SON GÜVENLİK: hala object var mı?
    if X_train.select_dtypes(include=['object']).shape[1] > 0:
        st.error("❌ Hala kategorik değişken var! Lütfen Adım 8'de encoding yapın.")
        if st.button("Encoding Adımına Dön"): st.session_state.step=8; st.rerun()
        st.stop()
    
    # Model tanımları
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
    elif problem == "Sınıflandırma":
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from xgboost import XGBClassifier
        from sklearn.svm import SVC
        models = {
            "Lojistik Regresyon": LogisticRegression(random_state=42, max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42),
            "SVM": SVC(random_state=42, probability=True)
        }
    else:  # Kümeleme
        from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
        from sklearn.mixture import GaussianMixture
        models = {
            "K-Means": KMeans(n_clusters=3, random_state=42),
            "Agglomerative": AgglomerativeClustering(n_clusters=3),
            "DBSCAN": DBSCAN(),
            "Gaussian Mixture": GaussianMixture(n_components=3, random_state=42)
        }
    
    selected = st.multiselect("Modelleri seç", list(models.keys()), default=list(models.keys()))
    if st.button("Modelleri Çalıştır", key="run"):
        results = []
        for name in selected:
            try:
                model = models[name]
                if problem != "Kümeleme":
                    model.fit(X_train, st.session_state.y_train)
                    y_pred = model.predict(X_test)
                    if problem == "Regresyon":
                        from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
                        r2 = r2_score(st.session_state.y_test, y_pred)
                        rmse = np.sqrt(mean_squared_error(st.session_state.y_test, y_pred))
                        mae = mean_absolute_error(st.session_state.y_test, y_pred)
                        results.append({"Model":name, "R2":r2, "RMSE":rmse, "MAE":mae})
                        fig,ax = plt.subplots()
                        ax.scatter(st.session_state.y_test, y_pred, alpha=0.5)
                        ax.plot([st.session_state.y_test.min(), st.session_state.y_test.max()],
                                [st.session_state.y_test.min(), st.session_state.y_test.max()], 'r--')
                        st.pyplot(fig)
                    else: # Sınıflandırma
                        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
                        acc = accuracy_score(st.session_state.y_test, y_pred)
                        f1 = f1_score(st.session_state.y_test, y_pred, average='weighted')
                        auc = 0
                        if hasattr(model, "predict_proba"):
                            try:
                                auc = roc_auc_score(st.session_state.y_test, model.predict_proba(X_test)[:,1])
                            except: pass
                        results.append({"Model":name, "Accuracy":acc, "F1":f1, "AUC":auc})
                else:
                    clusters = model.fit_predict(X_train)
                    if hasattr(model, "labels_"): clusters = model.labels_
                    from sklearn.metrics import silhouette_score, davies_bouldin_score
                    if len(set(clusters))>1:
                        sil = silhouette_score(X_train, clusters)
                        db = davies_bouldin_score(X_train, clusters)
                    else: sil,db = -1,-1
                    results.append({"Model":name, "Silhouette":sil, "Davies-Bouldin":db})
            except Exception as e:
                st.error(f"{name} hatası: {str(e)}")
                results.append({"Model":name, "Hata":str(e)})
        if results:
            df_res = pd.DataFrame(results)
            st.dataframe(df_res)
            # İlk hatasız modeli seç
            best = None
            for r in results:
                if "Hata" not in r:
                    best = r["Model"]
                    break
            if best:
                st.session_state.best_model_name = best
                st.session_state.best_model = models[best]
                st.success(f"Seçilen model: {best}")
                if st.button("Hiperparametreye Geç"): st.session_state.step=10; st.rerun()
            else: st.error("Hiçbir model başarılı olmadı")
    if st.button("⬅️ Geri"): st.session_state.step=8; st.rerun()

# ========================= ADIM 10 ========================= (hiperparametre - kısaltılmış)
elif st.session_state.step == 10:
    st.header("🔟 Hiperparametre Optimizasyonu")
    model = st.session_state.best_model
    problem = st.session_state.problem_type
    param_grids = {
        "RandomForestRegressor": {"n_estimators":[50,100]},
        "XGBRegressor": {"n_estimators":[50,100], "learning_rate":[0.01,0.1]},
        "RandomForestClassifier": {"n_estimators":[50,100]},
        "XGBClassifier": {"n_estimators":[50,100], "learning_rate":[0.01,0.1]},
        "SVC": {"C":[0.1,1,10], "kernel":["linear","rbf"]},
        "Ridge": {"alpha":[0.1,1,10]}
    }
    from sklearn.model_selection import GridSearchCV
    model_class = model.__class__.__name__
    param_grid = param_grids.get(model_class, {})
    if param_grid:
        st.write("Aranacak parametreler:", param_grid)
        if st.button("Grid Search Başlat"):
            if problem!="Kümeleme":
                gs = GridSearchCV(model, param_grid, cv=3, scoring='r2' if problem=="Regresyon" else 'accuracy')
                gs.fit(st.session_state.X_train, st.session_state.y_train)
                st.session_state.best_model = gs.best_estimator_
                st.success(f"En iyi: {gs.best_params_}")
                st.session_state.step=11; st.rerun()
            else: st.warning("Kümeleme için manuel optimizasyon gerekir")
    else:
        st.info("Bu model için hiperparametre tanımlı değil, atlanıyor")
        if st.button("Model Yorumlamaya Geç"): st.session_state.step=11; st.rerun()
    if st.button("⬅️ Geri"): st.session_state.step=9; st.rerun()

# ========================= ADIM 11 ========================= (yorumlama)
elif st.session_state.step == 11:
    st.header("1️⃣1️⃣ Model Yorumlama")
    model = st.session_state.best_model
    X_train = st.session_state.X_train
    if st.session_state.problem_type != "Kümeleme":
        if hasattr(model, "feature_importances_"):
            imp = pd.DataFrame({"Feature":X_train.columns, "Importance":model.feature_importances_}).sort_values("Importance",ascending=False).head(10)
            st.bar_chart(imp.set_index("Feature"))
        elif hasattr(model, "coef_"):
            coef = model.coef_.flatten() if len(model.coef_.shape)>1 else model.coef_
            imp = pd.DataFrame({"Feature":X_train.columns, "Coefficient":coef}).sort_values("Coefficient", key=abs, ascending=False).head(10)
            st.dataframe(imp)
        else: st.info("Değişken önemi gösterilemiyor")
        try:
            import shap
            if "tree" in str(type(model)).lower():
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train.iloc[:100])
                fig,ax = plt.subplots()
                shap.summary_plot(shap_values, X_train.iloc[:100], show=False)
                st.pyplot(fig)
        except: pass
        if st.button("Canlı Denemeye Geç"): st.session_state.step=12; st.rerun()
    else:
        clusters = model.fit_predict(X_train)
        X_train_copy = X_train.copy()
        X_train_copy['Cluster'] = clusters
        st.dataframe(X_train_copy.groupby('Cluster').mean())
        if st.button("Canlı Deneme"): st.session_state.step=12; st.rerun()
    if st.button("⬅️ Geri"): st.session_state.step=10; st.rerun()

# ========================= ADIM 12 ========================= (canlı deneme)
elif st.session_state.step == 12:
    st.header("1️⃣2️⃣ Canlı Deneme")
    model = st.session_state.best_model
    X_train = st.session_state.X_train
    feature_names = X_train.columns.tolist()
    selected = st.multiselect("En fazla 5 değişken seçin", feature_names, default=feature_names[:min(5,len(feature_names))])
    if len(selected)>5:
        st.warning("En fazla 5 seçin")
    else:
        user_input = {}
        for feat in selected:
            if X_train[feat].dtype in ['float64','int64']:
                val = st.number_input(f"{feat}", value=float(X_train[feat].mean()))
            else:
                options = X_train[feat].dropna().unique().tolist()
                val = st.selectbox(f"{feat}", options)
            user_input[feat] = val
        if st.button("Tahmin Yap"):
            input_df = pd.DataFrame([user_input])
            for col in feature_names:
                if col not in input_df.columns:
                    if X_train[col].dtype in ['float64','int64']:
                        input_df[col] = X_train[col].mean()
                    else:
                        input_df[col] = X_train[col].mode()[0] if len(X_train[col].mode())>0 else 0
            input_df = input_df[feature_names]
            pred = model.predict(input_df)[0]
            st.success(f"Tahmin: {pred}")
            if st.session_state.problem_type != "Kümeleme":
                y_pred_all = model.predict(st.session_state.X_test)
                fig,ax = plt.subplots()
                ax.hist(y_pred_all, bins=20, alpha=0.7, label="Test tahminleri")
                ax.axvline(pred, color='red', linestyle='--', label="Kullanıcı")
                ax.legend(); st.pyplot(fig)
    if st.button("⬅️ Geri"): st.session_state.step=11; st.rerun()

st.markdown("---")
st.caption("🤖 AutoML Pipeline | Düzeltilmiş ve kararlı sürüm")
