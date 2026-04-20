import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Sayfa yapılandırması
st.set_page_config(page_title="Veri Analisti Asistanı", layout="wide", page_icon="📊")

# Başlık
st.title("📊 Veri Analisti Asistanı")
st.markdown("Excel veya CSV dosyanızı yükleyin, gerisini bana bırakın!")

# Sidebar
with st.sidebar:
    st.header("⚙️ Ayarlar")
    st.markdown("---")
    st.info("Bu uygulama, yüklediğiniz veriyi otomatik analiz eder:\n\n- Temel istatistikler\n- Eksik değer analizi\n- Görselleştirmeler\n- Korelasyonlar\n- Aykırı değer tespiti")
    st.markdown("---")
    if st.button("🔄 Sayfayı Sıfırla"):
        st.cache_data.clear()
        st.rerun()

# Dosya yükleme
uploaded_file = st.file_uploader(
    "📂 Excel veya CSV dosyası yükleyin",
    type=['csv', 'xlsx', 'xls'],
    help="Bir veri dosyası seçin"
)

if uploaded_file is not None:
    # Dosyayı oku
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        
        # Başarı mesajı
        st.success(f"✅ Dosya başarıyla yüklendi! {df.shape[0]} satır, {df.shape[1]} sütun")
        
        # Veri önizleme
        with st.expander("🔍 Veri Önizleme (İlk 10 satır)", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
        
        # Temel bilgiler
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("📏 Toplam Satır", df.shape[0])
        col2.metric("📐 Toplam Sütun", df.shape[1])
        col3.metric("🔢 Sayısal Sütun", len(df.select_dtypes(include=[np.number]).columns))
        col4.metric("📝 Kategorik Sütun", len(df.select_dtypes(include=['object']).columns))
        
        # Veri tipleri tablosu
        with st.expander("📋 Veri Tipleri ve Örnekler"):
            dtype_df = pd.DataFrame({
                'Sütun': df.columns,
                'Veri Tipi': df.dtypes.astype(str),
                'Benzersiz Değer Sayısı': df.nunique().values,
                'Eksik Sayısı': df.isnull().sum().values,
                'Eksik %': (df.isnull().sum() / len(df) * 100).round(2).values,
                'Örnek Değerler': [str(df[col].dropna().iloc[:3].tolist()) for col in df.columns]
            })
            st.dataframe(dtype_df, use_container_width=True)
        
        # ========== EKSİK DEĞER ANALİZİ ==========
        st.header("❓ Eksik Değer Analizi")
        missing_cols = df.columns[df.isnull().any()].tolist()
        if missing_cols:
            st.warning(f"⚠️ {len(missing_cols)} sütunda eksik değer var:")
            missing_df = pd.DataFrame({
                'Sütun': missing_cols,
                'Eksik Sayısı': df[missing_cols].isnull().sum().values,
                'Eksik Yüzdesi': (df[missing_cols].isnull().sum() / len(df) * 100).round(2).values
            }).sort_values('Eksik Yüzdesi', ascending=False)
            st.dataframe(missing_df, use_container_width=True)
            
            # Görsel
            fig, ax = plt.subplots(figsize=(10, 4))
            missing_df.set_index('Sütun')['Eksik Yüzdesi'].plot(kind='barh', ax=ax, color='coral')
            ax.set_xlabel('Eksik Yüzdesi (%)')
            ax.set_title('Sütunlara Göre Eksik Değer Oranları')
            st.pyplot(fig)
            
            # Missingno matrisi
            try:
                import missingno as msno
                fig, ax = plt.subplots(figsize=(12, 4))
                msno.matrix(df, ax=ax, sparkline=False)
                st.pyplot(fig)
            except:
                st.info("Missingno kütüphanesi yüklü değil, eksiklik matrisi atlandı.")
        else:
            st.success("✅ Hiç eksik değer yok! Veri seti temiz.")
        
        # ========== TEMEL İSTATİSTİKLER ==========
        st.header("📊 Temel İstatistikler")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            st.subheader("Sayısal Değişkenlerin Özeti")
            st.dataframe(df[numeric_cols].describe().T, use_container_width=True)
        else:
            st.info("Sayısal değişken bulunmamaktadır.")
        
        # ========== DAĞILIM GRAFİKLERİ ==========
        st.header("📈 Dağılım Grafikleri")
        if len(numeric_cols) > 0:
            selected_num = st.selectbox("Sayısal değişken seçin", numeric_cols)
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 4))
                df[selected_num].hist(bins=30, edgecolor='black', ax=ax)
                ax.set_title(f'{selected_num} - Histogram')
                ax.set_xlabel(selected_num)
                st.pyplot(fig)
            with col2:
                fig, ax = plt.subplots(figsize=(8, 4))
                df.boxplot(column=selected_num, ax=ax)
                ax.set_title(f'{selected_num} - Box Plot')
                st.pyplot(fig)
            
            # Normallik testi
            if len(df[selected_num].dropna()) >= 3:
                stat, p = stats.shapiro(df[selected_num].dropna()[:5000])
                if p > 0.05:
                    st.success(f"📊 Shapiro-Wilk testi: p = {p:.4f} → Normal dağılıma uygun (p>0.05)")
                else:
                    st.warning(f"⚠️ Shapiro-Wilk testi: p = {p:.4f} → Normal dağılımdan sapma var (p<0.05)")
        else:
            st.info("Sayısal değişken olmadığından dağılım grafikleri gösterilemez.")
        
        # ========== KATEGORİK DEĞİŞKEN ANALİZİ ==========
        cat_cols = df.select_dtypes(include=['object']).columns
        if len(cat_cols) > 0:
            st.header("📊 Kategorik Değişken Analizi")
            selected_cat = st.selectbox("Kategorik değişken seçin", cat_cols)
            value_counts = df[selected_cat].value_counts().head(10)
            fig, ax = plt.subplots(figsize=(10, 5))
            value_counts.plot(kind='bar', ax=ax, color='skyblue')
            ax.set_title(f'{selected_cat} - En Sık 10 Değer')
            ax.set_xlabel(selected_cat)
            ax.set_ylabel('Frekans')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            
            # Pie chart (opsiyonel, eğer az sınıf varsa)
            if len(value_counts) <= 10:
                fig2, ax2 = plt.subplots(figsize=(8, 6))
                value_counts.plot(kind='pie', autopct='%1.1f%%', ax=ax2)
                ax2.set_ylabel('')
                ax2.set_title(f'{selected_cat} - Dağılım')
                st.pyplot(fig2)
        else:
            st.info("Kategorik değişken bulunmamaktadır.")
        
        # ========== KORELASYON ANALİZİ ==========
        if len(numeric_cols) > 1:
            st.header("🔥 Korelasyon Analizi")
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(12, 10))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax, square=True)
            ax.set_title('Pearson Korelasyon Matrisi')
            st.pyplot(fig)
            
            # En yüksek korelasyonlar
            corr_pairs = corr.unstack().sort_values(ascending=False)
            corr_pairs = corr_pairs[corr_pairs < 1]  # kendisiyle korelasyonu çıkar
            st.subheader("En Yüksek Korelasyonlar (|r| > 0.5)")
            high_corr = corr_pairs[abs(corr_pairs) > 0.5].head(10)
            if len(high_corr) > 0:
                st.dataframe(pd.DataFrame(high_corr, columns=['Korelasyon']).reset_index().rename(columns={'level_0':'Değişken 1', 'level_1':'Değişken 2'}))
            else:
                st.info("0.5'in üzerinde korelasyon yok.")
        else:
            st.info("Korelasyon için en az 2 sayısal değişken gereklidir.")
        
        # ========== AYKIRI DEĞER TESPİTİ ==========
        if len(numeric_cols) > 0:
            st.header("🔍 Aykırı Değer Tespiti (IQR Yöntemi)")
            outlier_summary = []
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                outlier_summary.append({
                    'Değişken': col,
                    'Alt Sınır': lower,
                    'Üst Sınır': upper,
                    'Aykırı Sayısı': len(outliers),
                    'Aykırı Oranı (%)': (len(outliers) / len(df) * 100).round(2)
                })
            outlier_df = pd.DataFrame(outlier_summary)
            st.dataframe(outlier_df, use_container_width=True)
            
            # Aykırı değer görselleştirme (seçimli)
            selected_outlier = st.selectbox("Aykırı değerleri görmek için değişken seçin", numeric_cols)
            Q1 = df[selected_outlier].quantile(0.25)
            Q3 = df[selected_outlier].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df[(df[selected_outlier] < lower) | (df[selected_outlier] > upper)]
            if len(outliers) > 0:
                st.write(f"**{selected_outlier}** için tespit edilen aykırı değerler:")
                st.dataframe(outliers[[selected_outlier]], use_container_width=True)
            else:
                st.info("Bu değişkende aykırı değer yok.")
        
        # ========== ZAMAN SERİSİ (eğer tarih varsa) ==========
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) == 0:
            # Otomatik tarih sütunu bulmaya çalış
            for col in df.columns:
                try:
                    pd.to_datetime(df[col])
                    date_cols = [col]
                    break
                except:
                    pass
        if len(date_cols) > 0:
            st.header("📅 Zaman Serisi Analizi")
            date_col = date_cols[0]
            st.success(f"Tarih sütunu olarak '{date_col}' algılandı.")
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
            # Trend grafiği (ilk sayısal değişken)
            num_for_time = df.select_dtypes(include=[np.number]).columns
            if len(num_for_time) > 0:
                selected_time = st.selectbox("Trendini görmek istediğiniz değişken", num_for_time)
                fig, ax = plt.subplots(figsize=(12, 5))
                df[selected_time].plot(ax=ax)
                ax.set_title(f'{selected_time} Zaman Trendi')
                ax.set_xlabel('Tarih')
                ax.set_ylabel(selected_time)
                st.pyplot(fig)
            df.reset_index(inplace=True)
        
        # ========== RAPOR İNDİRME ==========
        st.header("📥 Rapor İndir")
        st.info("Tüm analiz sonuçlarını içeren bir Excel dosyası oluşturmak için butona tıklayın.")
        
        def generate_report():
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Veri özeti
                df.head(100).to_excel(writer, sheet_name='Veri Önizleme', index=False)
                # Temel istatistikler
                if len(numeric_cols) > 0:
                    df[numeric_cols].describe().to_excel(writer, sheet_name='Temel İstatistikler')
                # Eksik değerler
                if missing_cols:
                    missing_df.to_excel(writer, sheet_name='Eksik Değerler', index=False)
                # Korelasyon
                if len(numeric_cols) > 1:
                    corr.to_excel(writer, sheet_name='Korelasyon Matrisi')
                # Aykırı değer özeti
                outlier_df.to_excel(writer, sheet_name='Aykırı Değer Özeti', index=False)
                # Veri tipleri
                dtype_df.to_excel(writer, sheet_name='Veri Tipleri', index=False)
            return output.getvalue()
        
        if st.button("📎 Raporu İndir (Excel)"):
            report_data = generate_report()
            st.download_button(
                label="📥 İndir",
                data=report_data,
                file_name="veri_analiz_raporu.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        # Footer
        st.markdown("---")
        st.caption("📊 Veri Analisti Asistanı | Yüklediğiniz veriyi keşfedin, analiz edin, raporlayın.")
        
    except Exception as e:
        st.error(f"Bir hata oluştu: {str(e)}")
        st.info("Lütfen dosyanızın geçerli bir Excel/CSV olduğundan emin olun.")
else:
    st.info("👈 Lütfen sol menüden bir Excel veya CSV dosyası yükleyin.")
