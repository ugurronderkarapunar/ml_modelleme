# 🤖 AutoML Pipeline - Otomatik Veri Bilimi Uygulaması

Streamlit tabanlı, adım adım yönlendirmeli otomatik makine öğrenmesi uygulaması. Kullanıcılar veri yükleyerek, problem tipini seçerek ve interaktif arayüz üzerinden ilerleyerek makine öğrenmesi modellerini eğitebilir.

## 🎯 Özellikler

### ✅ Tamamlanan (MVP - v0.1)
- **Adım 1**: Veri yükleme (CSV/Excel) ve problem tipi seçimi (Regresyon/Sınıflandırma/Kümeleme)
- **Adım 2**: Kapsamlı EDA (Temel istatistikler, eksik değer analizi, korelasyon, dağılım grafikleri)
- **Adım 3**: Değişken tipi belirleme (otomatik öneri + manuel düzenleme)
- **Adım 4**: Değişken sınıflandırma (sayısal, kategorik, yüksek kardinalite, binary)
- **Adım 5**: Veri bölme ve hedef değişken seçimi

### 🚧 Devam Eden (Roadmap)
- **Adım 6**: Feature Engineering (polynomial, interaction, domain-specific)
- **Adım 7**: Aykırı değer ve eksik değer yönetimi
- **Adım 8**: Encoding ve Scaling (pipeline ile)
- **Adım 9**: Model karşılaştırma (varsayılan parametreler)
- **Adım 10**: Hiperparametre optimizasyonu
- **Adım 11**: Model yorumlama (SHAP, feature importance)
- **Adım 12**: Real-time prediction arayüzü

## 📦 Kurulum

### 1. Repoyu klonlayın
```bash
git clone https://github.com/[kullanici-adi]/automl-pipeline.git
cd automl-pipeline
```

### 2. Virtual environment oluşturun (önerilen)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Gereksinimleri yükleyin
```bash
pip install -r requirements.txt
```

### 4. Uygulamayı çalıştırın
```bash
streamlit run app.py
```

## 🎮 Kullanım

1. **Veri Yükleme**: CSV veya Excel dosyanızı yükleyin
2. **Problem Tipi**: Regresyon, Sınıflandırma veya Kümeleme seçin
3. **EDA**: Otomatik oluşturulan analizleri inceleyin
4. **Değişken Tipleri**: Otomatik önerileri kontrol edin ve düzenleyin
5. **Veri Bölme**: Hedef değişkeni ve test oranını belirleyin
6. **İlerleyen Adımlar**: (Yakında eklenecek...)

## 🛠️ Teknolojiler

- **Frontend**: Streamlit
- **Veri İşleme**: Pandas, NumPy
- **Görselleştirme**: Matplotlib, Seaborn, Plotly
- **ML**: Scikit-learn, XGBoost
- **İstatistik**: SciPy, Statsmodels

## 📊 Veri Gereksinimleri

- Format: CSV veya Excel (.xlsx, .xls)
- Minimum satır sayısı: 50 (önerilen: 1000+)
- Hedef değişken: Regresyon/Sınıflandırma için gerekli
- Eksik değer: Desteklenir (otomatik yönetim)

## 🔒 Data Leakage Koruması

Bu uygulama, data leakage'i önlemek için şu prensipleri uygular:

1. **Train-Test Split İlk Adım**: Veri bölme işlemi pipeline'ın en başında yapılır
2. **Fit-Transform Mantığı**: Tüm dönüşümler (scaler, encoder) train setine fit edilir, test setine transform uygulanır
3. **Pipeline Kullanımı**: Sklearn pipeline ile tüm adımlar izole edilir

## 📝 Geliştirme Planı

### Faz 1: MVP (Tamamlandı ✅)
- Adım 1-5 implementasyonu
- Temel EDA ve veri hazırlığı

### Faz 2: Core ML (Devam Ediyor 🚧)
- Feature engineering
- Preprocessing (outlier, missing, encoding, scaling)
- Temel modelleme

### Faz 3: Advanced (Planlanan 📅)
- Hiperparametre optimizasyonu
- Model yorumlama
- Real-time prediction

## 🤝 Katkıda Bulunma

Pull request'ler memnuniyetle karşılanır! Büyük değişiklikler için önce issue açarak tartışalım.

## 📄 Lisans

MIT License

## 👤 Geliştirici

[İsminiz] - Data Analyst / ML Engineer
- LinkedIn: [profil-linki]
- GitHub: [github-username]
- Portfolio: [website]

## 📧 İletişim

Sorularınız için: [email@example.com]

---

⭐ Projeyi beğendiyseniz yıldız vermeyi unutmayın!
