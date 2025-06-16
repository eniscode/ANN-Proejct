# 🧠 ANN Project - Futbolcu Mevki Tahmini

**Yapay Sinir Ağı Kullanarak Futbolcunun Fiziksel Özelliklerine Göre Mevki Tahmini**

---

## 📌 Proje Hakkında

Günümüz spor dünyasında **veri analitiği** ve **yapay zekâ**, takımların ve oyuncuların performans analizinde kritik rol oynamaktadır. Bu projede, futbolcuların fiziksel özelliklerine (Boy, Hız, Fizik) dayanarak **hangi mevkide (Forvet, Orta Saha, Defans, Kaleci)** oynayabileceklerini tahmin eden bir **yapay sinir ağı (ANN)** modeli geliştirilmiştir.

---

## 🎯 Proje Amacı

Amaç, oyuncuların oyun tarzı ve fiziksel becerilerine göre pozisyonlarını tahmin edebilecek bir model oluşturmaktır. Bu model <img width="444" alt="Ekran Resmi 2025-06-16 20 07 43" src="https://github.com/user-attachments/assets/854067f4-4e65-4fe0-bd9c-afeb6ead679f" />
sayesinde, gözlem gerektirmeden, sadece veriye dayalı bir pozisyon analizi yapılabilir.

---

## 🧠 Kullanılan Yöntemler

- **Veri ön işleme:** Label Encoding ve StandardScaler ile ölçekleme
- **Model türü:** Sequential yapıda ANN
- **Aktivasyon fonksiyonları:** ReLU (gizli katmanlar), Softmax (çıkış)
- **Overfitting önlemleri:** Batch Normalization, Dropout
- **Eğitim değerlendirme:** Accuracy & Loss takibi, Confusion Matrix, F1-Score

---

## 📊 Model Başarımı

Model eğitildikten sonra yüksek doğruluk oranları elde edilmiştir. Overfitting gözlemlenmemiş, eğitim ve doğrulama başarıları birbirine yakın seyretmiştir.

- ✅ **Test Doğruluğu (Accuracy):** %92.86  
- ❌ **Test Kayıp (Loss):** 0.3515

---

## 📈 Görseller

Aşağıdaki grafiklerde veri dengesi ve eğitim sürecine ait doğruluk-kayıp grafikleri yer almaktadır:

### 🔹 Sınıf Dağılımı (Veri Dengesi)

![Tablo](https://github.com/user-attachments/assets/50dd488d-3005-44cf-aa92-e3337b2a36bb)

### 🔹 Eğitim Süreci (Accuracy / Loss)

![Grafik](https://github.com/user-attachments/assets/7d925f76-3f15-4d81-9df9-802e43ec947a)
### 🔹 Sonuçlar Ve Değerler
<img width="465" alt="Ekran Resmi 2025-06-16 20 16 45" src="https://github.com/user-attachments/assets/e3f1c614-620c-4d80-b0a0-9825bd224ce7" />

![Uploading Ekran Resmi 2025-06-16 20.07.43.png…]()

---






