# Limits-of-ANNs
Limits of Artificial Neural Networks

---

## Proje Amacı

Bu projede, farklı yapıda 5 problem (A–E) için sentetik veriler oluşturulmuş, yapay sinir ağı modelleri bu verilere uygulanmış ve çeşitli deneylerle modellerin sınırlılıkları incelenmiştir.

---

## 📁 Proje İçeriği

### 1. Veri Seti Üretimi

- Her problem için Python ile özel veri jeneratör fonksiyonları yazılmıştır.
- Her bir problem için:
  - **800 eğitim**, **200 test** örneği oluşturulmuştur.
  - Toplamda **1000 örnek/problem**
- Üretilen veriler `datasets/` klasörüne `problem_A.csv`, ..., `problem_E.csv` şeklinde kaydedilir.

### 2. Hiperparametre Optimizasyonu

- Keras Tuner kullanılarak `RandomSearch` yöntemi ile hiperparametre optimizasyonu yapılmıştır.
- Optimize edilen parametreler:
  - Katman sayısı: 1–3
  - Nöron sayısı: 32–128
  - Aktivasyon fonksiyonu: `relu`, `tanh`
  - Öğrenme oranı: `1e-4` – `1e-2` (log scale)
- En iyi model yeniden eğitilerek test verisi üzerinde değerlendirilmiştir.

### 3. 📈 Eğitim Verisi Boyutunun Etkisi

- Eğitim setinin %25, %50 ve %100’ü ile eğitim yapılarak test performansı karşılaştırılmıştır.
- Kullanılan metrikler:
  - **MSE (Mean Squared Error)**
  - **MAE (Mean Absolute Error)**
- Sonuçlar problem türüne göre anlamlı farklılıklar göstermiştir.

### 4. Görselleştirme

- Farklı eğitim oranlarına ve model yapılarına göre performans değişimleri grafiklerle sunulmuştur.

---

##  Gözlemler ve Sonuçlar

- Eğitim verisinin artırılması, özellikle karmaşık problemler (A ve C) için ciddi iyileşmeler sağlamıştır.
- Daha basit problemler (B ve E), daha az veride de düşük hata ile çözülebilmiştir.
- Hiperparametre optimizasyonu, modelin performansını belirgin şekilde artırmıştır.

---

