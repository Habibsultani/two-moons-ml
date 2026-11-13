"""Çalıştırılan deneylerden gerçek sayısal rapor hazırlığı üretir."""

import json


def write_summary(results, info, metrics, selected, pipelines, rates, epochs, majority):
    lines = [
        "TWO MOONS — DENEY ÖZETİ", "",
        "1. VERİ KÜMESİ",
        f"Dosya: {info['file']}; örnek: {info['samples']}; özellik: {info['features']}; sütunlar: {info['columns']}.",
        f"Sınıf dağılımı: {info['class_distribution']}; eksikler: {info['missing_values']}.",
        f"Yinelenen satır: {info['duplicate_rows']}; aynı özellik satırı: {info['duplicate_feature_rows']}.",
        "Sayısal aralıklar: " + json.dumps(info["ranges"], ensure_ascii=False),
        "IQR sınırları ve bayrakları: " + json.dumps(info["iqr_outliers"], ensure_ascii=False),
        info["outlier_action"], "SHA256: " + info["sha256"],
        info["original_source"],
        "Genel Two Moons tanımı: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html",
        "Bu bağlantı CSV'nin indirildiği yerin kanıtı değildir; hazır veri üreticisi kullanılmadı.", "",
        "2. ÖN İŞLEME VE DENEY DÜZENİ",
        "Etiketler zaten 0/1; pozitif sınıf 1. NumPy default_rng, seed=42; %80/%20 tabakalı bölme.",
        "Bölme sayıları: " + json.dumps(info["split_summary"], ensure_ascii=False),
        "Dış eğitimden %20 doğrulama ayrılır. Adayların ölçekleri yalnızca iç eğitimden öğrenilir.",
        "Özgün özellikler [x1,x2]; ek deney [x1,x2,x1²,x1*x2,x2²].",
        "Polinomlar ham girdilerden üretilir; sonra eğitim sütun ortalama/std değerleriyle ölçeklenir.",
        "ddof=0; sıfır standart sapma yerine 1. Eksik doldurma, örnek eleme ve sınıf ağırlığı yok.",
        f"Aday α: {rates}; epoch: {epochs}. Her deneyde aynı 9 aday denenir.",
        "Seçim: doğrulama sınıf-1 F1; eşitlikte doğruluk, daha az epoch, küçük α.",
        "Tüm seçimler sonrası dış eğitimin tamamında ölçekler ve modeller yeniden öğrenilir.",
        "Test seçime katılmaz. Dört son yapılandırma aynı test satırları ve sabit 0.5 eşiğiyle değerlendirilir.",
        "Başlangıç θ=0, θ₀=0. Batch Gradient Descent; düzenlileştirme ve erken durdurma yok.", "",
        "3. SEÇİLEN HİPERPARAMETRELER",
        *[f"{name}: {settings}" for name, settings in selected.items()], "",
        "4. ZORUNLU ANA KARŞILAŞTIRMA VE REFERANS",
        metrics.loc[["Doğrusal / Özgün", "Lojistik / Özgün", "Çoğunluk referansı"]].to_string(float_format=lambda x: f"{x:.6f}"),
        f"Çoğunluk etiketi eğitimden seçildi: {majority}; eşitlikte 0 seçilir.",
        "Karmaşıklık matrisi [[TN,FP],[FN,TP]]; satırlar gerçek, sütunlar tahmin.", "",
        "5. EK DENEY — DERECE 2 POLİNOM",
        metrics.loc[["Doğrusal / Polinom", "Lojistik / Polinom"]].to_string(float_format=lambda x: f"{x:.6f}"),
    ]
    for algorithm in ("Doğrusal", "Lojistik"):
        original, polynomial = f"{algorithm} / Özgün", f"{algorithm} / Polinom"
        accuracy_delta = metrics.loc[polynomial, "accuracy"] - metrics.loc[original, "accuracy"]
        f1_delta = metrics.loc[polynomial, "f1"] - metrics.loc[original, "f1"]
        lines.append(f"{algorithm}: polinom−özgün test doğruluk farkı {accuracy_delta:+.6f}, F1 farkı {f1_delta:+.6f}.")
    lines += ["", "6. EĞİTİM KAYBI VE YAKINSAMA"]
    for name, pipeline in pipelines.items():
        history = pipeline["model"].loss_history_
        lines.append(f"{name}: ilk J={history[0]:.8f}, son J={history[-1]:.8f}; "
                     f"son gradyan L2={metrics.loc[name, 'final_gradient_l2']:.8f}.")
    lines += [
        "Kayıplar eğitim kaybıdır; doğrulama kaybı değildir. MSE/2 ve BCE sayısal olarak kıyaslanmaz.",
        "Epoch sayısı doğrulama başarısıyla seçilmiştir; tam optimuma yakınsama garantisi yoktur.", "",
        "7. GEOMETRİ VE SINIRLILIKLAR",
        "Geometri tanığı (yalnızca dış eğitim): " + json.dumps(info["geometry_check"], ensure_ascii=False),
        "Kavisli iki sınıf yapısı tek doğruyla kusursuz ayırmayı zorlaştırır. Saçılım tek başına kanıt değildir.",
        "Karşıt sınıf dışbükey kılıflarında kesişim bulunduysa sıkı doğrusal ayırıcı olamaz.",
        "Bu kontrol doğrusal olmayan bir modelin kusursuz ayrımını veya genellemesini göstermez.",
        "Sigmoid doğrusal değildir fakat özgün özellikli lojistik 0.5 karar sınırı bir doğrudur.",
        "Polinom uzayındaki doğru, özgün (x1,x2) uzayında ikinci dereceden eğri olabilir.",
        "Derece 2, tüm kavisli veya gürültülü sınıf geometrilerini temsil etmek zorunda değildir.",
        "Polinomun kazancı/kaybı yalnızca bu bölmedeki sonuçtur; genelleme kanıtı değildir.",
        "300 örnek ve tek sabit bölme belirsizlik taşır; tekrarlar veya bağımsız veri ile araştırma yapılabilir.",
        "Test sonuçlarını gördükten sonra bu testte yeni ayar seçilmemelidir.",
        "Doğrusal ham çıktılar olasılık değildir; lojistik olasılık kalibrasyonu sınanmadı.",
        "Karar grafikleri gerçek model ızgarası ve eğitim ölçekleriyle hesaplanır; sınır elle çizilmez.",
        "Nokta renkleri gerçek sınıftır; arka plan model tahminidir; üçgenler test örnekleridir.",
        "CSV kaynağı, üretim tohumu ve gürültü düzeyi doğrulanamadı. Veri değiştirilmedi/üretilmedi.",
        "PDF hazırlanmadı; bu dosya ve grafikler PDF için inceleme materyalidir.",
    ]
    (results / "experiment_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
