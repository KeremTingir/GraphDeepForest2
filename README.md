# GraphDeepForest2

---

## Ders 1 — Cascade Forest (Makaledeki Mimari)

Bu derste makalenin cascade forest yapısı (2 Random Forest + 2 Extra Trees'ten oluşan katmanlar) incelendi; `X_0`, `P_0`, `X_l`, `P_l` kavramları ve dinamik durdurma mekanizması (validation accuracy 2 katman üst üste iyileşmezse durma) matematiksel olarak açıklanıp küçük sentetik veriyle Colab'da uygulandı.

Amaç, makalenin en soyut ve kritik bileşenini — global/local özelliklerin ve önceki katman olasılıklarının nasıl birleştiğini — somut kodla kavramaktı.

Sonuçta katman derinleştikçe girdi boyutunun nasıl büyüdüğü ve olasılık matrislerinin nasıl üretildiği net bir şekilde görüldü.

Bu derste bilerek basit tutulan implementasyonun (training set üzerinde `predict_proba` çağırma) aslında hatalı olduğu fark edildi ve bu, bir sonraki dersin motivasyonu oldu.

---

## Ders 2 — Out-of-Fold / Validation Probability

Ders 1'deki hatanın (aynı veri üzerinde hem eğitim hem tahmin yapma) **data leakage**'a yol açtığı gösterildi; bunun yerine k-fold cross-validation ile her örneğin kendisini görmemiş bir modelden tahmin alması gereken **OOF (Out-of-Fold) prediction** tekniği öğrenildi.

Amaç, cascade forest'a beslenecek olasılıkların gerçekçi (aşırı iyimser olmayan) olmasını sağlamaktı.

Leakage'lı ve OOF yaklaşımlar aynı veri üzerinde karşılaştırıldığında, leakage'lı yöntemin accuracy'yi yapay şekilde şişirdiği sayısal olarak kanıtlandı.

---

## Ders 3 — EEG'nin Graph Olarak Temsili

EEG kanallarının neden bir graph'ın node'ları olarak düşünülebileceği, edge'lerin ne anlama geldiği, adjacency matrix, weighted/undirected/directed graph kavramları ve fiziksel elektrot yakınlığı ile fonksiyonel bağlanabilirlik arasındaki fark işlendi.

Amaç, hem global branch'in (undirected) hem local branch'in (directed) dayandığı ortak graph teorisi temelini kurmaktı.

Sentetik kanallarla hem undirected hem directed örnek graph'lar NetworkX ile oluşturulup görselleştirildi.

Bu ders, ilerleyen derslerdeki MI (undirected) ve Granger causality (directed) arasındaki yapısal farkın zeminini hazırladı.

---

## Ders 4 — Graph Feature Representation

Node feature, edge feature, graph feature kavramları ile handcrafted (degree, clustering coefficient gibi elle tanımlı) özellikler ve GAT'ın öğrendiği (learned) temsiller arasındaki fark ele alındı.

Amaç, makalenin node feature olarak ham EEG sinyalini kullanıp bunu öğrenilmiş bir embedding'e dönüştürdüğünü — degree/clustering coefficient gibi klasik graph metriklerini kullanmadığını — netleştirmekti.

Ders 3'teki graph üzerinde weighted degree ve clustering coefficient hesaplanıp görselleştirildi.

Sonuç olarak, makalenin yaklaşımı ile klasik graph-teorisi tabanlı özniteliklerin birbirine karıştırılmaması gerektiği vurgulandı.

---

## Ders 5 — Mutual Information

Mutual Information'ın (MI) correlation'dan farkı, entropy tabanlı matematiksel tanımı, Gaussian kernel density estimation ile sürekli veride nasıl hesaplandığı ve Scott bandwidth kuralı işlendi.

Amaç, global brain network'ün (Section 2.2.1) gerçek kuruluş yöntemini öğrenmekti.

Nonlinear ilişkili sentetik sinyallerle, Pearson korelasyonunun kaçırdığı bir bağımlılığı MI'nin yakaladığı gösterildi ve çok kanallı bir MI matrisi oluşturuldu.

Makalenin bandwidth formülündeki (`h=N^{-1/5}\sigma`) tekil `\sigma` ifadesinin çok değişkenli durumda nasıl yorumlanması gerektiği belirsiz bulundu.

---

## Ders 6 — GRU

RNN'lerin vanishing gradient problemi ve GRU'nun reset/update kapılarıyla bu sorunu nasıl hafiflettiği önce teorik olarak, sonra elle hesaplanan örnekle işlendi.

Amaç, global branch'in zamansal bileşenini (GRU) anlamaktı.

İlk versiyonda sadece sayısal çıktılar üretilince yetersiz bulunup ders revize edildi: ani sıçramalı bir test sinyali üzerinde kapı değerlerinin gerçek zamanlı tepkisi, PyTorch `nn.GRU` çıktılarının ısı haritası ve RNN/GRU'nun gradyan sönümünün tüm zaman ekseni boyunca karşılaştırmalı eğrisi eklendi.

Sonuçta GRU'nun kapılarının gerçekten "önemli olayları" hafızasına kodladığı gözle görülür şekilde kanıtlandı.

---

## Ders 7 — GAT

Graph Attention Network'ün GCN'den farkı, attention coefficient hesaplama (skor → softmax → ağırlıklı toplam), neighbor aggregation ve multi-head attention kavramları işlendi.

Amaç, hem global hem local branch'te kullanılan ortak attention mekanizmasını sıfırdan kurmaktı.

Sentetik bir EEG graph'ı üzerinde attention ağırlıkları ısı haritası ve graph üzerinde görselleştirildi, 2 katmanlı bir GAT'ın node temsillerini nasıl birbirine yaklaştırdığı (over-smoothing'in erken belirtisi) gösterildi.

Makalenin multi-head attention'dan açıkça bahsetmediği ve global/local branch'in agregasyon formüllerinin (residual terim farkı) birbirinden ayrıldığı tespit edildi.

---

## Ders 8 — Global Branch'ın Birleştirilmesi

Ders 5–7'de ayrı öğrenilen MI-graph, GRU ve GAT bileşenleri tek bir "Graph GRU Module" içinde birleştirildi ve gerçekten eğitilebilir bir PyTorch modeli olarak kuruldu.

Amaç, EEG'den global feature'a (`\bar h^T`) kadar giden pipeline'ı uçtan uca çalıştırmaktı.

Sentetik iki-sınıflı veri üzerinde model eğitildi, loss düştü ve öğrenilen embedding'ler PCA ile 2 boyuta indirgenip sınıflar arasında görünür bir ayrışma olduğu gösterildi.

Bu sırada makalenin Eş. 3 (zaman boyutunu kaybeden doğrusal projeksiyon) ile Eş. 4-10 (zaman-indeksli işlemler) arasındaki iç tutarsızlık netleştirilerek, dersteki implementasyonun bu boşluğu nasıl doldurduğu açıklandı.

---

## Ders 9 — VMD (Variational Mode Decomposition)

VMD'nin sinyali merkez frekansları farklı, dar bantlı modlara ayırma amacı, K (mod sayısı) ve α (penalty) parametrelerinin rolü, ADMM tabanlı yakınsama ve Pearson korelasyonuyla mod seçimi işlendi.

Amaç, local branch'in ilk adımını (Section 2.3.1) kavramaktı.

`vmdpy` kütüphanesiyle gerçek bir VMD, makalenin K=8/α=2000 parametreleriyle sentetik bir non-stationary sinyale uygulandı; farklı K ve α değerlerinin modları nasıl over/under-decompose ettiği görselleştirildi.

Sonuç olarak makalenin optimal K=8 seçiminin gerçek EEG'nin karmaşıklığına özgü olduğu, bizim küçük sentetik sinyalimiz için çok daha düşük bir K'nın yeterli olduğu anlaşıldı.

---

## Ders 10 — Multi-Grained Scanning

Sinyali farklı pencere boyutlarıyla (τ=100/200/400) tarayan Multi-Grained Scanning (MGS) tekniği, window/stride/overlapping kavramları ve pencere sayısı formülü (`N_\tau`) işlendi.

Amaç, VMD çıktısındaki her modu zaman ekseninde daha küçük, çok ölçekli parçalara bölmekti.

VMD modları üzerinde orantılı küçültülmüş pencere boyutlarıyla (20/40/80) windowing uygulanıp örtüşen pencereler görselleştirildi, kısa/orta/uzun pencerelerin farklı sayıda segment ürettiği gösterildi.

Bu, makalenin neden tek bir pencere boyutu yerine üç farklı granülarite kullandığının somut gerekçesini ortaya koydu.

---

## Ders 11 — Local Feature Extraction (Segment Kombinasyonları)

Bu ders, VMD (Ders 9) ve MGS (Ders 10) çıktılarının birlikte kaç segment ürettiğini (`K' × toplam pencere sayısı`) hesaplayan bir ara/hazırlık dersiydi — henüz Granger causality ya da GAT uygulanmadı.

Amaç, ilerleyen iki dersin (Granger + local GAT) her segment için ayrı ayrı çalışacağını ve bu nedenle segment sayısının hesaplama maliyetini nasıl hızla artırdığını (combinatorial explosion) kavramaktı.

Mod × pencere boyutu kombinasyonları bir sözlük yapısında organize edilip ısı haritasıyla görselleştirildi.

Sonuç, "local feature extraction" teriminin makalede aslında Section 2.3.4'e karşılık geldiği, bugünkü dersin ona bir hazırlık aşaması olduğu netleştirildi.

---

## Ders 12 — Granger Causality

Granger causality'nin correlation'dan farkı ("X'in geçmişi Y'yi açıklıyor mu?"), kısıtlı/tam VAR modelleri, F-testi, lag order seçimi (AIC) ve nedensellik gücü (`GC_{i\to j}`) formülleri işlendi.

Amaç, local brain network'ün (Section 2.3.3) directed graph'ını gerçek istatistiksel testle kurmaktı.

Bilinçli olarak bir kanaldan diğerine gecikmeli nedensellik içeren sentetik 3 kanallı veri üzerinde `statsmodels` ile Granger testi uygulanıp, sadece doğru yöndeki bağlantının (ve tersinin değil) anlamlı çıktığı doğrulandı.

Sonuçta asimetrik bir adjacency matrix ve directed graph elde edildi.

---

## Ders 13 — Local GAT

Local branch'in GAT formülünün (residual terimli, Eş. 27) global branch'inkinden farkı işlendi.

Ders 12'de üretilen directed graph, 2 katmanlı residual'lı bir GAT ile işlenip kanal ortalamasıyla tek bir segment feature'ı (`h`) elde edildi.

Amaç, "ham segment → local embedding" dönüşümünü tam olarak kurmaktı.

Directed graph'ta gelen kenarı olmayan (in-degree=0) bir node'un embedding'inin neredeyse değişmediği, gelen kenarı olan node'un katmanlar boyunca biriktiği gösterildi.

Ardından birden fazla segmentin çıktıları birleştirilerek (Eş. 30) tek bir τ için `H_\tau` vektörü elde edildi.

---

## Ders 14 — Local Branch'ın Tamamı

Ders 9–13'te ayrı öğrenilen VMD, MGS, Granger causality ve local GAT adımları, gerçek çok kanallı sentetik EEG verisi üzerinde tek bir pipeline'da (shape assertion'larla) birleştirildi.

Amaç, local branch'i Ders 8'deki gibi uçtan uca çalışır hale getirmekti.

Süreç boyunca, VMD'nin her kanalda bağımsız çalışması nedeniyle "k-inci mod"un kanallar arasında frekans olarak hizalı olup olmadığının makalede net olmadığı fark edildi ve bu, bilinçli bir basitleştirmeyle (ortak mod indeksi varsayımı) aşıldı.

Sonuçta belirli bir τ için tam bir `H_\tau` vektörü üretildi.

---

## Ders 15 — Global + Local Feature Fusion

Ders 8'in global feature'ı (`\bar h^T`) ile Ders 14'ün local feature'ı (`H_\tau`), makalenin `X_0=[\bar h^T\|H_\tau]` denklemiyle (Eş. 31) birleştirildi.

Amaç, cascade forest'ın gerçek girdisini somut sayılarla kurmaktı.

Bu derste, makalenin `D` boyut formülünün (`|y|` tabanlı) bizim gerçek kodumuzdaki boyutlarla (`d` ve `\tau` tabanlı) örtüşmediği sayısal olarak (78 vs 768 gibi) kanıtlandı — bu, kurs boyunca işaretlenen en somut makale-içi tutarsızlıklardan biri oldu.

Ayrıca `X_0`'ın sadece ilk τ'nun local feature'ını içerdiği, diğerlerinin sonraki katmanlar için saklandığı netleştirildi.

---

## Ders 16 — Full Cascade Forest

Ders 1 (cascade mekanizması), Ders 2 (OOF) ve Ders 15 (`X_0` füzyonu) bir araya getirilerek **gerçek, leakage'sız, τ'ları katmanlar arasında döngüsel olarak değiştiren** tam bir cascade forest kuruldu.

Amaç, Ders 1'deki hatalı (leakage'lı) implementasyonu düzeltilmiş haliyle yeniden inşa etmekti.

Öğrenilebilir sinyal içeren sentetik veri üzerinde katman katman validation accuracy izlendi, dynamic stopping doğru katmanda tetiklendi ve leakage'lı/OOF yaklaşımların accuracy eğrileri karşılaştırılarak aradaki farkın önemi bir kez daha gösterildi.

---

## Ders 17 — Full GDF Architecture

Global branch, local branch ve cascade forest tek bir mimaride şema halinde birleştirildi.

Makalenin FLOPs tablosu (Table 5) formülleri kendi küçük kurulumumuza uygulanıp GRU'nun en maliyetli modül olduğu gösterildi.

Amaç, hangi aşamaların CPU/GPU gerektirdiğini ve hangilerinin computationally expensive olduğunu netleştirmekti.

Ayrıca Ders 7'de başlatılan over-smoothing deneyi resmi olarak tamamlanıp, katman arttıkça node feature'ların birbirine benzeştiği (kosinüs benzerliğinin 1'e yaklaştığı) sayısal olarak doğrulandı.

Bu, makalenin sığ GAT + derin cascade forest tercihinin gerekçesini destekledi.

---

## Ders 18 — Makale Reproduction Planı

DEAP ve AMIGOS veri setlerinin yapısı (kanal sayısı, sinyal uzunluğu, etiketleme), subject-dependent ile subject-independent değerlendirme arasındaki fark ve LOSO (Leave-One-Subject-Out) protokolü işlendi.

Amaç, makaleyi yeniden üretmeden önce sağlam bir deney planı kurmaktı.

Sentetik subject-trial verisiyle standart K-fold'un aynı kişinin verisini hem eğitim hem testte bırakabildiği (subject leakage riski), GroupKFold/LOSO'nun ise bunu önlediği gösterildi.

Makalenin ana tablolarında (Table 1-3) hangi CV protokolünün kullanıldığının açık olmadığı, bunun raporlanan yüksek accuracy'lerin bir kısmını açıklayabileceği belirtildi.

---

## Ders 19 — Ablation Study

Makalenin 5 koşullu ablation deneyi (w/o GRU, w/o Global, w/o Local, w/o Cascade, Full GDF, Fig. 8) ve bunun sonuçları (GRU'nun çıkarılmasının en büyük performans kaybına yol açması) incelendi.

Amaç, her bileşenin katkısını izole edip ölçmekti.

Üç bağımsız "gizli sinyal" (temporal, graph, local) içeren sentetik veri üzerinde benzer bir ablation deneyi kurulup, en yüksek ağırlık verilen bileşenin çıkarılmasının gerçekten en büyük accuracy düşüşüne yol açtığı doğrulandı.

Ayrıca kullanıcının orijinal ders programındaki daha geniş ablation grid'i ile makalenin kendi 5 koşulu arasındaki fark netleştirildi.

---

## Ders 20 — Final Reproduction

Kurs boyunca yazılan kod modüler hale getirildi, her bileşen için shape/sanity testleri eklendi, tüm rastgele süreçler için seed sabitlendi ve tam değerlendirme metrikleri (Accuracy, Precision, Recall, F1, confusion matrix) hesaplanıp sonuçlar JSON dosyasına kaydedildi.

Amaç, kursu profesyonel bir araştırma pratiğiyle kapatmaktı.

En önemli çıktı, 20 ders boyunca tespit edilen **12 maddelik "makale belirsizlikleri"** listesinin tek bir tabloda toplanması oldu.

Bu liste, bir reprodüksiyon denemesinin neden makalenin raporladığı sayılarla birebir örtüşmeyebileceğini sistematik olarak açıklıyor ve gerçek veriyle devam edecek bir sonraki çalışma için yol haritası sunuyor.

---

## Genel Öğrenme Akışı


Cascade Forest
      ↓
OOF / Data Leakage
      ↓
EEG → Graph Representation
      ↓
Graph Features
      ↓
Mutual Information (MI)
      ↓
GRU
      ↓
GAT
      ↓
Global Branch
      ↓
VMD
      ↓
Multi-Grained Scanning (MGS)
      ↓
Local Feature Extraction
      ↓
Granger Causality
      ↓
Local GAT
      ↓
Local Branch
      ↓
Global + Local Feature Fusion
      ↓
Full Cascade Forest
      ↓
Full GDF Architecture
      ↓
Reproduction Plan
      ↓
Ablation Study
      ↓
Final Reproduction
