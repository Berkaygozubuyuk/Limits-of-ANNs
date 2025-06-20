import numpy as np # dizi ve matris işlemleri için
import pandas as pd # dataframe işlemleri için
import matplotlib.pyplot as plt #grafik çizimi için
import keras_tuner as kt # hiperparametre optimizasyonu için
import tensorflow as tf # derin öğrenme modelleri ve veri işleme için
from pathlib import Path # dosya yolu işlemleri vb


# işlem yapılacak boyut
grid_size = 25


# iki nokta arasındaki öklid mesafesi
def generate_problem_A(n_samples, seed=None):
    #aynı rastgele değerleri üretmek amacıyla
    if seed is not None:
        np.random.seed(seed) #Aynı seed ile tekrar üretilebilirlik sağlamak için kullanıyoruz
    X = np.zeros((n_samples, grid_size, grid_size), dtype=np.uint8) #girdi
    y = np.zeros(n_samples, dtype=np.float32) #çıktı
    for i in range(n_samples):
        p1 = np.random.randint(0, grid_size, size=2) #birinci nokta
        p2 = np.random.randint(0, grid_size, size=2)# ikinci nokta
        while np.array_equal(p1, p2): #aynı nokta seçilmesin diye kontrol
            p2 = np.random.randint(0, grid_size, size=2)
        X[i, p1[0], p1[1]] = 1 # p1 konumunu işaretliyoruz
        X[i, p2[0], p2[1]] = 1 #p2 konumunu işaretliyoruz
        y[i] = np.linalg.norm(p1 - p2) #öklid mesafesi
    return X, y

# en yakın iki nokta arasındakini hesapla
def generate_problem_B(n_samples, min_pts=3, max_pts=10, seed=None):
    if seed is not None:
        np.random.seed(seed) #yukarıda anlatılan ile aynı şekilde çalışırı
    X = np.zeros((n_samples, grid_size, grid_size), dtype=np.uint8)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        N = np.random.randint(min_pts, max_pts + 1)#her örnek için nokta sayısı
        points = set(); coords = [] #tekrarı önlemek için küme ve nokta kordinatları için liste
        while len(points) < N: #oluşturma işlemleri ve ızgarada işretleme yaparız
            p = tuple(np.random.randint(0, grid_size, size=2))
            if p not in points:
                points.add(p); coords.append(p)
                X[i, p[0], p[1]] = 1
        # tüm çiftler için hesaplama yapıp en kısa olanı seçiyoruz
        dists = [np.linalg.norm(np.array(coords[j]) - np.array(coords[k]))
                 for j in range(N) for k in range(j+1, N)]
        y[i] = np.min(dists)
    return X, y

# b kısmının en uzak olarak değiştirlmiş hali
def generate_problem_C(n_samples, min_pts=3, max_pts=10, seed=None):
    if seed is not None:
        np.random.seed(seed)
    X = np.zeros((n_samples, grid_size, grid_size), dtype=np.uint8)
    y = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        N = np.random.randint(min_pts, max_pts + 1)
        points = set(); coords = []
        while len(points) < N:
            p = tuple(np.random.randint(0, grid_size, size=2))
            if p not in points:
                points.add(p); coords.append(p)
                X[i, p[0], p[1]] = 1
        dists = [np.linalg.norm(np.array(coords[j]) - np.array(coords[k]))
                 for j in range(N) for k in range(j+1, N)]
        y[i] = np.max(dists) # sadece burada min yerine max yazmamız gerekiyor
    return X, y

# kaç nokta olduğunu bulma işlemi
def generate_problem_D(n_samples, min_pts=1, max_pts=10, seed=None):
    if seed is not None:
        np.random.seed(seed) # yine aynı şekilde oluşum
    X = np.zeros((n_samples, grid_size, grid_size), dtype=np.uint8)
    y = np.zeros(n_samples, dtype=np.int32) # tam sayı olacak sonuçta nokta sayıyoruz
    for i in range(n_samples):
        N = np.random.randint(min_pts, max_pts + 1) # 1 ile 10 arası nokta sayısı
        points = set(); count = 0
        while count < N:
            p = tuple(np.random.randint(0, grid_size, size=2))
            if p not in points:
                points.add(p); X[i, p[0], p[1]] = 1; count += 1 # aynı nokta olup olmadığını kontrol ederek işaretliyruz
        y[i] = N # nokta sayısını etiket olarak atıyoruz
    return X, y

# kare sayısını çıktı olarak verme işlemi
def generate_problem_E(n_samples, min_squares=1, max_squares=10, seed=None):
    if seed is not None: # aynı şekilde
        np.random.seed(seed)
    X = np.zeros((n_samples, grid_size, grid_size), dtype=np.uint8) # girdi
    y = np.zeros(n_samples, dtype=np.int32) #çıktı
    for i in range(n_samples):
        N = np.random.randint(min_squares, max_squares + 1) # 1 ile 10 arası kare sayısı
        y[i] = N # kare sayısını ata
        for _ in range(N):
            size = np.random.randint(1, grid_size) # karenin kenar uzunluğu
            x0 = np.random.randint(0, grid_size - size + 1) # üst sol kçşe
            y0 = np.random.randint(0, grid_size - size + 1) #
            X[i, x0:x0+size, y0:y0+size] = 1 # kareyi doldur
    return X, y

# TensorFlow data set formatına dönüştürme
def get_dataset(generate_fn, total_samples=1000, train_samples=800, batch_size=32, seed=None):
    # üretilirken seed keyword ile verilmeli
    X, y = generate_fn(total_samples, seed=seed) #veri üretme
    #boyut ekliyoruz
    X = X[..., np.newaxis].astype(np.float32)
    #aynı şekilde kanal boyutu ekliyoruz
    y = y.astype(np.float32)#etiketleri float yapıyoruz
    ds = tf.data.Dataset.from_tensor_slices((X, y)) #numpy dizilerinden tf dataset oluşturma
    ds = ds.shuffle(train_samples, seed=seed)# örnek sayısı kadar buffer ve aynı seed ile
    #eğitim seti yani ilk train samples örneğini alıyoruz
    train_ds = ds.take(train_samples).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    # test seti yani geri kalan örnekleri alıyoruz
    test_ds  = ds.skip(train_samples).batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds

# Model yapıları yani problemler için farklı mimariler

def build_model_A():
    inputs = tf.keras.Input((grid_size, grid_size, 1)) # girdi şekli 25*26*1
    x = tf.keras.layers.Flatten()(inputs) # vektöre çevir
    x = tf.keras.layers.Dense(64, activation='relu')(x) # 64 nöronlu gizli katman
    x = tf.keras.layers.Dense(32, activation='relu')(x)# 32 nöronlu gizli katman
    outputs = tf.keras.layers.Dense(1)(x) # çıkış tek değer regresyon
    model = tf.keras.Model(inputs, outputs) # modelin tanımı
    model.compile('adam', 'mse', metrics=['mae']) # derleme işlemi
    return model


def build_model_B():
    inputs = tf.keras.Input((grid_size, grid_size, 1)) #girdi ölçütleri
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs) #3*3 filtre 32 kanal
    x = tf.keras.layers.MaxPool2D()(x) #2*2 max pooling sınırı
    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)# 3×3 filtre 64 kanal
    x = tf.keras.layers.GlobalAveragePooling2D()(x)# Özelliklerin uzaysal ortalamasını al
    outputs = tf.keras.layers.Dense(1)(x) #çıktı
    model = tf.keras.Model(inputs, outputs)#model
    model.compile('adam', 'mse', metrics=['mae'])#derleme
    return model


def build_model_C():
    inputs = tf.keras.Input((grid_size, grid_size, 1)) #girdi
    x = tf.keras.layers.Reshape((grid_size, grid_size))(inputs) #(25,25) dizi
    x = tf.keras.layers.SimpleRNN(64, return_sequences=True)(x)# 64 birimli rnn ara çıktı döndür
    x = tf.keras.layers.SimpleRNN(32)(x)# 32 birimli rnn son çıkışı al
    outputs = tf.keras.layers.Dense(1)(x) # çıktı
    model = tf.keras.Model(inputs, outputs)#model
    model.compile('adam', 'mse', metrics=['mae'])#derleme
    return model


def build_model_D():
    inputs = tf.keras.Input((grid_size, grid_size, 1)) #girdi
    x = tf.keras.layers.Reshape((grid_size*grid_size, 1))(inputs) #(625,1) dizi
    x = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=16)(x, x) #kendi kendine dikkat etmesi lazım
    x = tf.keras.layers.GlobalAveragePooling1D()(x) #zaman boyutunda ortalama alma
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile('adam', 'mse', metrics=['mae'])
    return model


def build_model_E():
    inputs = tf.keras.Input((grid_size, grid_size, 1))
    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(inputs) #basic cnn
    x = tf.keras.layers.Flatten()(x)#düzleştirme
    x = tf.keras.layers.Dense(64, activation='relu')(x) #64 nöronlu katmanımız
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile('adam', 'mse', metrics=['mae'])
    return model

# Deney fonksiyonu ve farklı eğitim oranları ve hiperparametre optimizasyonu
def run_experiment(problem_name, generate_fn, build_model_fn):
    print(f"--- {problem_name} ---") # başlığımız
    # temel deneylerimiz %25, %50, %100 eğitim
    X_all, y_all = generate_fn(1000, seed=42)
    for frac in [0.25, 0.5, 1.0]:
        n_train = int(800 * frac) #eğitim örneği sayımız
        model = build_model_fn() # yeni model oluşturuyoruz
        #ilk olarak eğitim dataseti
        train_ds = get_dataset(generate_fn, total_samples=800, train_samples=n_train, seed=42, batch_size=32)[0]
        #sonra test data seti
        test_ds  = get_dataset(generate_fn, total_samples=1000, train_samples=800, seed=42, batch_size=32)[1]
        model.fit(train_ds, validation_data=test_ds, epochs=10, verbose=0)#eğitim yapıyoruz
        loss, mae = model.evaluate(test_ds, verbose=0) # ve testlerin sonucunu alıyoruz
        print(f"{int(frac*100)}% Eğitim  MSE: {loss:.4f}, MAE: {mae:.4f}") # çıktı oluşyuruyoruz

    # hiperparametre optimizasyonu yapıyoruz yani keras tuner ile random search
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model_fn(),#yapılacak model
        objective='val_mae', max_trials=10,#amacımız yani doğrulama mae si ve denem sayımız
        executions_per_trial=1, directory='tuner', project_name=problem_name
    )
    # burada ise tüm eğitim setlerimizi ve test setlerimizi hazırlıyoruz
    train_ds_full, test_ds_full = get_dataset(generate_fn, total_samples=1000, train_samples=800, batch_size=32, seed=42)
    tuner.search(train_ds_full, validation_data=test_ds_full, epochs=10, verbose=0)
    #bunların içinde en iyi olan ayaları alıyoruz
    best_hp = tuner.get_best_hyperparameters(1)[0]
    #en iyileri kullanıcıya gösteriyoruz
    print(f"{problem_name} en iyi HP değerleri: {best_hp.values}")
    #en iyi hiperparametre modelini oluştururuz eğitiriz ve değerlendiririz
    best_model = tuner.hypermodel.build(best_hp)
    best_model.fit(train_ds_full, validation_data=test_ds_full, epochs=10, verbose=0)
    loss, mae = best_model.evaluate(test_ds_full, verbose=0)
    print(f"Tuned → MSE: {loss:.4f}, MAE: {mae:.4f}\n")

# Veri setlerini dışarıya kaydetme yaparız
output_dir = Path("./datasets") # hangi klasöre kaydedileceği
output_dir.mkdir(exist_ok=True) # hata kontrol

def export_to_csv(problem_name, generate_fn):
    X, y = generate_fn(1000, seed=42) # 1000 örnek üretiriz
    n_samples = X.shape[0] # örnek sayısı
    X_flat = X.reshape(n_samples, -1)  #25*25 = 625 sütun
    columns = [f'pixel_{i}' for i in range(X_flat.shape[1])]# sütunları isimlendirme
    df = pd.DataFrame(X_flat, columns=columns) #sonrasında dataframe oluşturma
    df['label'] = y# etiket sütunu da ekliyoruz
    csv_path = output_dir / f'problem_{problem_name}.csv' #dosya yolu
    df.to_csv(csv_path, index=False)#katdetme
    print(f"Saved: {csv_path}")#çıktı

# Problem ABCDE için export
export_to_csv('A', generate_problem_A)
export_to_csv('B', generate_problem_B)
export_to_csv('C', generate_problem_C)
export_to_csv('D', generate_problem_D)
export_to_csv('E', generate_problem_E)



if __name__ == '__main__':
    run_experiment('Problem_A', generate_problem_A, build_model_A)
    run_experiment('Problem_B', generate_problem_B, build_model_B)
    run_experiment('Problem_C', generate_problem_C, build_model_C)
    run_experiment('Problem_D', generate_problem_D, build_model_D)
    run_experiment('Problem_E', generate_problem_E, build_model_E)
