# %% Veri setinin hazirlanmasi ve preprocessing

from keras.callbacks import EarlyStopping, ModelCheckpoint # Sırasıyla ogrenme durdurma ve parametre kaydı
from keras.models import Sequential #Sıralı model
from keras.layers import Dense, Dropout, BatchNormalization  # Bağlı katmanlar

import matplotlib.pyplot as plt # Görselleştirme
import warnings
warnings. filterwarnings ("ignore")
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns 
import numpy as np
from keras.optimizers import Adam
from sklearn.utils import class_weight

data = [
    # 15 Defans (Güçlü fizik, orta hız)
    {"İsim": "Sergio Ramos", "Boy": 184, "Hız": 68, "Fizik": 88, "Mevki": "Defans"},
    {"İsim": "Virgil van Dijk", "Boy": 193, "Hız": 65, "Fizik": 90, "Mevki": "Defans"},
    {"İsim": "Ruben Dias", "Boy": 187, "Hız": 70, "Fizik": 86, "Mevki": "Defans"},
    {"İsim": "Kalidou Koulibaly", "Boy": 187, "Hız": 63, "Fizik": 89, "Mevki": "Defans"},
    {"İsim": "Aymeric Laporte", "Boy": 191, "Hız": 68, "Fizik": 84, "Mevki": "Defans"},
    {"İsim": "Matthijs de Ligt", "Boy": 189, "Hız": 71, "Fizik": 87, "Mevki": "Defans"},
    {"İsim": "Milan Skriniar", "Boy": 188, "Hız": 64, "Fizik": 85, "Mevki": "Defans"},
    {"İsim": "John Stones", "Boy": 188, "Hız": 72, "Fizik": 83, "Mevki": "Defans"},
    {"İsim": "David Alaba", "Boy": 180, "Hız": 71, "Fizik": 80, "Mevki": "Defans"},
    {"İsim": "Marquinhos", "Boy": 183, "Hız": 70, "Fizik": 82, "Mevki": "Defans"},
    {"İsim": "Carles Puyol", "Boy": 178, "Hız": 66, "Fizik": 90, "Mevki": "Defans"},
    {"İsim": "Fabio Cannavaro", "Boy": 175, "Hız": 62, "Fizik": 85, "Mevki": "Defans"},
    {"İsim": "Alessandro Nesta", "Boy": 187, "Hız": 69, "Fizik": 88, "Mevki": "Defans"},
    {"İsim": "Paolo Maldini", "Boy": 186, "Hız": 65, "Fizik": 86, "Mevki": "Defans"},
    {"İsim": "John Terry", "Boy": 187, "Hız": 62, "Fizik": 91, "Mevki": "Defans"},
    {"İsim": "Raphael Varane", "Boy": 191, "Hız": 72, "Fizik": 84, "Mevki": "Defans"},
    {"İsim": "Gerard Pique", "Boy": 194, "Hız": 61, "Fizik": 87, "Mevki": "Defans"},
    {"İsim": "Thiago Silva", "Boy": 183, "Hız": 66, "Fizik": 82, "Mevki": "Defans"},
    {"İsim": "Pepe", "Boy": 188, "Hız": 65, "Fizik": 88, "Mevki": "Defans"},
    {"İsim": "Jules Koundé", "Boy": 180, "Hız": 79, "Fizik": 80, "Mevki": "Defans"},
    {"İsim": "Presnel Kimpembe", "Boy": 189, "Hız": 68, "Fizik": 84, "Mevki": "Defans"},
    {"İsim": "Ronald Araújo", "Boy": 188, "Hız": 74, "Fizik": 86, "Mevki": "Defans"},
    {"İsim": "Ben White", "Boy": 185, "Hız": 71, "Fizik": 80, "Mevki": "Defans"},
    {"İsim": "Eder Militao", "Boy": 186, "Hız": 75, "Fizik": 85, "Mevki": "Defans"},
    {"İsim": "Stefan de Vrij", "Boy": 189, "Hız": 66, "Fizik": 83, "Mevki": "Defans"},
    {"İsim": "Wesley Fofana", "Boy": 190, "Hız": 73, "Fizik": 82, "Mevki": "Defans"},
    {"İsim": "Fikayo Tomori", "Boy": 185, "Hız": 76, "Fizik": 81, "Mevki": "Defans"},
    {"İsim": "Antonio Rudiger", "Boy": 190, "Hız": 78, "Fizik": 86, "Mevki": "Defans"},
    {"İsim": "Niklas Süle", "Boy": 195, "Hız": 61, "Fizik": 90, "Mevki": "Defans"},
    {"İsim": "Joe Gomez", "Boy": 188, "Hız": 76, "Fizik": 82, "Mevki": "Defans"},
    {"İsim": "Caglar Söyüncü", "Boy": 185, "Hız": 70, "Fizik": 83, "Mevki": "Defans"},
    {"İsim": "Dayot Upamecano", "Boy": 186, "Hız": 77, "Fizik": 85, "Mevki": "Defans"},
    {"İsim": "Marc Bartra", "Boy": 184, "Hız": 68, "Fizik": 78, "Mevki": "Defans"},
    {"İsim": "Simon Kjaer", "Boy": 190, "Hız": 64, "Fizik": 84, "Mevki": "Defans"},
    {"İsim": "Samuel Umtiti", "Boy": 182, "Hız": 70, "Fizik": 81, "Mevki": "Defans"},
    

    # 15 Forvet (Yüksek hız, değişken fizik)
    {"İsim": "Kylian Mbappe", "Boy": 178, "Hız": 97, "Fizik": 77, "Mevki": "Forvet"},
    {"İsim": "Erling Haaland", "Boy": 194, "Hız": 89, "Fizik": 90, "Mevki": "Forvet"},
    {"İsim": "Robert Lewandowski", "Boy": 185, "Hız": 78, "Fizik": 85, "Mevki": "Forvet"},
    {"İsim": "Harry Kane", "Boy": 188, "Hız": 87, "Fizik": 83, "Mevki": "Forvet"},
    {"İsim": "Lautaro Martinez", "Boy": 174, "Hız": 84, "Fizik": 76, "Mevki": "Forvet"},
    {"İsim": "Dusan Vlahovic", "Boy": 190, "Hız": 87, "Fizik": 86, "Mevki": "Forvet"},
    {"İsim": "Victor Osimhen", "Boy": 185, "Hız": 88, "Fizik": 84, "Mevki": "Forvet"},
    {"İsim": "Karim Benzema", "Boy": 185, "Hız": 87, "Fizik": 82, "Mevki": "Forvet"},
    {"İsim": "Gabriel Jesus", "Boy": 175, "Hız": 85, "Fizik": 74, "Mevki": "Forvet"},
    {"İsim": "Romelu Lukaku", "Boy": 191, "Hız": 85, "Fizik": 89, "Mevki": "Forvet"},
    {"İsim": "Ronaldo Nazario", "Boy": 183, "Hız": 95, "Fizik": 82, "Mevki": "Forvet"},
    {"İsim": "Thierry Henry", "Boy": 188, "Hız": 92, "Fizik": 85, "Mevki": "Forvet"},
    {"İsim": "Andriy Shevchenko", "Boy": 183, "Hız": 88, "Fizik": 84, "Mevki": "Forvet"},
    {"İsim": "Raul Gonzalez", "Boy": 180, "Hız": 86, "Fizik": 78, "Mevki": "Forvet"},
    {"İsim": "Filippo Inzaghi", "Boy": 178, "Hız": 84, "Fizik": 72, "Mevki": "Forvet"},
    {"İsim": "João Félix", "Boy": 181, "Hız": 85, "Fizik": 74, "Mevki": "Forvet"},
    {"İsim": "Memphis Depay", "Boy": 176, "Hız": 89, "Fizik": 80, "Mevki": "Forvet"},
    {"İsim": "Timo Werner", "Boy": 180, "Hız": 91, "Fizik": 75, "Mevki": "Forvet"},
    {"İsim": "Jamie Vardy", "Boy": 179, "Hız": 89, "Fizik": 76, "Mevki": "Forvet"},
    {"İsim": "Alexandre Lacazette", "Boy": 175, "Hız": 82, "Fizik": 78, "Mevki": "Forvet"},
    {"İsim": "Antoine Griezmann", "Boy": 176, "Hız": 88, "Fizik": 73, "Mevki": "Forvet"},
    {"İsim": "Álvaro Morata", "Boy": 190, "Hız": 88, "Fizik": 79, "Mevki": "Forvet"},
    {"İsim": "Zlatan Ibrahimović", "Boy": 195, "Hız": 80, "Fizik": 90, "Mevki": "Forvet"},
    {"İsim": "Edinson Cavani", "Boy": 184, "Hız": 82, "Fizik": 84, "Mevki": "Forvet"},
    {"İsim": "Pierre-Emerick Aubameyang", "Boy": 187, "Hız": 93, "Fizik": 76, "Mevki": "Forvet"},
    {"İsim": "Roberto Firmino", "Boy": 181, "Hız": 86, "Fizik": 75, "Mevki": "Forvet"},
    {"İsim": "Olivier Giroud", "Boy": 192, "Hız": 82, "Fizik": 85, "Mevki": "Forvet"},
    {"İsim": "Luis Suárez", "Boy": 182, "Hız": 83, "Fizik": 82, "Mevki": "Forvet"},
    {"İsim": "Fernando Torres", "Boy": 186, "Hız": 88, "Fizik": 80, "Mevki": "Forvet"},
    {"İsim": "Carlos Tevez", "Boy": 170, "Hız": 84, "Fizik": 83, "Mevki": "Forvet"},
    {"İsim": "Sadio Mane", "Boy": 175, "Hız": 90, "Fizik": 76, "Mevki": "Forvet"},
    {"İsim": "Raheem Sterling", "Boy": 170, "Hız": 91, "Fizik": 70, "Mevki": "Forvet"},
    {"İsim": "Angel Di Maria", "Boy": 180, "Hız": 88, "Fizik": 72, "Mevki": "Forvet"},
    {"İsim": "Marco Reus", "Boy": 180, "Hız": 85, "Fizik": 75, "Mevki": "Forvet"},
    {"İsim": "Heung-Min Son", "Boy": 183, "Hız": 89, "Fizik": 77, "Mevki": "Forvet"},

    # 15 Orta Saha (Dengeli özellikler)
    {"İsim": "Kevin De Bruyne", "Boy": 181, "Hız": 76, "Fizik": 78, "Mevki": "OrtaSaha"},
    {"İsim": "Luka Modric", "Boy": 172, "Hız": 72, "Fizik": 78, "Mevki": "OrtaSaha"},
    {"İsim": "Toni Kroos", "Boy": 183, "Hız": 65, "Fizik": 76, "Mevki": "OrtaSaha"},
    {"İsim": "Frenkie de Jong", "Boy": 180, "Hız": 78, "Fizik": 75, "Mevki": "OrtaSaha"},
    {"İsim": "Joshua Kimmich", "Boy": 177, "Hız": 74, "Fizik": 73, "Mevki": "OrtaSaha"},
    {"İsim": "N'Golo Kante", "Boy": 168, "Hız": 82, "Fizik": 70, "Mevki": "OrtaSaha"},
    {"İsim": "Casemiro", "Boy": 185, "Hız": 66, "Fizik": 78, "Mevki": "OrtaSaha"},
    {"İsim": "Pedri", "Boy": 174, "Hız": 76, "Fizik": 70, "Mevki": "OrtaSaha"},
    {"İsim": "Bernardo Silva", "Boy": 173, "Hız": 80, "Fizik": 72, "Mevki": "OrtaSaha"},
    {"İsim": "Jude Bellingham", "Boy": 186, "Hız": 75, "Fizik": 77, "Mevki": "OrtaSaha"},
    {"İsim": "Zinedine Zidane", "Boy": 185, "Hız": 82, "Fizik": 76, "Mevki": "OrtaSaha"},
    {"İsim": "Andrea Pirlo", "Boy": 177, "Hız": 75, "Fizik": 70, "Mevki": "OrtaSaha"},
    {"İsim": "Steven Gerrard", "Boy": 183, "Hız": 84, "Fizik": 78, "Mevki": "OrtaSaha"},
    {"İsim": "Paul Scholes", "Boy": 170, "Hız": 78, "Fizik": 76, "Mevki": "OrtaSaha"},
    {"İsim": "Claude Makelele", "Boy": 170, "Hız": 80, "Fizik": 73, "Mevki": "OrtaSaha"},
    {"İsim": "Ilkay Gündogan", "Boy": 180, "Hız": 70, "Fizik": 75, "Mevki": "OrtaSaha"},
    {"İsim": "Bruno Fernandes", "Boy": 179, "Hız": 78, "Fizik": 76, "Mevki": "OrtaSaha"},
    {"İsim": "Declan Rice", "Boy": 185, "Hız": 74, "Fizik": 75, "Mevki": "OrtaSaha"},
    {"İsim": "Rodri", "Boy": 190, "Hız": 68, "Fizik": 83, "Mevki": "OrtaSaha"},
    {"İsim": "Leon Goretzka", "Boy": 189, "Hız": 76, "Fizik": 76, "Mevki": "OrtaSaha"},
    {"İsim": "Mason Mount", "Boy": 178, "Hız": 80, "Fizik": 74, "Mevki": "OrtaSaha"},
    {"İsim": "Eduardo Camavinga", "Boy": 182, "Hız": 78, "Fizik": 77, "Mevki": "OrtaSaha"},
    {"İsim": "Nicolo Barella", "Boy": 175, "Hız": 82, "Fizik": 75, "Mevki": "OrtaSaha"},
    {"İsim": "Martin Ødegaard", "Boy": 178, "Hız": 79, "Fizik": 70, "Mevki": "OrtaSaha"},
    {"İsim": "Thiago Alcantara", "Boy": 174, "Hız": 74, "Fizik": 72, "Mevki": "OrtaSaha"},
    {"İsim": "Christian Eriksen", "Boy": 181, "Hız": 76, "Fizik": 72, "Mevki": "OrtaSaha"},
    {"İsim": "Ivan Rakitic", "Boy": 184, "Hız": 70, "Fizik": 75, "Mevki": "OrtaSaha"},
    {"İsim": "Isco", "Boy": 176, "Hız": 71, "Fizik": 68, "Mevki": "OrtaSaha"},
    {"İsim": "Mesut Özil", "Boy": 180, "Hız": 73, "Fizik": 76, "Mevki": "OrtaSaha"},
    {"İsim": "Arturo Vidal", "Boy": 180, "Hız": 73, "Fizik": 75, "Mevki": "OrtaSaha"},
    {"İsim": "Blaise Matuidi", "Boy": 180, "Hız": 76, "Fizik": 70, "Mevki": "OrtaSaha"},
    {"İsim": "Joan Capdevila", "Boy": 182, "Hız": 68, "Fizik": 77, "Mevki": "OrtaSaha"},
    {"İsim": "Adrien Rabiot", "Boy": 188, "Hız": 75, "Fizik": 72, "Mevki": "OrtaSaha"},
    {"İsim": "Paul Pogba", "Boy": 191, "Hız": 77, "Fizik": 77, "Mevki": "OrtaSaha"},
    {"İsim": "Xabi Alonso", "Boy": 183, "Hız": 65, "Fizik": 72, "Mevki": "OrtaSaha"},


    
    

    # 15 Kaleci (Düşük hız, özel fizik)
    {"İsim": "Thibaut Courtois", "Boy": 199, "Hız": 48, "Fizik": 85, "Mevki": "Kaleci"},
    {"İsim": "Alisson Becker", "Boy": 193, "Hız": 52, "Fizik": 88, "Mevki": "Kaleci"},
    {"İsim": "Ederson Moraes", "Boy": 188, "Hız": 55, "Fizik": 83, "Mevki": "Kaleci"},
    {"İsim": "Jan Oblak", "Boy": 188, "Hız": 50, "Fizik": 86, "Mevki": "Kaleci"},
    {"İsim": "Marc-Andre ter Stegen", "Boy": 187, "Hız": 53, "Fizik": 84, "Mevki": "Kaleci"},
    {"İsim": "Manuel Neuer", "Boy": 193, "Hız": 50, "Fizik": 87, "Mevki": "Kaleci"},
    {"İsim": "Mike Maignan", "Boy": 191, "Hız": 54, "Fizik": 85, "Mevki": "Kaleci"},
    {"İsim": "Gianluigi Donnarumma", "Boy": 196, "Hız": 51, "Fizik": 89, "Mevki": "Kaleci"},
    {"İsim": "Keylor Navas", "Boy": 185, "Hız": 56, "Fizik": 80, "Mevki": "Kaleci"},
    {"İsim": "Emiliano Martinez", "Boy": 195, "Hız": 49, "Fizik": 86, "Mevki": "Kaleci"},
    {"İsim": "Gianluigi Buffon", "Boy": 192, "Hız": 52, "Fizik": 88, "Mevki": "Kaleci"},
    {"İsim": "Iker Casillas", "Boy": 182, "Hız": 50, "Fizik": 85, "Mevki": "Kaleci"},
    {"İsim": "Oliver Kahn", "Boy": 188, "Hız": 59, "Fizik": 92, "Mevki": "Kaleci"},
    {"İsim": "Peter Schmeichel", "Boy": 191, "Hız": 54, "Fizik": 90, "Mevki": "Kaleci"},
    {"İsim": "Dino Zoff", "Boy": 182, "Hız": 51, "Fizik": 84, "Mevki": "Kaleci"},
    {"İsim": "Aaron Ramsdale", "Boy": 188, "Hız": 52, "Fizik": 80, "Mevki": "Kaleci"},
    {"İsim": "David de Gea", "Boy": 192, "Hız": 53, "Fizik": 82, "Mevki": "Kaleci"},
    {"İsim": "Kepa Arrizabalaga", "Boy": 186, "Hız": 51, "Fizik": 79, "Mevki": "Kaleci"},
    {"İsim": "Yassine Bounou", "Boy": 190, "Hız": 54, "Fizik": 84, "Mevki": "Kaleci"},
    {"İsim": "Unai Simon", "Boy": 190, "Hız": 55, "Fizik": 83, "Mevki": "Kaleci"},
    {"İsim": "Alex Meret", "Boy": 190, "Hız": 52, "Fizik": 81, "Mevki": "Kaleci"},
    {"İsim": "Lukáš Hrádecký", "Boy": 192, "Hız": 50, "Fizik": 82, "Mevki": "Kaleci"},
    {"İsim": "Kasper Schmeichel", "Boy": 189, "Hız": 53, "Fizik": 84, "Mevki": "Kaleci"},
    {"İsim": "André Onana", "Boy": 190, "Hız": 56, "Fizik": 83, "Mevki": "Kaleci"},
    {"İsim": "Samir Handanovic", "Boy": 193, "Hız": 48, "Fizik": 86, "Mevki": "Kaleci"},
    {"İsim": "Bernd Leno", "Boy": 190, "Hız": 52, "Fizik": 80, "Mevki": "Kaleci"},
    {"İsim": "Sergio Romero", "Boy": 192, "Hız": 50, "Fizik": 81, "Mevki": "Kaleci"},
    {"İsim": "Pepe Reina", "Boy": 188, "Hız": 49, "Fizik": 80, "Mevki": "Kaleci"},
    {"İsim": "Steve Mandanda", "Boy": 185, "Hız": 51, "Fizik": 78, "Mevki": "Kaleci"},
    {"İsim": "Claudio Bravo", "Boy": 184, "Hız": 54, "Fizik": 77, "Mevki": "Kaleci"},
    {"İsim": "Tim Krul", "Boy": 193, "Hız": 50, "Fizik": 82, "Mevki": "Kaleci"},
    {"İsim": "Hugo Lloris", "Boy": 188, "Hız": 56, "Fizik": 85, "Mevki": "Kaleci"},
    {"İsim": "Fraser Forster", "Boy": 201, "Hız": 45, "Fizik": 88, "Mevki": "Kaleci"},
    {"İsim": "Jordan Pickford", "Boy": 185, "Hız": 55, "Fizik": 79, "Mevki": "Kaleci"},
    {"İsim": "Matt Turner", "Boy": 191, "Hız": 52, "Fizik": 81, "Mevki": "Kaleci"}


]
# Veri hazırlama (Önişleme)
df = pd.DataFrame(data).sample(frac=1, random_state=42).reset_index(drop=True)
# Veriyi pandas ile dataframe ye çevirip, rastgele karıştırıldı

# Özellikler ve etiketler
X = df[['Boy', 'Hız', 'Fizik']]
y = df['Mevki']
# Özellik ve Etiket Ayrımı yapıldı

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)
# Özellikler sayıya dönüştürüldü

# Veri bölme
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sss1.split(X, y_encoded):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y_encoded[train_index], y_encoded[test_index]

sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, val_index in sss2.split(X_train, y_train):
    X_train_final, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_train_final, y_val = y_train[train_index], y_train[val_index]
# Veriler %80 train ve %20 test olarak bölündü Trainin ise %75'i final train - %25 validation olarak bölündü

# Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_final)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
#Özellikler Ölçeklendirildi

# Sınıf ağırlıkları 
unique_classes = np.unique(y_train_final)
class_weights = class_weight.compute_class_weight('balanced',
                                                classes=unique_classes,
                                                y=y_train_final)
class_weights = {i: weight for i, weight in enumerate(class_weights)}
#Sınıf ağırlıkları belirlendi

# Model oluşturma 
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),  
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(len(np.unique(y_train_final)), activation='softmax')

])
#64 Nöronlu 3 katmanlı bir yapı kullanıldı
#Dropout ve BatchNorm ile overfitting'e karşı koruma sağlandı

# Model derleme 
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']  
)
#Model derlemesi yapıldı

# Callback'ler (GÜNCELLENDİ)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', monitor='val_accuracy', save_best_only=True)
]
#Model durdurma ve kaydetme tanımlandı
# Model eğitimi 
try:
    history = model.fit(
        X_train_scaled,
        y_train_final,
        epochs=200,
        batch_size=32,  # 16 -> 32
        validation_data=(X_val_scaled, y_val),
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
except Exception as e:
    print("Hata detayı:", str(e))
    # Veri boyutlarını kontrol et
    print("X_train_scaled shape:", X_train_scaled.shape)
    print("y_train_final shape:", y_train_final.shape)
# Model Eğitimi tamamlandı
# En iyi modeli yükle
model = tf.keras.models.load_model('best_model.keras')

# Model Değerlendirme ve Görselleştirme
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.legend()
plt.show()

# Test seti değerlendirme
test_loss, test_acc = model.evaluate(X_test_scaled, y_test, verbose=0)

print("📊 Final Test Performance 📊")
print(f"✅ Test Accuracy: {test_acc*100:.2f}%")
print(f"❌ Test Loss: {test_loss:.4f}")

# Confusion Matrix (Sadeleştirilmiş Versiyon)
y_pred = model.predict(X_test_scaled)
y_pred_classes = np.argmax(y_pred, axis=1)

plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le.classes_, 
            yticklabels=le.classes_)
plt.title('Confusion Matrix (Test Set)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Sınıf bazlı rapor
print("\n📝 Classification Report:")
print(classification_report(y_test, y_pred_classes, target_names=le.classes_))

# Modeli kaydet
model.save('final_model.keras')
print("\nModel 'final_model.keras' olarak kaydedildi")

model.summary()
def mevki_tahmin_et():
    print("Yeni oyuncu bilgilerini gir:")
    boy = float(input("Boy (cm): "))
    hiz = float(input("Hız [0-100]: "))
    fizik = float(input("Fizik [0-100]: "))

    yeni_oyuncu = [[boy, hiz, fizik]]
    yeni_oyuncu_scaled = scaler.transform(yeni_oyuncu)
    tahmin = model.predict(yeni_oyuncu_scaled)
    mevki = le.inverse_transform([tahmin.argmax()])[0]
    print(f"Tahmin edilen mevki: {mevki}")

# Fonksiyonu çağır
mevki_tahmin_et()
mevki_tahmin_et()
mevki_tahmin_et()
"""[
  {"İsim": "Victor Nelsson", "Boy": 185, "Hız": 65, "Fizik": 85, "Mevki": "Stoper"},
  {"İsim": "Dusan Tadic", "Boy": 181, "Hız": 72, "Fizik": 78, "Mevki": "OfansifOrtaSaha"},
  {"İsim": "Vincent Aboubakar", "Boy": 184, "Hız": 76, "Fizik": 85, "Mevki": "Forvet"},
  {"İsim": "Mauro Icardi", "Boy": 181, "Hız": 85, "Fizik": 82, "Mevki": "Forvet"},
  {"İsim": "Omar Colley", "Boy": 191, "Hız": 72, "Fizik": 84, "Mevki": "Stoper"},
  {"İsim": "Fred", "Boy": 169, "Hız": 78, "Fizik": 75, "Mevki": "OrtaSaha"},
  {"İsim": "Davinson Sánchez", "Boy": 187, "Hız": 75, "Fizik": 85, "Mevki": "Stoper"},
 
]
"""

