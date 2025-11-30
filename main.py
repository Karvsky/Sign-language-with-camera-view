import tensorflow as tf
from model import cnns_model        # Importujemy Twój lekki model
from data_import import data_import 

# --- 1. WCZYTANIE DANYCH ---
print("Krok 1: Wczytuję dane...")
# Funkcja data_import zwraca dwa zestawy danych
train_ds, test_ds = data_import()

# --- 2. BUDOWA MODELU ---
print("Krok 2: Buduję model...")
# Tworzymy model (tę lżejszą wersję, o której rozmawialiśmy)
model = cnns_model()

# Wyświetl podsumowanie (zobaczysz ile ma parametrów)
model.summary()

# --- 3. KOMPILACJA (To o co pytałeś) ---
print("Krok 3: Kompilacja...")
# Tutaj mówimy modelowi, jak ma się uczyć (Adam) i jak mierzyć błędy.
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- 4. TRENING (FIT) ---
print("Krok 4: Start treningu...")
# Ustawiam 5 epok, bo na CPU to i tak chwilę potrwa.
# validation_data służy do sprawdzania postępów na bieżąco (na zbiorze testowym).
history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=5
)

# --- 5. EWALUACJA KOŃCOWA ---
print("Krok 5: Sprawdzanie wyników...")
loss, accuracy = model.evaluate(test_ds, verbose=1)
print(f"--------------------------------------")
print(f"Twoja skuteczność (Accuracy): {accuracy * 100:.2f}%")
print(f"Twój błąd (Loss): {loss:.4f}")
print(f"--------------------------------------")

# --- 6. ZAPIS (KLUCZOWE!) ---
print("Krok 6: Zapisywanie modelu...")
# Bez tego nie będziesz mógł użyć modelu z kamerą!
model.save('moj_model_migowy.h5')
print("Gotowe! Plik 'moj_model_migowy.h5' został utworzony.")

