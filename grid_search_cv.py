from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

data = pd.read_csv('outputs2/cleaned_data_duplicate.csv')

# Misalkan 'data' adalah DataFrame yang sudah siap
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Membagi data menjadi train dan test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Standardisasi fitur
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Pengaturan parameter untuk GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Nilai C yang berbeda
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Jenis kernel yang berbeda
    'gamma': ['scale', 'auto', 1, 0.1, 0.01],  # Nilai gamma untuk kernel non-linear
    'degree': [2, 3, 4]  # Derajat untuk polynomial kernel, hanya efektif jika kernel='poly'
}

# Membuat SVM Classifier dan GridSearchCV objek
model = SVC()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Menampilkan parameter terbaik dan skor
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: ", grid_search.best_score_)

# Prediksi pada test set dan evaluasi
y_pred = grid_search.predict(X_test)
print(classification_report(y_test, y_pred))
