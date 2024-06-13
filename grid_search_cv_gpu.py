from cuml import SVC
from cuml.model_selection import train_test_split
from cuml.preprocessing import StandardScaler
from cuml.model_selection import GridSearchCV
from dask.distributed import Client, wait
from dask_cuda import LocalCUDACluster
from dask.utils import parse_bytes
import cudf
import dask_cudf

def main():
    print('setting up Dask cluster...')
    cluster = LocalCUDACluster()
    client = Client(cluster)
    print(client)

    print('reading csv file...')
    # Konversi pandas DataFrame ke cuDF DataFrame
    data_cudf = dask_cudf.read_csv('workspace/cleaned_data_duplicate.csv')

    print('separating features and target...')
    # Misalkan 'data_cudf' adalah DataFrame yang sudah siap
    X = data_cudf.iloc[:, :-1]
    y = data_cudf.iloc[:, -1]

    print('using train_test_split...')
    # Data harus dalam format yang kompatibel dengan cuDF atau NumPy/CuPy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    print('scaling with standart scaler...')
    # Scaling menggunakan cuML
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Pengaturan GridSearch
    params = {
        'C': [0.1, 1, 10, 100],  # Nilai C yang berbeda
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Jenis kernel yang berbeda
        'gamma': ['scale', 'auto', 1, 0.1, 0.01],  # Nilai gamma untuk kernel non-linear
        'degree': [2, 3, 4]  # Derajat untuk polynomial kernel, hanya efektif jika kernel='poly'
    }

    print('creating model...')
    svm_model = SVC()
    print('searching kernel from params...')
    grid = GridSearchCV(svm_model, params)

    print('training model...')
    grid.fit(X_train, y_train)
    print("Best parameters:", grid.best_params_)
    print("Best cross-validation score:", grid.best_score_)

    # Evaluasi
    y_pred = grid.predict(X_test)
    print('the program is done...')
    
if __name__ == '__main__':
    main()