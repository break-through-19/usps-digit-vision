import scipy.io
from sklearn.preprocessing import StandardScaler

def load_data(path):
    mat = scipy.io.loadmat(path)
    X = mat['fea']
    y = mat['gnd'].flatten()
    
    # Split data
    X_train = X[:7291]
    y_train = y[:7291]
    X_test = X[7291:]
    y_test = y[7291:]
    
    # Normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test
