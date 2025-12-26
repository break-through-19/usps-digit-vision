import scipy.io
from sklearn.preprocessing import StandardScaler

def load_data(path):
    mat = scipy.io.loadmat(path)
    features = mat['fea']
    labels = mat['gnd'].flatten()
    
    # Split data
    # X_train = X[:7291]
    # y_train = y[:7291]
    # X_test = X[7291:]
    # y_test = y[7291:]
    
    # Normalize data
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    return features, labels
