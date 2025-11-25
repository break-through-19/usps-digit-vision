import time
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def train_evaluate_knn(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("K-NEAREST NEIGHBORS MODEL")
    print("="*50)
    results = []
    
    # 2.2.1 Different values of k
    k_values = [1, 3, 5, 7]
    
    for k in k_values:
        start_time = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn.fit(X_train, y_train)
        # Training time for KNN is 0 technically, but let's include fit time
        train_time = time.time() - start_time
        
        # Prediction time is the heavy part
        pred_start = time.time()
        y_pred = knn.predict(X_test)
        pred_time = time.time() - pred_start
        
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'model': 'KNN',
            'config': {'k': k, 'weights': 'distance'},
            'accuracy': acc,
            'confusion_matrix': cm,
            'time': train_time + pred_time # Total time
        })
        print(f"k={k}, weights='distance', Accuracy Rate: {acc:.2%}, Training Time: {train_time + pred_time:.4f}s")
        
        # Plot Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'KNN Confusion Matrix (k={k}, distance)')
        
        # Save confusion matrix
        os.makedirs('../results/confusion_matrices', exist_ok=True)
        plt.savefig(f'../results/confusion_matrices/knn_confusion_matrix_k{k}_distance.png')
        plt.show()

    # 2.2.2 Distance weighting
    start_time = time.time()
    knn_dist = KNeighborsClassifier(n_neighbors=3, weights='uniform')
    knn_dist.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    pred_start = time.time()
    y_pred = knn_dist.predict(X_test)
    pred_time = time.time() - pred_start
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    results.append({
        'model': 'KNN',
        'config': {'k': 3, 'weights': 'uniform'},
        'accuracy': acc,
        'confusion_matrix': cm,
        'time': train_time + pred_time
    })
    print(f"k=3, weights='uniform', Accuracy Rate: {acc:.2%}, Training Time: {train_time + pred_time:.4f}s")
    
    # Plot Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title(f'KNN Confusion Matrix (k=3, uniform)')
    
    # Save confusion matrix
    os.makedirs('../results/confusion_matrices', exist_ok=True)
    plt.savefig(f'../results/confusion_matrices/knn_confusion_matrix_k3_uniform.png')
    plt.show()
    
    return results
