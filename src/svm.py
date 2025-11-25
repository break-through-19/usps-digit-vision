import time
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

def train_evaluate_svm(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("SUPPORT VECTOR MACHINE MODEL")
    print("="*50)
    results = []
    
    configs = [
        {'kernel': 'linear', 'decision_function_shape': 'ovr'},
        {'kernel': 'poly', 'decision_function_shape': 'ovr', 'degree': 2}, # Polynomial degree 2
        {'kernel': 'poly', 'decision_function_shape': 'ovr', 'degree': 3}, # Polynomial degree 3
    ]
    
    for config in configs:
        start_time = time.time()
        clf = SVC(**config)
        clf.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        
        results.append({
            'model': 'SVM',
            'config': config,
            'accuracy': acc,
            'confusion_matrix': cm,
            'time': train_time
        })
        print(f"Config: {config}, Accuracy Rate: {acc:.2%}, Training Time: {train_time:.4f}s")
        
        # Plot Confusion Matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.title(f'SVM Confusion Matrix\n{config}')
        
        # Save confusion matrix
        os.makedirs('../results/confusion_matrices', exist_ok=True)
        config_str = "_".join([f"{k}-{v}" for k, v in config.items()])
        plt.savefig(f'../results/confusion_matrices/svm_confusion_matrix_{config_str}.png')
        plt.show()
        
    return results
