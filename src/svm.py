import time
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def train_evaluate_svm(features, labels):
    print("\n" + "="*50)
    print("SUPPORT VECTOR MACHINE MODEL")
    print("="*50)
    results = []
    
    configs = [
        {'kernel': 'linear', 'decision_function_shape': 'ovr'},
        {'kernel': 'poly', 'decision_function_shape': 'ovr', 'degree': 2}, # Polynomial degree 2
        {'kernel': 'poly', 'decision_function_shape': 'ovr', 'degree': 3}, # Polynomial degree 3
    ]
    cross_validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    for config in configs:
        start_time = time.time()
        clf = SVC(**config)
        predicted_labels = cross_val_predict(clf, features, labels, cv=cross_validator)
        train_time = time.time() - start_time
        
        acc = accuracy_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels, average='macro')
        cm = confusion_matrix(labels, predicted_labels)
        
        results.append({
            'model': 'SVM',
            'config': config,
            'accuracy': acc,
            'f1_score': f1,
            'confusion_matrix': cm,
            'time': train_time
        })
        print(f"Config: {config}, Accuracy Rate: {acc:.2%}, F1 Score: {f1:.4f}, Training Time: {train_time:.4f}s")
        
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
