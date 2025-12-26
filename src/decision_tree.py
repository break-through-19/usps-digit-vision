import time
import os
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

def train_evaluate_dt(features, labels):
    print("\n" + "="*50)
    print("DECISION TREE MODEL")
    print("="*50)
    results = []
    
    # 2.1.1 Print out the tree (using default params first or a small one)
    dt_default = DecisionTreeClassifier(random_state=42)
    dt_default.fit(features, labels)
    print("Decision Tree Structure:")
    print(export_text(dt_default, feature_names=[f'pixel_{i}' for i in range(features.shape[1])]))
    
    # Plot the tree
    plt.figure(figsize=(20,10))
    plot_tree(dt_default, max_depth=2, feature_names=[f'pixel_{i}' for i in range(features.shape[1])], filled=True)
    plt.title("Decision Tree Visualization (Top Levels)")
    
    # Save figure
    os.makedirs('../results/figures', exist_ok=True)
    plt.savefig('../results/figures/decision_tree_structure.png')
    plt.show()
    
    # 2.1.2 Variations
    configs = [
        {},
        {'max_leaf_nodes': 10, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'max_leaf_nodes': None, 'min_samples_split': 10, 'min_samples_leaf': 1, 'criterion': 'gini'},
        {'max_leaf_nodes': None, 'min_samples_split': 2, 'min_samples_leaf': 5, 'criterion': 'gini'},
        {'max_leaf_nodes': None, 'min_samples_split': 2, 'min_samples_leaf': 1, 'criterion': 'entropy'},
    ]

    cross_validator = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for config in configs:
        start_time = time.time()
        clf = DecisionTreeClassifier(random_state=42, **config)
        predicted_labels = cross_val_predict(clf, features, labels, cv=cross_validator)
        train_time = time.time() - start_time
        
        acc = accuracy_score(labels, predicted_labels)
        f1 = f1_score(labels, predicted_labels, average='macro')
        cm = confusion_matrix(labels, predicted_labels)
        
        results.append({
            'model': 'Decision Tree',
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
        plt.title(f'DT Confusion Matrix\n{config}')
        
        # Save confusion matrix
        os.makedirs('../results/confusion_matrices', exist_ok=True)
        config_str = "_".join([f"{k}-{v}" for k, v in config.items()]) if config else "default"
        plt.savefig(f'../results/confusion_matrices/dt_confusion_matrix_{config_str}.png')
        plt.show()
        
    return results
