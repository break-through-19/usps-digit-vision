import numpy as np

def analyze_results(results):
    print("\n" + "="*50)
    print("AUTOMATED INSIGHTS & ANALYSIS")
    print("="*50)
    
    # 1. Per-Model Instance Analysis
    print("\n--- 1. Worst Performing Class per Model Instance ---")
    
    overall_class_accuracies = {} # digit -> list of accuracies across all models
    
    for res in results:
        cm = res['confusion_matrix']
        # Calculate per-class accuracy
        # Avoid division by zero
        row_sums = cm.sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            class_acc = np.diag(cm) / row_sums
            class_acc = np.nan_to_num(class_acc) # Replace NaNs with 0 if any class has 0 samples
        
        # Track for overall analysis
        for digit, acc in enumerate(class_acc):
            if digit not in overall_class_accuracies:
                overall_class_accuracies[digit] = []
            overall_class_accuracies[digit].append(acc)
            
        worst_class = np.argmin(class_acc)
        worst_acc = class_acc[worst_class]
        
        # Find most frequent misclassification
        # Mask diagonal to ignore correct predictions
        np.fill_diagonal(cm, 0)
        # Find index of max value
        if cm.sum() > 0:
            mis_idx = np.unravel_index(np.argmax(cm, axis=None), cm.shape)
            true_label, pred_label = mis_idx
            mis_count = cm[true_label, pred_label]
            mis_desc = f"Digit {true_label} often confused as {pred_label} ({mis_count} times)"
        else:
            mis_desc = "None"
            
        print(f"\nModel: {res['model']}")
        print(f"Config: {res['config']}")
        print(f"  -> Worst Class: Digit {worst_class} (Accuracy Rate: {worst_acc:.2%})")
        print(f"  -> Most Frequent Misclassification: {mis_desc}")

    # 2. Overall Hardest Digit
    print("\n--- 2. Overall Hardest & Easiest Digits ---")
    avg_class_acc = {d: np.mean(accs) for d, accs in overall_class_accuracies.items()}
    hardest_digit = min(avg_class_acc, key=avg_class_acc.get)
    easiest_digit = max(avg_class_acc, key=avg_class_acc.get)
    
    print(f"Hardest Digit to Classify (Avg Accuracy Rate): {hardest_digit} ({avg_class_acc[hardest_digit]:.2%})")
    print(f"Easiest Digit to Classify (Avg Accuracy Rate): {easiest_digit} ({avg_class_acc[easiest_digit]:.2%})")

    # 3. Comparative Explanations
    print("\n--- 3. Comparative Explanations ---")
    
    # SVM Analysis
    svm_results = [r for r in results if r['model'] == 'SVM']
    if svm_results:
        linear_acc = next((r['accuracy'] for r in svm_results if r['config']['kernel'] == 'linear'), 0)
        poly_acc = max((r['accuracy'] for r in svm_results if r['config']['kernel'] == 'poly'), default=0)
        
        print("\n[SVM Analysis]")
        if poly_acc > linear_acc:
            print(f"  - Polynomial kernel ({poly_acc:.2%}) outperformed Linear kernel ({linear_acc:.2%}).")
            print("  - Insight: The data is likely not linearly separable. Mapping to a higher-dimensional space helped separate the classes.")
        else:
            print(f"  - Linear kernel performed similarly or better than Polynomial.")
            print("  - Insight: The classes are reasonably well-separated by hyperplanes in the original feature space.")

    # KNN Analysis
    knn_results = [r for r in results if r['model'] == 'KNN' and r['config']['weights'] == 'uniform']
    if knn_results:
        # Sort by k
        knn_results.sort(key=lambda x: x['config']['k'])
        best_k_res = max(knn_results, key=lambda x: x['accuracy'])
        best_k = best_k_res['config']['k']
        
        print("\n[KNN Analysis]")
        print(f"  - Best k value was {best_k} with accuracy {best_k_res['accuracy']:.2%}.")
        if best_k <= 3:
            print("  - Insight: Lower k values performed well, suggesting that the decision boundaries are complex and local neighborhood structure is very informative.")
        else:
            print("  - Insight: Higher k values performed better, suggesting that smoothing the decision boundary helped reduce noise.")

    # Decision Tree Analysis
    dt_results = [r for r in results if r['model'] == 'Decision Tree']
    if dt_results:
        best_dt = max(dt_results, key=lambda x: x['accuracy'])
        worst_dt = min(dt_results, key=lambda x: x['accuracy'])
        
        print("\n[Decision Tree Analysis]")
        print(f"  - Best Config: {best_dt['config']} ({best_dt['accuracy']:.2%})")
        print(f"  - Worst Config: {worst_dt['config']} ({worst_dt['accuracy']:.2%})")
        
        if best_dt['config'].get('max_leaf_nodes') is None and worst_dt['config'].get('max_leaf_nodes') is not None:
             print("  - Insight: Restricting the tree size (max_leaf_nodes) significantly hurt performance, indicating that a complex tree is needed to capture the nuances of handwritten digits.")
