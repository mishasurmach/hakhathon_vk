import numpy as np
from sklearn.metrics import average_precision_score, ndcg_score, roc_curve, auc
import matplotlib.pyplot as plt
from config import Config

def calculate_ndcg_by_group(y_true, y_pred, groups):
    unique_groups = np.unique(groups)
    ndcg_scores = []
    
    for group in unique_groups:
        group_indices = np.where(groups == group)
        y_true_group = y_true[group_indices]
        y_pred_group = y_pred[group_indices]
        
        if len(y_true_group) > 1: 
            ndcg = ndcg_score([y_true_group], [y_pred_group])
            ndcg_scores.append(ndcg)
    
    return np.mean(ndcg_scores)

def evaluate(y_pred, y_test, group_test):
    ap = average_precision_score(y_test, y_pred)
    print(f"Average Precision: {ap:.4f}")

    ndcg = calculate_ndcg_by_group(y_test, y_pred, group_test)
    print(f"nDCG: {ndcg:.4f}")

    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f"ROC-AUC: {roc_auc:.4f}")

    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    if Config.ENABLE_LOGGING:
        plt.savefig(Config.ROC_CURVE_PLOT_PATH)
    plt.show()
