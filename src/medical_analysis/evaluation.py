import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, ConfusionMatrixDisplay
from sklearn.preprocessing import label_binarize

def calculate_medical_metrics(y_true, y_pred_probs, class_names=['Normal', 'Pneumonia']):
    """
    Computes key clinical metrics (Sensitivity, Specificity, AUC) and prints classification report.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Classification Report
    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)

    # Sensitivity (Recall of positive class) and Specificity (Recall of negative class)
    # Binary Case: Normal (0), Pneumonia (1)
    if len(class_names) == 2:
        tn, fp, fn, tp = cm.ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        print("-" * 60)
        print(f"Clinical Sensitivity (Recall for Pneumonia) : {sensitivity:.4f}")
        print(f"Clinical Specificity (Recall for Normal)    : {specificity:.4f}")
        print("=" * 60)
        return cm, sensitivity, specificity
    else:
        # Multiclass: overall average sensitivity & specificity per class
        sensitivities = []
        specificities = []
        for i in range(len(class_names)):
            tp = cm[i, i]
            fn = np.sum(cm[i, :]) - tp
            fp = np.sum(cm[:, i]) - tp
            tn = np.sum(cm) - tp - fp - fn
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            sensitivities.append(sens)
            specificities.append(spec)
        print("-" * 60)
        print(f"Average Multi-class Sensitivity: {np.mean(sensitivities):.4f}")
        print(f"Average Multi-class Specificity: {np.mean(specificities):.4f}")
        print("=" * 60)
        return cm, np.mean(sensitivities), np.mean(specificities)

def plot_cm(cm, class_names=['Normal', 'Pneumonia']):
    """
    Plots a Confusion Matrix.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Clinical Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc(y_true, y_pred_probs, class_names=['Normal', 'Pneumonia']):
    """
    Plots the ROC curve and computes AUC score.
    """
    n_classes = len(class_names)

    # If binary classification, convert true labels to binary representation if not already
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    if n_classes == 2:
        # For binary, select probability of the positive class (Pneumonia)
        fpr, tpr, _ = roc_curve(y_true, y_pred_probs[:, 1])
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='tomato', lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate (1 - Specificity)')
        plt.ylabel('True Positive Rate (Sensitivity)')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        # Multiclass ROC plotting
        plt.figure(figsize=(8, 6))
        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_names[i]} (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Multi-class ROC Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.show()

def analyze_false_negatives(y_true, y_pred_probs, class_names=['Normal', 'Pneumonia']):
    """
    Logs and displays stats about missed Pneumonia cases (False Negatives),
    which is a major critical safety check in clinical settings.
    """
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Assuming positive class is Pneumonia (index 1)
    positive_class_idx = 1

    # Identify indices of False Negatives: True is Pneumonia (1), predicted is Normal (0)
    fn_indices = np.where((y_true == positive_class_idx) & (y_pred != positive_class_idx))[0]
    total_positives = np.sum(y_true == positive_class_idx)

    print("\n" + "=" * 55)
    print("CRITICAL CLINICAL SAFETY: FALSE NEGATIVE REPORT")
    print("=" * 55)
    print(f"Total True Pneumonia Cases     : {total_positives}")
    print(f"Missed Cases (False Negatives) : {len(fn_indices)}")
    if total_positives > 0:
        fn_rate = (len(fn_indices) / total_positives) * 100
        print(f"False Negative Rate (FNR)      : {fn_rate:.2f}%")
    print("=" * 55)

    return fn_indices
