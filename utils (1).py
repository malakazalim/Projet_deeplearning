import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def set_seed(seed=42):
    """
    Fixe les seeds pour assurer la reproductibilité des expériences.
    Conforme aux consignes de qualité du projet EMSI.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Seed fixé à {seed} pour la reproductibilité.")

def try_gpu(i=0):
    """
    Renvoie torch.device('cuda:i') si disponible, sinon torch.device('cpu').
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

class EarlyStopping:
    """
    Arrête l'entraînement prématurément si la loss de validation ne s'améliore pas
    après une certaine patience afin d'éviter l'overfitting.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='best_model.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Sauvegarde le modèle lorsque la loss de validation diminue."""
        if self.verbose:
            print(f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...")
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

def plot_learning_curves(train_losses, val_losses, train_metrics, val_metrics, metric_name='Accuracy'):
    """
    Génère les courbes d'apprentissage obligatoires (Loss et Métrique) pour le diagnostic.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Courbe des pertes (Loss)
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Courbes de Perte (Loss)')
    plt.xlabel('Époques')
    plt.ylabel('Perte')
    plt.legend()
    plt.grid(True)
    
    # Courbe de la métrique
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_metrics, 'b-', label=f'Train {metric_name}')
    plt.plot(epochs, val_metrics, 'r-', label=f'Validation {metric_name}')
    plt.title(f'Courbes de Performance ({metric_name})')
    plt.xlabel('Époques')
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_sns(y_true, y_pred, classes, title='Matrice de Confusion'):
    """
    Affiche une matrice de confusion stylisée à l'aide de Seaborn.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Vrais Labels')
    plt.xlabel('Labels Prédits')
    plt.show()

def plot_roc_curve(y_true, y_scores, title='Courbe ROC'):
    """
    Calcule et affiche la courbe ROC ainsi que l'AUC.
    """
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de Faux Positifs (FPR)')
    plt.ylabel('Taux de Vrais Positifs (TPR)')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()