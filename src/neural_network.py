"""Adam training with persistent optimizer and explicit checkpoint restoration."""
from copy import deepcopy
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import log_loss, accuracy_score


def train_network(X, y, X_validation, y_validation, seed: int = 42,
                  max_epochs: int = 150, patience: int = 15) -> tuple:
    """One partial_fit call per epoch preserves Adam's state across epochs."""
    if max_epochs < 1 or patience < 1:
        raise ValueError('max_epochs and patience must be positive')
    model = MLPClassifier(hidden_layer_sizes=(16, 8), activation='relu', solver='adam',
                          learning_rate_init=.01, alpha=.001, random_state=seed,
                          early_stopping=False, batch_size='auto', shuffle=True)
    history = []
    best_loss, stale, best = float('inf'), 0, None
    for epoch in range(1, max_epochs + 1):
        model.partial_fit(X, y, classes=np.array([0, 1]))
        train_p = model.predict_proba(X)[:, 1]
        val_p = model.predict_proba(X_validation)[:, 1]
        loss = log_loss(y_validation, val_p, labels=[0, 1])
        history.append({'epoch': epoch, 'train_loss': log_loss(y, train_p, labels=[0, 1]),
                        'validation_loss': loss,
                        'train_accuracy': accuracy_score(y, train_p >= .5),
                        'validation_accuracy': accuracy_score(y_validation, val_p >= .5)})
        if loss < best_loss - 1e-8:
            best_loss, stale, best_epoch = loss, 0, epoch
            best = deepcopy(model)
        else:
            stale += 1
        if stale >= patience:
            break
    best.best_epoch_ = best_epoch
    best.best_validation_loss_ = best_loss
    return best, history
