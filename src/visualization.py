"""Headless saved figures generated exclusively from measured data."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from src.data_processing import NUMERIC
from src.fuzzy_logic import PARAMETERS, TERMS, triangular, infer
from src.evaluate import risk_band


def generate_figures(clean, test, probabilities, metrics, histories, tnorm, output='results'):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style='whitegrid', palette='Set2')

    def save(name):
        plt.tight_layout()
        plt.savefig(output / f'{name}.png', dpi=160, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(6, 4))
    sns.countplot(data=clean, x='HeartDisease')
    plt.title('Dataset class distribution')
    save('class_distribution')

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, (key, triples) in zip(axes.flat, PARAMETERS.items()):
        x = np.linspace(triples[0][0], triples[-1][2], 500)
        for term, params in zip(TERMS, triples):
            ax.plot(x, triangular(x, *params), label=term)
        ax.set(title=key, ylim=(-.02, 1.05), ylabel='Membership')
        ax.legend()
    axes.flat[-1].axis('off')
    save('membership_functions')

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, key in zip(axes.flat, NUMERIC):
        sns.histplot(data=clean, x=key, hue='HeartDisease', bins=25,
                     element='step', stat='density', common_norm=False, ax=ax)
    axes.flat[-1].axis('off')
    save('feature_distributions')

    plt.figure(figsize=(8, 6))
    sns.heatmap(clean[NUMERIC + ['FastingBS', 'HeartDisease']].corr(),
                annot=True, fmt='.2f', cmap='vlag', vmin=-1, vmax=1)
    plt.title('Numeric / binary correlation (EDA only)')
    save('correlation_heatmap')

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for name, history in histories.items():
        h = pd.DataFrame(history)
        for ax, metric in zip(axes, ['loss', 'accuracy']):
            ax.plot(h.epoch, h[f'train_{metric}'], linestyle='--', label=f'{name} train')
            ax.plot(h.epoch, h[f'validation_{metric}'], label=f'{name} validation')
            best_epoch = int(h.loc[h.validation_loss.idxmin(), 'epoch'])
            ax.axvline(best_epoch, alpha=.25)
            ax.set(xlabel='Epoch', ylabel=metric)
            ax.legend(fontsize=7)
    save('training_curves')

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (name, values) in zip(axes.flat, metrics.items()):
        sns.heatmap(values['confusion_matrix'], annot=True, fmt='d', cbar=False,
                    ax=ax, cmap='Blues', xticklabels=[0, 1], yticklabels=[0, 1])
        ax.set(title=name, xlabel='Predicted', ylabel='Actual')
    save('confusion_matrices')

    plt.figure(figsize=(7, 6))
    for name, values in metrics.items():
        plt.plot(values['roc']['fpr'], values['roc']['tpr'],
                 label=f"{name} (AUC={values['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False positive rate'); plt.ylabel('True positive rate'); plt.legend()
    save('roc_curves')

    sample = test.sample(min(12, len(test)), random_state=42)
    strengths = infer(sample, tnorm)[1]
    plt.figure(figsize=(11, 6))
    sns.heatmap(strengths, annot=True, fmt='.2f', vmin=0, vmax=1,
                xticklabels=[f'R{i}' for i in range(1, 11)], yticklabels=sample.index)
    plt.title('Rule strengths: twelve real held-out patients')
    save('rule_activation_heatmap')

    plt.figure(figsize=(6, 4))
    bands = pd.Series([risk_band(p) for p in probabilities['Neuro-Fuzzy ANN']])
    counts = bands.value_counts().reindex(['Low', 'Moderate', 'High'], fill_value=0)
    counts.plot.bar(rot=0)
    plt.ylabel('Test patients'); plt.title('Illustrative risk bands')
    save('risk_distribution')

    table = pd.DataFrame({name: {k: v[k] for k in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']}
                          for name, v in metrics.items()}).T
    plt.figure(figsize=(9, 4))
    sns.heatmap(table, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Held-out performance')
    save('performance_comparison')
