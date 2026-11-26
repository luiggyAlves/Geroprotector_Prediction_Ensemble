import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, StratifiedKFold
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, GridSearchCV
import pandas as pd
from sklearn.experimental import enable_halving_search_cv  # necessário para habilitar o Halving
from sklearn.model_selection import HalvingGridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier


def plot_learning_curve(classificador, X, y, titulo='Curva de aprendizado', n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # --- Curva de aprendizado (apenas accuracy) ---
    train_sizes, train_scores, test_scores = learning_curve(
        classificador, X, y,
        cv=skf,
        scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 10),
        n_jobs=-1
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    test_mean = test_scores.mean(axis=1)
    test_std = test_scores.std(axis=1)

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, 'o-', label='Treino', color='tab:blue')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                      alpha=0.15, color='tab:blue')
    plt.plot(train_sizes, test_mean, 'o-', label='Teste (CV)', color='tab:orange')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                      alpha=0.15, color='tab:orange')
    plt.xlabel('Número de amostras de treino')
    plt.ylabel('Accuracy')
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    print("Médias de treino (accuracy, por tamanho):", train_mean)
    print("Médias de teste (accuracy, por tamanho):", test_mean)

    # --- Médias gerais para todas as métricas (accuracy, recall, f1, roc_auc) ---
    scores = cross_validate(
        classificador, X, y, cv=skf,
        scoring=['accuracy', 'recall', 'f1', 'roc_auc'],
        return_train_score=True,
        n_jobs=-1
    )

    resultado = {
        'accuracy_train': scores['train_accuracy'].mean(),
        'accuracy_test': scores['test_accuracy'].mean(),
        'recall_train': scores['train_recall'].mean(),
        'recall_test': scores['test_recall'].mean(),
        'f1_train': scores['train_f1'].mean(),
        'f1_test': scores['test_f1'].mean(),
        'roc_auc_train': scores['train_roc_auc'].mean(),
        'roc_auc_test': scores['test_roc_auc'].mean(),
    }

    print("\nMédias gerais (todas as métricas):")
    for k, v in resultado.items():
        print(f"{k}: {v:.4f}")

    return resultado



def grid_search_rf(X, y, metrica, param_grid=None, skf=None, n_splits=5, 
                    random_state=42, n_jobs=-1, verbose=2, factor=3, 
                    min_resources='exhaust'):
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [2, 5, 10, 20],
            'min_samples_split': [2, 3, 5, 10],
            'min_samples_leaf': [1, 2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    if skf is None:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    classificador_grid = RandomForestClassifier(random_state=random_state, n_jobs=1)
    
    grid = HalvingGridSearchCV(
        estimator=classificador_grid,
        param_grid=param_grid,
        cv=skf,
        scoring=metrica,
        n_jobs=n_jobs,
        verbose=verbose,
        factor=factor,              # fração de candidatos eliminados a cada iteração
        resource='n_samples',       # recurso que aumenta a cada rodada (padrão)
        min_resources=min_resources,# 'exhaust' usa o máximo possível de forma otimizada
        random_state=random_state,
        return_train_score=True
    )
    
    grid.fit(X, y)
    print("Melhores parâmetros:", grid.best_params_)
    print("Melhor score (validação):", grid.best_score_)
    melhor_modelo = grid.best_estimator_
    
    # Comparar treino vs teste (checar overfitting)
    idx_melhor = grid.best_index_
    resultados = grid.cv_results_
    print(f"Score treino: {resultados['mean_train_score'][idx_melhor]:.4f}")
    print(f"Score teste (validação): {resultados['mean_test_score'][idx_melhor]:.4f}")
    
    return grid, melhor_modelo