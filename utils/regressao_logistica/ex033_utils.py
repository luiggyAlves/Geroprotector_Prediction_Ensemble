"""Funções do EX_033.

O EX_033 é o EX_032 com as limitações do artigo endereçadas. Nada do protocolo muda: os
mesmos 18 encoders, as mesmas 15 folds e os mesmos checkpoints dos 1.080 membros são
reaproveitados. O que entra é:

* **Stacking calibrado** (limitação vi): calibração isotônica e de Platt ajustadas dentro do
  fold, sobre as probabilidades out-of-fold internas do meta-modelo.
* **Seleção gulosa por PR-AUC** (limitação v): critério sem limiar, que tira a circularidade
  entre a escolha dos membros e o estudo do limiar.
* **Bootstrap por scaffold e teste pareado** (limitação i): intervalos de confiança e
  comparações, reaproveitando o `ex027_utils`.
* **Colapso por InChIKey** (limitação iii): reavalia as predições da base não padronizada com
  uma predição por substância, para testar se a queda da padronização é efeito de contagem.
* **Ablação dos conflitos de rótulo** (limitação iv): as três políticas (positivo, removido,
  negativo) para as 82 substâncias que a fusão por InChIKey deixou com rótulos divergentes.

Reaproveita do ex032_utils todo o pipeline e do ex027_utils as funções de bootstrap.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from ex027_utils import (all_metrics, corrected_resampled_ttest,  # noqa: F401
                         group_bootstrap_ci, group_bootstrap_indices,
                         paired_group_bootstrap)
from ex032_utils import (ENSEMBLES, GRADE_LIMIAR, K_TRIAGEM, LIMIAR_PADRAO,  # noqa: F401
                         MEDIDAS_DIVERSIDADE, METRICAS, NOMES_METRICAS, RANDOM_STATE,
                         Checkpoints, acertos_oof, bedroc, build_pipeline, chemistry_table,
                         comparar, correlation_measure, disagreement_measure,
                         double_fault_measure, duplicate_audit, encoders_do_modelo,
                         estudo_limiar, external_cuts, folds_internos, formatar_dp,
                         inner_oof_and_C, log_external_read, media_dp, media_por_modelo,
                         metricas, metricas_triagem, padronizar_base, pairwise_matrix,
                         precision_at_k, prever_finais, prever_modelo, q_statistic,
                         recall_at_k, repeated_random_folds, repeated_scaffold_folds,
                         rodar_membros, scaffolds_compartilhados, selecao_gulosa,
                         tabela_metricas, treinar_finais)

# Os ensembles do EX_033: os quatro do EX_032 mais os dois stackings calibrados e a seleção
# gulosa por PR-AUC. "Seleção gulosa" continua sendo a do EX_032 (F1 no limiar 0,5).
ENSEMBLES_NOVOS = ["Stacking (isotônica)", "Stacking (Platt)", "Seleção gulosa (PR-AUC)"]
ENSEMBLES_033 = list(ENSEMBLES) + ENSEMBLES_NOVOS

POLITICAS_CONFLITO = ["positivo", "removido", "negativo"]

# Grade larga para o limiar escolhido dentro do fold. A grade do EX_032 (0,40 a 0,60) só serve
# para modelos cujas probabilidades já estão perto de 0,5; um modelo calibrado com prevalência
# de 17% tem o ótimo bem abaixo disso.
GRADE_LIMIAR_LARGA = np.round(np.arange(0.05, 0.9501, 0.01), 2)


# --------------------------------------------------------------------------------------
# 1. Stacking calibrado (limitação vi)
# --------------------------------------------------------------------------------------

def meta_modelo(random_state=RANDOM_STATE):
    """Meta-modelo do stacking: regressão logística L2 sobre as probabilidades dos membros."""
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                              random_state=random_state)


def calibrar(escore_interno, y_treino, metodo):
    """Calibrador ajustado nas probabilidades out-of-fold internas do meta-modelo.

    O meta-modelo usa ``class_weight="balanced"``, o que desloca as probabilidades para cima:
    no limiar 0,5 ele classifica positivo demais, com recall alto e precision baixa. O
    calibrador aprende a transformação monótona que leva o escore de volta a uma probabilidade,
    e é ajustado só com a parte de treino (nas predições internas, que o meta-modelo final não
    viu durante o ajuste de cada subfold).

    * ``"isotonica"``: regressão isotônica, flexível e sem forma funcional;
    * ``"platt"``: regressão logística de uma variável (escala sigmoide).

    Devolve uma função que aplica a calibração a um vetor de escores.
    """
    escore_interno = np.asarray(escore_interno, dtype=float)
    y_treino = np.asarray(y_treino)
    if metodo == "isotonica":
        modelo = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        modelo.fit(escore_interno, y_treino)
        return lambda escores: np.clip(modelo.predict(np.asarray(escores, dtype=float)), 0, 1)
    if metodo == "platt":
        modelo = LogisticRegression(C=1e10, max_iter=5000, solver="lbfgs")
        modelo.fit(escore_interno.reshape(-1, 1), y_treino)
        return lambda escores: modelo.predict_proba(
            np.asarray(escores, dtype=float).reshape(-1, 1))[:, 1]
    raise ValueError(f"método de calibração desconhecido: {metodo}")


# --------------------------------------------------------------------------------------
# 2. Seleção gulosa sem limiar (limitação v)
# --------------------------------------------------------------------------------------

def selecao_gulosa_pr_auc(matriz_interna, y_treino, nomes):
    """Seleção gulosa (forward) maximizando a PR-AUC das probabilidades médias.

    A versão do EX_032 maximiza o F1 no limiar 0,5, o que deixa o escore interno otimizado
    naquele ponto e torna o estudo do limiar circular para esse ensemble. A PR-AUC não depende
    de limiar e é a métrica de ranking mais próxima do uso em triagem, dada a prevalência de
    17%. O resto do procedimento é idêntico: só a parte de treino do fold é usada e, em empate,
    fica o subconjunto menor.
    """
    colunas = {nome: matriz_interna[:, j] for j, nome in enumerate(nomes)}
    restantes, escolhidos, caminho = list(nomes), [], []
    while restantes:
        avaliados = []
        for candidato in restantes:
            media = np.mean([colunas[m] for m in escolhidos + [candidato]], axis=0)
            avaliados.append((average_precision_score(y_treino, media), candidato))
        pr_auc, candidato = max(avaliados, key=lambda par: par[0])
        escolhidos.append(candidato)
        restantes.remove(candidato)
        caminho.append({"n_membros": len(escolhidos), "adicionado": candidato,
                        "membros": tuple(escolhidos), "pr_auc_interno": float(pr_auc)})
    melhor = max(caminho, key=lambda passo: (passo["pr_auc_interno"], -passo["n_membros"]))
    return list(melhor["membros"]), pd.DataFrame(caminho)


# --------------------------------------------------------------------------------------
# 3. Ensembles do fold, com os três novos
# --------------------------------------------------------------------------------------

def ensembles_do_fold(y, tr, registros_fold, nomes, seed):
    """Os sete ensembles de um fold externo.

    Os quatro do EX_032 (soft voting, soft ponderado, stacking e seleção gulosa por F1) mais
    os dois stackings calibrados e a seleção gulosa por PR-AUC. Toda decisão sai das
    probabilidades out-of-fold internas da parte de treino; o teste do fold não entra em
    nenhuma escolha.
    """
    interna = np.column_stack([registros_fold[n]["inner_oof"] for n in nomes])
    treino = np.column_stack([registros_fold[n]["train_proba"] for n in nomes])
    teste = np.column_stack([registros_fold[n]["test_proba"] for n in nomes])
    y_tr = y[tr]

    pesos = np.array([registros_fold[n]["inner_auc"] for n in nomes])
    pesos = pesos / pesos.sum()

    meta = meta_modelo().fit(interna, y_tr)
    cv_meta = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    meta_interno = cross_val_predict(clone(meta_modelo()), interna, y_tr, cv=cv_meta,
                                     method="predict_proba")[:, 1]
    meta_treino = meta.predict_proba(treino)[:, 1]
    meta_teste = meta.predict_proba(teste)[:, 1]

    subconjunto, caminho = selecao_gulosa(interna, y_tr, nomes)
    colunas = [nomes.index(n) for n in subconjunto]
    subconjunto_pr, caminho_pr = selecao_gulosa_pr_auc(interna, y_tr, nomes)
    colunas_pr = [nomes.index(n) for n in subconjunto_pr]

    saida = {
        "Soft voting": {"train_proba": treino.mean(axis=1), "test_proba": teste.mean(axis=1),
                        "inner_score": interna.mean(axis=1)},
        "Soft voting ponderado (ROC-AUC)": {
            "train_proba": treino @ pesos, "test_proba": teste @ pesos,
            "inner_score": interna @ pesos, "pesos": dict(zip(nomes, pesos))},
        "Stacking": {"train_proba": meta_treino, "test_proba": meta_teste,
                     "inner_score": meta_interno,
                     "coeficientes": dict(zip(nomes, meta.coef_.ravel()))},
        "Seleção gulosa": {"train_proba": treino[:, colunas].mean(axis=1),
                           "test_proba": teste[:, colunas].mean(axis=1),
                           "inner_score": interna[:, colunas].mean(axis=1),
                           "membros": subconjunto, "caminho": caminho},
        "Seleção gulosa (PR-AUC)": {"train_proba": treino[:, colunas_pr].mean(axis=1),
                                    "test_proba": teste[:, colunas_pr].mean(axis=1),
                                    "inner_score": interna[:, colunas_pr].mean(axis=1),
                                    "membros": subconjunto_pr, "caminho": caminho_pr},
    }
    # Os dois stackings calibrados reusam o mesmo meta-modelo: só a transformação final muda,
    # então a ordenação (e portanto ROC-AUC e PR-AUC) é idêntica à do stacking original.
    for metodo, nome in (("isotonica", "Stacking (isotônica)"), ("platt", "Stacking (Platt)")):
        calibrador = calibrar(meta_interno, y_tr, metodo)
        saida[nome] = {"train_proba": calibrador(meta_treino),
                       "test_proba": calibrador(meta_teste),
                       "inner_score": calibrador(meta_interno),
                       "metodo_calibracao": metodo}
    return saida


def limiar_in_fold(y, folds, registros, modelos, grade=GRADE_LIMIAR_LARGA,
                   limiar_padrao=LIMIAR_PADRAO):
    """Métricas de teste no limiar 0,5 e no limiar escolhido dentro de cada fold.

    Em cada fold, o limiar é o que maximiza o F1 nas probabilidades out-of-fold internas da
    parte de treino, sobre uma grade larga; o teste do fold não entra na escolha. Calibrar um
    modelo e continuar lendo-o em 0,5 não resolve nada: a calibração encolhe as probabilidades
    na direção da prevalência (17%), então em 0,5 quase nada é previsto positivo e o recall
    desaba. O par calibração + limiar escolhido no fold é que responde à pergunta.
    """
    y = np.asarray(y)
    linhas = []
    for (seed, k), (tr, te) in folds.items():
        for modelo in modelos:
            registro = registros[(seed, k, modelo)]
            interno, teste = registro["inner_score"], registro["test_proba"]
            f1_interno = [f1_score(y[tr], (interno >= limiar).astype(int), zero_division=0)
                          for limiar in grade]
            escolhido = float(grade[int(np.argmax(f1_interno))])
            linhas.append({
                "modelo": modelo, "seed": seed, "fold": k, "limiar escolhido": escolhido,
                **{f"{k_}: 0,5": v for k_, v in metricas(y[te], teste, limiar_padrao).items()},
                **{f"{k_}: limiar do fold": v for k_, v in metricas(y[te], teste, escolhido).items()},
            })
    return pd.DataFrame(linhas)


def frequencia_selecao(ensembles, nomes, modelo="Seleção gulosa"):
    """Em quantos folds cada representação entrou numa das seleções gulosas."""
    contagem = {n: 0 for n in nomes}
    total = 0
    for (seed, k, nome), registro in ensembles.items():
        if nome == modelo:
            total += 1
            for n in registro["membros"]:
                contagem[n] += 1
    return (pd.Series(contagem, name="folds").sort_values(ascending=False).to_frame()
            .assign(frequencia=lambda d: d["folds"] / total))


# --------------------------------------------------------------------------------------
# 4. Bootstrap por scaffold e testes pareados (limitação i)
# --------------------------------------------------------------------------------------

def oof_de_uma_repeticao(y, folds, registros, modelos, seed=RANDOM_STATE):
    """Monta o vetor de predições out-of-fold de uma repetição, por modelo.

    Numa repetição, as 5 folds cobrem a base inteira, então cada composto é previsto
    exatamente uma vez, por um modelo que não o viu. É sobre esse vetor que o bootstrap
    reamostra scaffolds inteiros.
    """
    y = np.asarray(y)
    proba = {m: np.full(len(y), np.nan) for m in modelos}
    for (s, k), (tr, te) in folds.items():
        if s != seed:
            continue
        for m in modelos:
            proba[m][te] = registros[(s, k, m)]["test_proba"]
    faltando = {m: int(np.isnan(v).sum()) for m, v in proba.items() if np.isnan(v).any()}
    assert not faltando, f"predições OOF incompletas: {faltando}"
    return proba


def tabela_intervalos(y, proba, grupos, modelos, metricas_=None, n_boot=2000,
                      limiar=LIMIAR_PADRAO, seed=RANDOM_STATE):
    """IC95% de cada modelo, reamostrando scaffolds inteiros."""
    metricas_ = metricas_ or ["roc_auc", "average_precision", "f1", "matthews_corrcoef",
                              "bedroc_20", "precision_at_100"]
    linhas = []
    for m in modelos:
        tabela = group_bootstrap_ci(y, proba[m], grupos, limiar, metrics=metricas_,
                                    n_boot=n_boot, seed=seed)
        linhas.append(tabela.assign(modelo=m))
    return pd.concat(linhas, ignore_index=True)[
        ["modelo", "metric", "point", "ci_low", "ci_high", "n_replicates"]]


def tabela_pareada(y, proba, grupos, desafiantes, referencia, metricas_=None, n_boot=2000,
                   limiar=LIMIAR_PADRAO, seed=RANDOM_STATE):
    """Diferença desafiante − referência, com o mesmo reamostramento nos dois modelos."""
    metricas_ = metricas_ or ["roc_auc", "average_precision", "f1", "matthews_corrcoef",
                              "bedroc_20"]
    linhas = []
    for m in desafiantes:
        tabela = paired_group_bootstrap(y, proba[m], proba[referencia], grupos, limiar, limiar,
                                        metrics=metricas_, n_boot=n_boot, seed=seed)
        linhas.append(tabela.assign(modelo=m, referencia=referencia))
    return pd.concat(linhas, ignore_index=True)[
        ["modelo", "referencia", "metric", "A", "B", "difference", "ci_low", "ci_high",
         "p_two_sided", "excludes_zero"]]


def tabela_nadeau_bengio(por_fold, desafiantes, referencia, metricas_=None, parte="teste"):
    """Teste t corrigido de Nadeau–Bengio sobre as 15 folds.

    As folds de uma CV repetida compartilham dados de treino, então o teste t pareado comum é
    anticonservador. A correção troca o fator de variância 1/n por 1/n + n_teste/n_treino. Com
    5 folds, n_teste/n_treino = 1/4, e o fator vai de 1/15 para 0,3167.
    """
    metricas_ = metricas_ or ["f1", "roc_auc", "average_precision", "mcc"]
    tabela = por_fold.query("parte == @parte").set_index(["modelo", "seed", "fold"])
    linhas = []
    for m in desafiantes:
        for metrica in metricas_:
            a = tabela.loc[m, metrica].sort_index()
            b = tabela.loc[referencia, metrica].sort_index()
            resultado = corrected_resampled_ttest(a.to_numpy(), b.to_numpy(),
                                                  n_train=4, n_test=1)
            linhas.append({"modelo": m, "referencia": referencia, "metrica": metrica,
                           "Δ médio": resultado["mean_difference"],
                           "p corrigido": resultado["p_value"],
                           "p ingênuo": resultado["naive_p_value"]})
    return pd.DataFrame(linhas)


# --------------------------------------------------------------------------------------
# 5. Colapso por InChIKey (limitação iii)
# --------------------------------------------------------------------------------------

def colapsar_por_inchikey(y, proba, chaves):
    """Agrega as predições da base não padronizada em uma predição por substância.

    Na base não padronizada, uma substância com várias formas é contada várias vezes no fold
    de teste. Se as substâncias com mais formas forem mais fáceis, o F1 fica inflado sem que o
    modelo seja melhor. Colapsar por InChIKey (média das probabilidades, rótulo pela mesma
    regra "qualquer forma positiva" da padronização) remove esse efeito de contagem sem mexer
    no modelo. Se o Δ para a base padronizada sumir, a queda era contagem.

    Linhas sem InChIKey ficam de fora. Devolve o rótulo e a probabilidade por substância.
    """
    tabela = pd.DataFrame({"chave": np.asarray(chaves), "y": np.asarray(y)})
    for nome, vetor in proba.items():
        tabela[nome] = np.asarray(vetor)
    tabela = tabela[tabela["chave"].notna()]
    agregado = tabela.groupby("chave", sort=False).agg(
        {"y": "max", **{nome: "mean" for nome in proba}})
    return agregado["y"].to_numpy(), {n: agregado[n].to_numpy() for n in proba}


# --------------------------------------------------------------------------------------
# 6. Ablação dos conflitos de rótulo (limitação iv)
# --------------------------------------------------------------------------------------

def aplicar_politica_conflito(base, politica):
    """Aplica uma das três políticas às substâncias com rótulos conflitantes.

    A fusão por InChIKey junta formas da mesma substância que vinham com rótulos diferentes;
    o EX_032 resolve todas como positivas. As três políticas medem quanto dessa escolha
    aparece nos resultados:

    * ``"positivo"``: a regra do EX_032 (nada muda);
    * ``"removido"``: as substâncias conflitantes saem da base;
    * ``"negativo"``: viram negativas.

    Devolve a base com a política aplicada e as posições mantidas, para recortar embeddings e
    grupos de scaffold na mesma ordem.
    """
    conflitante = base["rotulo_conflitante"].to_numpy().astype(bool)
    if politica == "positivo":
        return base.copy(), np.arange(len(base))
    if politica == "negativo":
        saida = base.copy()
        saida.loc[conflitante, "label"] = 0
        return saida, np.arange(len(base))
    if politica == "removido":
        mantidas = np.flatnonzero(~conflitante)
        return base.iloc[mantidas].reset_index(drop=True), mantidas
    raise ValueError(f"política desconhecida: {politica}")
