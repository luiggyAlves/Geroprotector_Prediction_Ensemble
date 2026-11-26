"""Funções do EX_032.

O EX_032 refaz o pipeline do EX_027 a partir de uma base padronizada quimicamente: as
duplicatas estruturais que o EX_027 só contava agora são removidas antes de gerar os
embeddings. A partir daí:

* regressão logística L2 por representação, com C escolhido por CV interna aninhada (como no
  EX_027), em 5 folds x 3 seeds, nos protocolos scaffold e aleatório;
* métricas no limiar 0,5, em treino e teste, por fold e na média;
* três ensembles com as 20 representações (soft, soft ponderado por ROC-AUC, stacking) e a
  seleção gulosa, todos decididos só com a parte de treino de cada fold;
* modelos finais treinados com a base inteira.

Reaproveita do ex027_utils a padronização (RDKit), os folds, o pipeline da regressão
logística, a CV interna e as medidas de diversidade, para que os dois experimentos usem
exatamente o mesmo código nessas partes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict

from ex027_utils import (RANDOM_STATE, Checkpoints, bedroc, build_pipeline,  # noqa: F401
                         chemistry_table, correlation_measure, disagreement_measure,
                         double_fault_measure, duplicate_audit, external_cuts, inner_oof_and_C,
                         log_external_read, pairwise_matrix, precision_at_k, q_statistic,
                         repeated_random_folds, repeated_scaffold_folds)

LIMIAR_PADRAO = 0.5
METRICAS = ["accuracy", "recall", "precision", "f1", "balanced_accuracy", "mcc", "roc_auc",
            "average_precision"]
NOMES_METRICAS = {"accuracy": "Accuracy", "recall": "Recall", "precision": "Precision",
                  "f1": "F1", "balanced_accuracy": "Balanced accuracy", "mcc": "MCC",
                  "roc_auc": "ROC-AUC", "average_precision": "AP (PR-AUC)"}
K_TRIAGEM = (50, 100, 200)
ENSEMBLES = ["Soft voting", "Soft voting ponderado (ROC-AUC)", "Stacking", "Seleção gulosa"]
GRADE_LIMIAR = np.round(np.arange(0.40, 0.6001, 0.01), 2)


# --------------------------------------------------------------------------------------
# 1. Padronização química
# --------------------------------------------------------------------------------------

def padronizar_base(base, coluna_smiles="smiles", coluna_rotulo="label"):
    """Padroniza a base e remove as duplicatas estruturais.

    Cada SMILES passa pela mesma padronização do EX_027 (cleanup, fragmento pai, neutralização,
    tautômero canônico). Linhas com o mesmo InChIKey padronizado são a mesma substância e viram
    uma linha só:

    * SMILES: o tautômero canônico padronizado da primeira ocorrência;
    * rótulo: positivo se qualquer forma for positiva -- a mesma regra que a base já usa para
      conflitos entre fontes;
    * as demais colunas guardam a primeira ocorrência; ``n_formas`` diz quantas linhas foram
      unidas e ``rotulo_conflitante`` marca as substâncias que tinham rótulos diferentes.

    Moléculas que o RDKit não consegue ler são descartadas. Devolve a base padronizada, a
    tabela de padronização por linha original e um resumo com as contagens.
    """
    base = base.reset_index(drop=True)
    quimica = chemistry_table(base[coluna_smiles].astype(str).tolist())
    quimica[coluna_rotulo] = base[coluna_rotulo].to_numpy()
    validas = quimica["L5_inchikey"].notna()

    linhas = []
    for chave, grupo in quimica[validas].groupby("L5_inchikey", sort=False):
        primeira = grupo.index[0]
        rotulos = grupo[coluna_rotulo]
        linha = base.loc[primeira].to_dict()
        linha.update({
            "smiles_original": base.loc[primeira, coluna_smiles],
            coluna_smiles: grupo.loc[primeira, "L4_tautomer"],
            coluna_rotulo: int(rotulos.max()),
            "inchikey": chave,
            "n_formas": len(grupo),
            "rotulo_conflitante": bool(rotulos.nunique() > 1),
        })
        linhas.append(linha)
    padronizada = pd.DataFrame(linhas)

    resumo = {
        "linhas_originais": len(base),
        "nao_lidas_pelo_rdkit": int((~validas).sum()),
        "substancias_distintas": len(padronizada),
        "linhas_removidas_como_duplicata": int(validas.sum() - len(padronizada)),
        "grupos_com_duplicatas": int((padronizada["n_formas"] > 1).sum()),
        "grupos_com_rotulo_conflitante": int(padronizada["rotulo_conflitante"].sum()),
        "positivos_antes": int(base[coluna_rotulo].sum()),
        "positivos_depois": int(padronizada[coluna_rotulo].sum()),
    }
    return padronizada, quimica, resumo


# --------------------------------------------------------------------------------------
# 2. Métricas no limiar fixo
# --------------------------------------------------------------------------------------

def metricas(y, proba, limiar=LIMIAR_PADRAO):
    """As oito métricas do estudo para um vetor de probabilidades. As seis primeiras usam o corte
    em ``limiar``; ROC-AUC e AP (área sob a curva precision-recall) não dependem de limiar."""
    y = np.asarray(y)
    proba = np.asarray(proba)
    pred = (proba >= limiar).astype(int)
    duas_classes = len(np.unique(y)) > 1
    return {
        "accuracy": accuracy_score(y, pred),
        "recall": recall_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "roc_auc": roc_auc_score(y, proba) if duas_classes else np.nan,
        "average_precision": average_precision_score(y, proba) if duas_classes else np.nan,
    }


def recall_at_k(y, escores, k):
    """Fração dos positivos que aparecem entre os ``k`` compostos de maior escore."""
    y = np.asarray(y)
    k = min(k, len(y))
    topo = np.argsort(-np.asarray(escores), kind="mergesort")[:k]
    return float(y[topo].sum() / y.sum()) if y.sum() else np.nan


def metricas_triagem(y, escores, ks=K_TRIAGEM):
    """Métricas de triagem, livres de limiar: Precision@K, Recall@K e BEDROC (alpha = 20)."""
    saida = {"compostos": len(y), "positivos": int(np.sum(y)), "prevalência": float(np.mean(y))}
    for k in ks:
        saida[f"Precision@{k}"] = precision_at_k(y, escores, k)
        saida[f"Recall@{k}"] = recall_at_k(y, escores, k)
    saida["BEDROC (α=20)"] = bedroc(y, escores, alpha=20.0)
    return saida


def scaffolds_compartilhados(folds, grupos):
    """Por fold, quantos grupos de scaffold aparecem no treino e no teste ao mesmo tempo."""
    grupos = np.asarray(grupos)
    return pd.DataFrame([{"seed": seed, "fold": k,
                          "scaffolds no teste": len(set(grupos[te])),
                          "scaffolds em treino e teste": len(set(grupos[tr]) & set(grupos[te]))}
                         for seed, k, tr, te in folds])


# --------------------------------------------------------------------------------------
# 3. Modelos individuais: CV interna aninhada por fold
# --------------------------------------------------------------------------------------

def folds_internos(y_treino, grupos_treino, protocolo, seed, n_splits=5):
    """Folds internos da parte de treino, com o mesmo critério do fold externo."""
    vazio = np.zeros(len(y_treino))
    if protocolo == "scaffold":
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(cv.split(vazio, y_treino, groups=grupos_treino))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(vazio, y_treino))


def ajustar_membro(X, y, tr, te, internos, C_grid, binario, n_jobs=1):
    """Um modelo de uma representação em um fold externo.

    A CV interna (só com a parte de treino) escolhe C pelo ROC-AUC e devolve as
    probabilidades out-of-fold internas, que alimentam os ensembles. O modelo final do fold é
    treinado na parte de treino inteira e prevê treino e teste.
    """
    interno = inner_oof_and_C(X[tr], y[tr], internos, C_grid, binary=binario, n_jobs=n_jobs)
    modelo = build_pipeline(binary=binario, C=interno["C"]).fit(X[tr], y[tr])
    return {"C": interno["C"], "inner_auc": interno["inner_auc"],
            "inner_oof": interno["inner_oof"],
            "train_proba": modelo.predict_proba(X[tr])[:, 1],
            "test_proba": modelo.predict_proba(X[te])[:, 1]}


def rodar_membros(X, y, folds, nomes, binarios, C_grid, checkpoints, protocolo,
                  grupos=None, n_jobs=1, prefixo=None):
    """Todos os modelos individuais de um protocolo, com checkpoint por (fold, representação).

    ``prefixo`` separa os checkpoints de bases diferentes (padrão: o nome do protocolo).
    """
    registros = {}
    for seed, k, tr, te in folds:
        internos = folds_internos(y[tr], None if grupos is None else grupos[tr], protocolo, seed)
        for nome in nomes:
            chave = f"membro_{prefixo or protocolo}_seed{seed}_fold{k}_{nome}"
            registros[(seed, k, nome)] = checkpoints.run_or_load(
                chave,
                lambda nome=nome: ajustar_membro(X[nome], y, tr, te, internos, C_grid,
                                                 nome in binarios, n_jobs=n_jobs),
                verbose=False)
    return registros


def tabela_metricas(y, folds, probas, protocolo, base):
    """Métricas de treino e teste por fold. ``probas[(seed, k, modelo)]`` tem as chaves
    ``train_proba`` e ``test_proba``."""
    linhas = []
    for (seed, k, modelo), registro in probas.items():
        tr, te = folds[(seed, k)]
        for parte, idx, chave in (("treino", tr, "train_proba"), ("teste", te, "test_proba")):
            linhas.append({"base": base, "protocolo": protocolo, "modelo": modelo, "seed": seed, "fold": k,
                           "parte": parte, **metricas(y[idx], registro[chave])})
    return pd.DataFrame(linhas)


def media_por_modelo(tabela):
    """Média e desvio-padrão das 15 folds, por base, protocolo, modelo e parte."""
    agregado = tabela.groupby(["base", "protocolo", "modelo", "parte"])[METRICAS].agg(["mean", "std"])
    agregado.columns = [f"{m}_{estatistica}" for m, estatistica in agregado.columns]
    return agregado.reset_index()


def media_dp(media, desvio, casas=4):
    """Texto "média ± desvio-padrão". Aceita escalares ou Series com o mesmo índice."""
    if isinstance(media, pd.Series):
        return pd.Series([media_dp(m, d, casas) for m, d in zip(media, desvio.loc[media.index])],
                         index=media.index)
    return f"{media:.{casas}f} ± {desvio:.{casas}f}"


def formatar_dp(tabela, casas=4):
    """Junta cada coluna ``X`` com a sua ``X dp`` em um texto "média ± desvio-padrão"."""
    saida = tabela.copy()
    for coluna in [c for c in tabela.columns if f"{c} dp" in tabela.columns]:
        saida[coluna] = media_dp(tabela[coluna], tabela[f"{coluna} dp"], casas)
    return saida.drop(columns=[c for c in tabela.columns if str(c).endswith(" dp")])


def comparar(media, coluna, a, b, metricas_=("f1", "roc_auc"), parte="teste"):
    """Média e desvio-padrão das 15 folds de cada modelo em dois valores de ``coluna``
    (protocolo ou base), lado a lado, e a diferença ``b - a``. ``media`` vem de
    ``media_por_modelo`` já filtrada no outro eixo. As folds dos dois lados não são pareadas,
    então o desvio do Δ é a combinação dos dois desvios, √(dp_a² + dp_b²)."""
    selecionada = media[media["parte"] == parte]
    saida = {}
    for m in metricas_:
        largo = selecionada.pivot(index="modelo", columns=coluna, values=f"{m}_mean")
        desvio = selecionada.pivot(index="modelo", columns=coluna, values=f"{m}_std")
        nome = NOMES_METRICAS[m]
        saida[f"{nome} ({a})"] = largo[a]
        saida[f"{nome} ({a}) dp"] = desvio[a]
        saida[f"{nome} ({b})"] = largo[b]
        saida[f"{nome} ({b}) dp"] = desvio[b]
        saida[f"Δ {nome} ({b} − {a})"] = largo[b] - largo[a]
        saida[f"Δ {nome} ({b} − {a}) dp"] = np.sqrt(desvio[a] ** 2 + desvio[b] ** 2)
    return pd.DataFrame(saida)


# --------------------------------------------------------------------------------------
# 4. Ensembles e seleção gulosa, decididos só com a parte de treino do fold
# --------------------------------------------------------------------------------------

def selecao_gulosa(matriz_interna, y_treino, nomes, limiar=LIMIAR_PADRAO):
    """Seleção gulosa (forward) para soft voting, maximizando o F1 no limiar fixo.

    A cada passo entra a representação que, somada às já escolhidas, dá o maior F1 da média
    das probabilidades out-of-fold internas. Fica o melhor passo do caminho; em empate, o
    subconjunto menor. Usa só a parte de treino do fold.
    """
    colunas = {nome: matriz_interna[:, j] for j, nome in enumerate(nomes)}
    restantes, escolhidos, caminho = list(nomes), [], []
    while restantes:
        avaliados = []
        for candidato in restantes:
            media = np.mean([colunas[m] for m in escolhidos + [candidato]], axis=0)
            avaliados.append((f1_score(y_treino, (media >= limiar).astype(int), zero_division=0),
                              candidato))
        f1, candidato = max(avaliados, key=lambda par: par[0])
        escolhidos.append(candidato)
        restantes.remove(candidato)
        caminho.append({"n_membros": len(escolhidos), "adicionado": candidato,
                        "membros": tuple(escolhidos), "f1_interno": float(f1)})
    melhor = max(caminho, key=lambda passo: (passo["f1_interno"], -passo["n_membros"]))
    return list(melhor["membros"]), pd.DataFrame(caminho)


def meta_modelo(random_state=RANDOM_STATE):
    """Meta-modelo do stacking: regressão logística L2 sobre as 20 probabilidades."""
    return LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                              random_state=random_state)


def ensembles_do_fold(y, tr, registros_fold, nomes, seed):
    """Os quatro ensembles de um fold externo.

    Para cada ensemble devolve as probabilidades em treino e teste e um escore interno (sobre
    a parte de treino, sem olhar o teste) usado depois para escolher limiar:

    * soft voting: média das 20 probabilidades;
    * soft voting ponderado: média ponderada pelo ROC-AUC interno de cada membro;
    * stacking: meta-modelo treinado nas probabilidades out-of-fold internas dos membros;
    * seleção gulosa: soft voting do subconjunto escolhido nas probabilidades internas.
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

    subconjunto, caminho = selecao_gulosa(interna, y_tr, nomes)
    colunas = [nomes.index(n) for n in subconjunto]

    return {
        "Soft voting": {"train_proba": treino.mean(axis=1), "test_proba": teste.mean(axis=1),
                        "inner_score": interna.mean(axis=1)},
        "Soft voting ponderado (ROC-AUC)": {
            "train_proba": treino @ pesos, "test_proba": teste @ pesos,
            "inner_score": interna @ pesos, "pesos": dict(zip(nomes, pesos))},
        "Stacking": {"train_proba": meta.predict_proba(treino)[:, 1],
                     "test_proba": meta.predict_proba(teste)[:, 1],
                     "inner_score": meta_interno,
                     "coeficientes": dict(zip(nomes, meta.coef_.ravel()))},
        "Seleção gulosa": {"train_proba": treino[:, colunas].mean(axis=1),
                           "test_proba": teste[:, colunas].mean(axis=1),
                           "inner_score": interna[:, colunas].mean(axis=1),
                           "membros": subconjunto, "caminho": caminho},
    }


def frequencia_selecao(ensembles, nomes):
    """Em quantos folds cada representação entrou na seleção gulosa."""
    contagem = {n: 0 for n in nomes}
    total = 0
    for (seed, k, modelo), registro in ensembles.items():
        if modelo == "Seleção gulosa":
            total += 1
            for n in registro["membros"]:
                contagem[n] += 1
    return (pd.Series(contagem, name="folds").sort_values(ascending=False).to_frame()
            .assign(frequencia=lambda d: d["folds"] / total))


# --------------------------------------------------------------------------------------
# 5. Diversidade
# --------------------------------------------------------------------------------------

def acertos_oof(y, folds, registros, nomes, seed, limiar=LIMIAR_PADRAO):
    """Vetor de acertos (no limiar fixo) de cada membro, montado com as predições de teste de
    uma repetição: cada composto é previsto uma vez, pelo modelo que não o viu."""
    y = np.asarray(y)
    proba = {n: np.full(len(y), np.nan) for n in nomes}
    for (s, k), (tr, te) in folds.items():
        if s != seed:
            continue
        for n in nomes:
            proba[n][te] = registros[(s, k, n)]["test_proba"]
    return {n: (proba[n] >= limiar).astype(int) == y for n in nomes}


MEDIDAS_DIVERSIDADE = {
    "Double-fault": double_fault_measure,
    "Q-statistic": q_statistic,
    "Disagreement": disagreement_measure,
    "Correlation coefficient": correlation_measure,
}


# --------------------------------------------------------------------------------------
# 6. Estudo do limiar
# --------------------------------------------------------------------------------------

def estudo_limiar(y, folds, probas, grade=GRADE_LIMIAR):
    """Curvas de F1 médio por limiar, no OOF interno e no teste, e o limiar escolhido.

    ``probas[(seed, k)]`` tem ``test_proba`` e ``inner_score``. O limiar é um só por modelo: o
    pico da curva OOF média das 15 folds. O teste não entra na escolha, mas o OOF de uma fold
    vem de um treino que contém compostos de teste de outras folds, então há um vazamento leve
    (um único escalar). A curva de teste é só para visualização. Devolve o F1 por fold e limiar,
    as curvas (média e desvio-padrão das 15 folds) e o limiar escolhido.
    """
    linhas = []
    for (seed, k), registro in probas.items():
        tr, te = folds[(seed, k)]
        for limiar in grade:
            linhas.append({
                "seed": seed, "fold": k, "limiar": limiar,
                "f1_oof": f1_score(y[tr], (registro["inner_score"] >= limiar).astype(int),
                                   zero_division=0),
                "f1_teste": f1_score(y[te], (registro["test_proba"] >= limiar).astype(int),
                                     zero_division=0)})
    por_fold = pd.DataFrame(linhas)
    curvas = por_fold.groupby("limiar")[["f1_oof", "f1_teste"]].agg(["mean", "std"])
    curvas.columns = [f"{c}_{estatistica}" for c, estatistica in curvas.columns]
    return por_fold, curvas, float(curvas["f1_oof_mean"].idxmax())


# --------------------------------------------------------------------------------------
# 7. Modelos finais com a base inteira
# --------------------------------------------------------------------------------------

def treinar_finais(X, y, nomes, binarios, C_grid, grupos, seed=RANDOM_STATE, n_jobs=1):
    """Modelos individuais e ensembles treinados com todos os dados.

    C de cada representação e as probabilidades out-of-fold que alimentam pesos, meta-modelo
    e seleção gulosa vêm de uma CV de 5 folds por scaffold sobre a base inteira.
    """
    y = np.asarray(y)
    internos = folds_internos(y, grupos, "scaffold", seed)
    individuais, oof, auc = {}, {}, {}
    for nome in nomes:
        interno = inner_oof_and_C(X[nome], y, internos, C_grid, binary=nome in binarios,
                                  n_jobs=n_jobs)
        individuais[nome] = build_pipeline(binary=nome in binarios,
                                           C=interno["C"]).fit(X[nome], y)
        oof[nome], auc[nome] = interno["inner_oof"], interno["inner_auc"]
    matriz = np.column_stack([oof[n] for n in nomes])
    pesos = np.array([auc[n] for n in nomes])
    subconjunto, caminho = selecao_gulosa(matriz, y, nomes)
    return {
        "individuais": individuais,
        "C": {n: individuais[n].named_steps["clf"].C for n in nomes},
        "pesos_roc_auc": dict(zip(nomes, pesos / pesos.sum())),
        "meta_modelo": meta_modelo().fit(matriz, y),
        "membros_gulosos": subconjunto,
        "caminho_guloso": caminho,
        "nomes": list(nomes),
    }


def encoders_do_modelo(finais, modelo):
    """Representações de que um modelo final precisa para prever."""
    if modelo == "Seleção gulosa":
        return list(finais["membros_gulosos"])
    if modelo in ENSEMBLES:
        return list(finais["nomes"])
    return [modelo]


def prever_modelo(finais, X, modelo):
    """Probabilidades de um modelo final (individual ou ensemble) para novos compostos.

    ``X`` precisa ter só as representações de ``encoders_do_modelo``."""
    nomes = encoders_do_modelo(finais, modelo)
    matriz = np.column_stack([finais["individuais"][n].predict_proba(X[n])[:, 1] for n in nomes])
    if modelo == "Soft voting ponderado (ROC-AUC)":
        return matriz @ np.array([finais["pesos_roc_auc"][n] for n in nomes])
    if modelo == "Stacking":
        return finais["meta_modelo"].predict_proba(matriz)[:, 1]
    return matriz.mean(axis=1)


def prever_finais(finais, X):
    """Probabilidades dos modelos finais para novos compostos (dicionário de embeddings)."""
    nomes = finais["nomes"]
    matriz = np.column_stack([finais["individuais"][n].predict_proba(X[n])[:, 1] for n in nomes])
    pesos = np.array([finais["pesos_roc_auc"][n] for n in nomes])
    colunas = [nomes.index(n) for n in finais["membros_gulosos"]]
    saida = pd.DataFrame(matriz, columns=nomes)
    saida["Soft voting"] = matriz.mean(axis=1)
    saida["Soft voting ponderado (ROC-AUC)"] = matriz @ pesos
    saida["Stacking"] = finais["meta_modelo"].predict_proba(matriz)[:, 1]
    saida["Seleção gulosa"] = matriz[:, colunas].mean(axis=1)
    return saida
