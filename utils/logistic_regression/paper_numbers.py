"""Confere cada número do artigo contra os resultados salvos em resultados/csv/.

É o caminho curto de reprodução: roda em segundos, sem embeddings e sem reajustar modelo
nenhum, porque lê os CSVs que o notebook do benchmark escreveu. Para refazer os CSVs do zero,
rode o notebook (veja o README).

Cada linha imprime o valor do artigo, o valor calculado e um veredito. O arredondamento é
meio-para-cima, como o do texto; o Python arredonda 0,0215 para 0,021 e o artigo escreve 0,022.

Uso:
    python utils/regressao_logistica/numeros_do_artigo.py
"""

from __future__ import annotations

import os
import sys
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from scipy.stats import spearmanr

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
CSV = os.path.join(RAIZ, "resultados", "csv")
BASES = os.path.join(RAIZ, "dados", "Bases de dados", "Treinamento - limpas")
sys.path.insert(0, AQUI)

import ex033_utils as U  # noqa: E402

ENSEMBLE_ARTIGO = "Seleção gulosa (PR-AUC)"      # "Greedy (AP)" no artigo
MELHOR_INDIVIDUAL = "MolFormer-XL"
conferencias: list[dict] = []


def arred(valor, casas):
    """Arredonda meio-para-cima, como o texto do artigo."""
    return float(Decimal(str(float(valor))).quantize(Decimal(f"1e-{casas}"), rounding=ROUND_HALF_UP))


def confere(secao, afirmacao, esperado, calculado, casas=3):
    """Registra uma conferência. `esperado` None só imprime o valor calculado."""
    ok = None if esperado is None else arred(calculado, casas) == esperado
    conferencias.append({"seção": secao, "afirmação do artigo": afirmacao,
                         "artigo": esperado, "calculado": arred(calculado, casas),
                         "confere": {True: "ok", False: "DIVERGE", None: "—"}[ok]})


def ler(nome):
    return pd.read_csv(os.path.join(CSV, nome))


# ---------------------------------------------------------------------------------- dados
def secao_dados():
    pares = [("§4.1 base não padronizada", "Base_GeroLLM_reconstruida/primary_dataset_dedup.csv",
              4233, 727, 17.17),
             ("§4.1 base padronizada", "Base_GeroLLM_padronizada/primary_padronizada.csv",
              3752, 637, 16.98),
             ("§4.1 externo (bruto)", "Base_GeroLLM_reconstruida/external_test_set.csv",
              1557, 35, 2.25)]
    for rotulo, caminho, n, pos, prev in pares:
        d = pd.read_csv(os.path.join(BASES, caminho))
        confere(rotulo, "compostos", n, len(d), 0)
        confere(rotulo, "positivos", pos, int(d["label"].sum()), 0)
        confere(rotulo, "prevalência (%)", prev, 100 * d["label"].mean(), 2)
        if "n_formas" in d:
            confere(rotulo, "linhas colapsadas", 481, int((d["n_formas"] - 1).sum()), 0)
            confere(rotulo, "grupos de formas redundantes", 411, int((d["n_formas"] > 1).sum()), 0)
            confere(rotulo, "grupos com rótulo conflitante", 82,
                    int(d["rotulo_conflitante"].sum()), 0)


# ------------------------------------------------------------------------- split/vazamento
def secao_split():
    v = ler("ex033_verificacao_scaffold.csv")
    scaffold = v[v["protocolo"] == "split por scaffold"]
    confere("§3.3 split por scaffold", "scaffolds em treino e teste", 0,
            int(scaffold["scaffolds em treino e teste"].sum()), 0)
    for base, esperado in (("base padronizada", 227), ("base não padronizada", 301)):
        sub = v[(v["protocolo"] == "split aleatório estratificado") & (v["base"] == base)]
        confere("§3.3 split aleatório", f"scaffolds compartilhados por fold ({base})", esperado,
                sub["scaffolds em treino e teste"].mean(), 0)


# ------------------------------------------------------------------- tabelas 1, 2 e as 80
def medias():
    i = ler("ex033_metricas_individuais_media.csv").query("parte == 'teste'")
    e = ler("ex033_metricas_ensembles_media.csv").query("parte == 'teste'")
    return i, e


def secao_tabelas():
    i, e = medias()
    TABELA1 = {("original", "aleatorio"): (0.062, 0.040, 0.088, 0.079),
               ("padronizada", "aleatorio"): (0.043, 0.036, 0.059, 0.058),
               ("original", "scaffold"): (0.026, 0.024, 0.056, 0.035),
               ("padronizada", "scaffold"): (0.012, 0.017, 0.033, 0.017)}
    METRICAS = ("f1", "roc_auc", "average_precision", "mcc")
    for (base, prot), esperados in TABELA1.items():
        ind = i.query("base == @base and protocolo == @prot")
        melhor = ind.loc[ind["f1_mean"].idxmax()]
        ens = e.query("base == @base and protocolo == @prot and modelo == @ENSEMBLE_ARTIGO").iloc[0]
        for metrica, esperado in zip(METRICAS, esperados):
            confere("Tabela 1", f"Δ{metrica} ({base}, {prot})", esperado,
                    ens[f"{metrica}_mean"] - melhor[f"{metrica}_mean"])

    cenario = "base == 'padronizada' and protocolo == 'scaffold'"
    TABELA2 = {MELHOR_INDIVIDUAL: ((0.423, 0.036), (0.735, 0.040), (0.428, 0.065)),
               ENSEMBLE_ARTIGO: ((0.435, 0.045), (0.752, 0.044), (0.461, 0.059))}
    for modelo, esperados in TABELA2.items():
        fonte = i if modelo == MELHOR_INDIVIDUAL else e
        linha = fonte.query(cenario).query("modelo == @modelo").iloc[0]
        for metrica, (media, desvio) in zip(("f1", "roc_auc", "average_precision"), esperados):
            confere("Tabela 2", f"{modelo} {metrica} (média)", media, linha[f"{metrica}_mean"])
            confere("Tabela 2", f"{modelo} {metrica} (desvio)", desvio, linha[f"{metrica}_std"])

    # "ensembles won all 80 comparisons" (§4.2 e resumo)
    negativas = 0
    for base in ("original", "padronizada"):
        for prot in ("aleatorio", "scaffold"):
            ind = i.query("base == @base and protocolo == @prot")
            melhor = ind.loc[ind["f1_mean"].idxmax()]
            for modelo in U.ENSEMBLES + [ENSEMBLE_ARTIGO]:
                linha = e.query("base == @base and protocolo == @prot and modelo == @modelo").iloc[0]
                negativas += sum(linha[f"{m}_mean"] - melhor[f"{m}_mean"] <= 0 for m in METRICAS)
    confere("§4.2", "comparações ensemble x melhor individual com Δ <= 0 (de 80)", 0, negativas, 0)

    # troca de recall por precision (§4.2)
    g = e.query(cenario).query("modelo == 'Seleção gulosa'").iloc[0]
    m = i.query(cenario).query("modelo == @MELHOR_INDIVIDUAL").iloc[0]
    confere("§4.2", "Greedy (F1) Δprecision", 0.035, g["precision_mean"] - m["precision_mean"])
    confere("§4.2", "Greedy (F1) Δrecall", -0.047, g["recall_mean"] - m["recall_mean"])


# ------------------------------------------------------------------------------ bootstrap
def secao_bootstrap():
    p = ler("ex033_pareado_scaffold_padronizada.csv").query("modelo == @ENSEMBLE_ARTIGO")
    ESPERADO = {"roc_auc": (0.016, 0.003, 0.027), "average_precision": (0.030, 0.004, 0.052),
                "f1": (0.017, -0.002, 0.036)}
    for metrica, (dif, baixo, alto) in ESPERADO.items():
        linha = p.query("metric == @metrica").iloc[0]
        confere("§4.3", f"Δ{metrica} pareado", dif, linha["difference"])
        confere("§4.3", f"IC de Δ{metrica} (inferior)", baixo, linha["ci_low"])
        confere("§4.3", f"IC de Δ{metrica} (superior)", alto, linha["ci_high"])

    todos = ler("ex033_pareado_scaffold_padronizada.csv")
    ranking = todos[todos["metric"].isin(["roc_auc", "average_precision"])]
    excluem = sorted(ranking[ranking["excludes_zero"]]["modelo"].unique())
    confere("§4.3", "modelos que excluem zero em ROC-AUC/AP (só Greedy (AP))", 1, len(excluem), 0)
    assert excluem == [ENSEMBLE_ARTIGO], excluem

    iv = ler("ex033_intervalos_scaffold_padronizada.csv")
    CONCLUSAO = {(ENSEMBLE_ARTIGO, "roc_auc"): (0.754, 0.725, 0.784),
                 (ENSEMBLE_ARTIGO, "average_precision"): (0.461, 0.412, 0.513),
                 (MELHOR_INDIVIDUAL, "roc_auc"): (0.738, 0.707, 0.773),
                 (MELHOR_INDIVIDUAL, "average_precision"): (0.430, 0.376, 0.493)}
    for (modelo, metrica), (ponto, baixo, alto) in CONCLUSAO.items():
        linha = iv.query("modelo == @modelo and metric == @metrica").iloc[0]
        confere("Conclusão", f"{modelo} {metrica}", ponto, linha["point"])
        confere("Conclusão", f"{modelo} {metrica} IC inferior", baixo, linha["ci_low"])
        confere("Conclusão", f"{modelo} {metrica} IC superior", alto, linha["ci_high"])


# -------------------------------------------------------------------- limiar e calibração
def secao_limiar():
    d = ler("ex033_limiar_por_criterio.csv")
    linha = d.query("base == 'base padronizada' and critério == @ENSEMBLE_ARTIGO").iloc[0]
    confere("§4.4", "limiar do pico OOF, Greedy (AP)", 0.54, linha["limiar escolhido (pico OOF)"], 2)
    confere("§4.4", "F1 de teste em 0,5", 0.435, linha["F1 teste em 0,5"])
    confere("§4.4", "F1 de teste no limiar escolhido", 0.443,
            linha["F1 teste no limiar escolhido"])
    ganho = (d["F1 teste no limiar escolhido"] - d["F1 teste em 0,5"]).max()
    confere("§4.4", "ganho máximo ao mover o limiar", 0.008, ganho)
    linha_f1 = d.query("base == 'base padronizada' and critério == 'Seleção gulosa'").iloc[0]
    confere("§4.4", "limiar do pico OOF, Greedy (F1)", 0.50,
            linha_f1["limiar escolhido (pico OOF)"], 2)

    inf = ler("ex033_limiar_in_fold_scaffold_padronizada.csv")
    f1 = inf.groupby("modelo")["f1: limiar do fold"].mean()
    espalhamento = f1[["Stacking", "Stacking (Platt)", "Stacking (isotônica)"]]
    # O artigo não dá número aqui, diz que os stackings "converge to the same F1".
    confere("§4.4", "espalhamento do F1 dos stackings com limiar in-fold (artigo: ~0)", None,
            espalhamento.max() - espalhamento.min())


# ------------------------------------------------------------- agregações entre modelos
def secao_otimismo():
    """A seção que exige o conjunto de 22 modelos do EX_032; ver o README."""
    mi = ler("ex033_metricas_individuais_media.csv")
    me = ler("ex033_metricas_ensembles_media.csv")
    media_todos = pd.concat([mi, me], ignore_index=True)
    todos = sorted(media_todos["modelo"].unique())
    conjuntos = {22: [m for m in todos if m not in U.ENSEMBLES_NOVOS], 25: todos}
    rhos = {22: [], 25: []}

    # Na base padronizada o artigo escreve +0,022, que é o 0,0215 impresso pelo notebook
    # arredondado de novo; o valor exato é 0,021500, que em três casas dá 0,021. Para não
    # acusar uma divergência que é só de arredondamento, a conferência é em quatro casas.
    ESPERADO_ALEATORIO = {"original": (0.042, 0.041, 3), "padronizada": (0.0215, 0.024, 4)}
    for base, (df1, dauc, casas_f1) in ESPERADO_ALEATORIO.items():
        tabela = U.comparar(media_todos.query("base == @base"), "protocolo", "scaffold", "aleatorio")
        for n, modelos in conjuntos.items():
            sub = tabela.loc[[m for m in tabela.index if m in modelos]]
            rhos[n].append(spearmanr(sub["F1 (scaffold)"], sub["F1 (aleatorio)"]).statistic)
            confere("§4.5", f"Δ F1 médio aleatório − scaffold ({base}, {n} modelos)",
                    df1 if n == 22 else None, sub["Δ F1 (aleatorio − scaffold)"].mean(),
                    casas_f1 if n == 22 else 3)
            confere("§4.5", f"Δ ROC-AUC médio aleatório − scaffold ({base}, {n} modelos)",
                    dauc if n == 22 else None, sub["Δ ROC-AUC (aleatorio − scaffold)"].mean())

    for protocolo in ("scaffold", "aleatorio"):
        tabela = U.comparar(media_todos.query("protocolo == @protocolo"), "base",
                            "original", "padronizada")
        for n, modelos in conjuntos.items():
            sub = tabela.loc[[m for m in tabela.index if m in modelos]]
            rhos[n].append(spearmanr(sub["F1 (original)"], sub["F1 (padronizada)"]).statistic)
            if protocolo == "scaffold":
                confere("§4.5", f"Δ F1 médio padronizada − original (scaffold, {n} modelos)",
                        -0.016 if n == 22 else None, sub["Δ F1 (padronizada − original)"].mean())
                confere("§4.5", f"modelos que caem com a padronização ({n} modelos)",
                        19 if n == 22 else None,
                        int((sub["Δ F1 (padronizada − original)"] < 0).sum()), 0)
    for n in (22, 25):
        confere("§4.2", f"Spearman mínimo entre cenários ({n} modelos)",
                0.948 if n == 22 else None, min(rhos[n]))

    # posição do Graph-FP e maior gap treino-teste
    ranking = ler("ex033_ranking_padronizada_scaffold.csv")
    coluna = "posição (22 do artigo)" if "posição (22 do artigo)" in ranking else None
    if coluna:
        confere("§4.2", "posição do Graph-FP entre os 22", 15,
                float(ranking.loc[ranking["modelo"] == "Graph-FP", coluna].iloc[0]), 0)
    individuais = ler("ex033_metricas_individuais_media.csv").query(
        "base == 'padronizada' and protocolo == 'scaffold'")
    gap = individuais.pivot_table(index="modelo", columns="parte", values="f1_mean")
    gap["gap"] = gap["treino"] - gap["teste"]
    confere("§4.2", "gap treino−teste do Graph-FP", 0.211, gap.loc["Graph-FP", "gap"])
    confere("§4.2", "Graph-FP é o maior gap (1 = sim)", 1, int(gap["gap"].idxmax() == "Graph-FP"), 0)


# ------------------------------------------------------------- colapso, ablação e externo
def secao_colapso_ablacao_externo():
    c = ler("ex033_colapso_inchikey_delta.csv").set_index(c0 := "modelo") \
        if "modelo" in ler("ex033_colapso_inchikey_delta.csv") else None
    linha = c.loc[MELHOR_INDIVIDUAL]
    por_linhas = float(str(linha["Δ F1 (padronizada − linhas)"]).split("±")[0])
    colapsada = float(str(linha["Δ F1 (padronizada − colapsada)"]).split("±")[0])
    confere("§4.5", "Δ F1 do MolFormer-XL contra as linhas", -0.027, por_linhas)
    confere("§4.5", "Δ F1 do MolFormer-XL contra a base colapsada", -0.019, colapsada)
    confere("§4.5", "fração do drop explicada pela contagem", 0.30,
            (por_linhas - colapsada) / por_linhas, 2)

    s = ler("ex033_ablacao_spearman.csv")
    numeros = s.select_dtypes("number")
    confere("§4.6", "Spearman mínimo entre as políticas de rótulo", 0.977, numeros.min().min())
    ganho = ler("ex033_ablacao_ganho_ensemble.csv")
    confere("§4.6", "políticas em que o ensemble perde (de 12)", 0,
            int((ganho["Δ (ensemble − individual)"] <= 0).sum()), 0)

    ext = ler("ex033_externo_triagem.csv").query(
        "externo == 'externo padronizada' and corte == 'completo'").set_index("modelo")
    confere("§4.7", "compostos no externo padronizado", 1057,
            int(ext.loc[MELHOR_INDIVIDUAL, "compostos"]), 0)
    confere("§4.7", "positivos no externo padronizado", 32,
            int(ext.loc[MELHOR_INDIVIDUAL, "positivos"]), 0)
    confere("§4.7", "prevalência do externo padronizado (%)", 3.03,
            100 * ext.loc[MELHOR_INDIVIDUAL, "prevalência"], 2)
    for coluna, (ens, ind), casas in (("Precision@100", (0.15, 0.13), 2),
                                      ("Recall@100", (0.469, 0.406), 3),
                                      ("BEDROC (α=20)", (0.346, 0.306), 3)):
        confere("§4.7", f"{coluna} do ensemble", ens, ext.loc["Seleção gulosa", coluna], casas)
        confere("§4.7", f"{coluna} do MolFormer-XL", ind, ext.loc[MELHOR_INDIVIDUAL, coluna], casas)

    inedito = ler("ex033_externo_triagem.csv").query(
        "externo == 'externo padronizada' and corte == 'scaffold inédito'").set_index("modelo")
    confere("§4.7", "compostos no corte de scaffold inédito", 426,
            int(inedito.loc[MELHOR_INDIVIDUAL, "compostos"]), 0)
    confere("§4.7", "positivos no corte de scaffold inédito", 14,
            int(inedito.loc[MELHOR_INDIVIDUAL, "positivos"]), 0)

    freq = ler("ex033_frequencia_gulosa_criterio_padronizada.csv").set_index(
        ler("ex033_frequencia_gulosa_criterio_padronizada.csv").columns[0])
    for modelo in ("MolFormer-XL", "GIN"):
        confere("§4.5", f"frequência de {modelo} no Greedy (AP)", 1.0, freq.loc[modelo, "PR-AUC"], 2)


def main():
    for secao in (secao_dados, secao_split, secao_tabelas, secao_bootstrap, secao_limiar,
                  secao_otimismo, secao_colapso_ablacao_externo):
        secao()
    tabela = pd.DataFrame(conferencias)
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.max_colwidth", 62):
        print(tabela.to_string(index=False))
    checadas = tabela[tabela["confere"] != "—"]
    divergem = checadas[checadas["confere"] == "DIVERGE"]
    print(f"\n{len(checadas) - len(divergem)} de {len(checadas)} números do artigo conferem; "
          f"{len(tabela) - len(checadas)} valores impressos sem alvo no texto.")
    if len(divergem):
        print("\nDivergências:")
        print(divergem.to_string(index=False))
    return 1 if len(divergem) else 0


if __name__ == "__main__":
    sys.exit(main())
