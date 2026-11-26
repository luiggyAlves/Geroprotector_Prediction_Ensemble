"""Gera o notebook do benchmark completo (EX_032 + EX_033), o que reproduz o artigo.

É a junção dos dois experimentos num notebook só: o EX_033 já repetia o EX_032 inteiro, e o
que faltava para fechar com o artigo era o conjunto de modelos das agregações entre cenários.
O EX_032 agregava 22 modelos (18 representações + 4 ensembles); o EX_033 acrescentou três
ensembles e passou a agregar 25, o que muda as médias entre modelos e o Spearman das seções 7,
8 e 11 — e o artigo cita os números de 22. Aqui as duas versões são calculadas, impressas e
salvas lado a lado, com o conjunto explícito em cada linha.

Do EX_033, que é o EX_032 com as limitações do artigo endereçadas, vem:

* stacking calibrado dentro do fold, isotônica e Platt (limitação vi);
* seleção gulosa por PR-AUC, sem limiar (limitação v);
* bootstrap por scaffold e teste de Nadeau-Bengio na base padronizada (limitação i);
* colapso por InChIKey das predições da base não padronizada (limitação iii);
* ablação das três políticas para os 82 conflitos de rótulo (limitação iv).

Funções em ex033_utils.py, que importa o ex032_utils e o ex027_utils.

Uso:
    python utils/regressao_logistica/build_ex032_033.py
"""

import json
import os

NOTEBOOK = "EX_032_033_benchmark_completo.ipynb"
cells = []


def md(texto):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": texto.strip("\n")})


def code(texto):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                  "source": texto.strip("\n")})


# =========================================================================================
md("""
# Benchmark completo (EX_032 + EX_033) — 18 representações congeladas, cinco ensembles, quatro protocolos de avaliação

Este notebook é a junção do EX_032 e do EX_033 e é a base dos números do artigo. O EX_033 já repetia o EX_032 inteiro: os mesmos 18 encoders, as mesmas 15 folds e os mesmos 1.080 checkpoints, sem reajustar nenhum membro — as seções 1 a 12 são o EX_032, a seção 5 ganha três ensembles novos e as seções 13 a 16 fecham cinco das oito limitações listadas no artigo.

**O que a junção acrescenta, e por que ela é necessária para reproduzir o artigo.** As seções 7, 8 e 11 agregam *entre modelos*: média do Δ de cada modelo e Spearman entre os rankings. O resultado depende de quais modelos entram na conta, e os dois experimentos usam conjuntos diferentes — o EX_032 tem 22 modelos (18 representações + 4 ensembles) e o EX_033 tem 25, porque acrescentou Seleção gulosa (PR-AUC), Stacking (Platt) e Stacking (isotônica). Os números do artigo (Δ F1 de +0,042 e +0,022, Δ ROC-AUC de +0,041 e +0,024, a queda de −0,016 com 19 de 22 modelos piorando, e o Spearman mínimo de 0,948) são os de **22 modelos**. Rodar só o EX_033 dá +0,047, +0,024, −0,021 e Spearman mínimo de 0,862 — correto para 25 modelos, e diferente do texto.

Aqui as duas agregações são calculadas e impressas lado a lado, identificadas por conjunto, e salvas em `ex033_agregados_por_conjunto.csv`. Nada mais muda: as métricas por modelo, os ensembles, os intervalos de confiança e o conjunto externo são os do EX_033, modelo a modelo.

| Limitação do artigo | Onde é endereçada |
|---|---|
| (i) Sem IC nem teste pareado na base padronizada | Seção 13 |
| (iii) Causa da queda da padronização não isolada | Seção 15 |
| (iv) Os 82 conflitos de rótulo sem ablação | Seção 16 |
| (v) Circularidade entre seleção gulosa e estudo do limiar | Seções 5 e 14 |
| (vi) Stacking descalibrado | Seções 5 e 14 |

As limitações (ii) MAT e R-MAT, (vii) o conjunto externo não virgem e (viii) extrator fixo com classificador linear não são endereçadas aqui: a primeira depende de um pacote indisponível no ambiente, e as outras duas não se resolvem por computação.

Pipeline completo, baseado no EX_027:

1. **Padronização química** da base de treino, removendo as duplicatas estruturais.
2. **Embeddings** de 18 encoders para a base padronizada, com checkpoint em disco. MAT e R-MAT, do EX_027, ficam de fora porque o pacote que os gera (`huggingmolecules`) não está disponível no ambiente. A base não padronizada usa os embeddings do EX_027.
3. **Regressão logística L2** por representação, com `C` escolhido por CV interna aninhada, em 5 folds × 3 seeds (42, 43, 44) = 15 folds. Métricas no limiar 0,5, em treino e teste.
4. **Ensembles** com as 18 representações: soft voting, soft voting ponderado por ROC-AUC e stacking.
5. **Avaliação dos ensembles** nas mesmas 15 folds.
6. **Seleção gulosa** do ensemble, decidida só com a parte de treino de cada fold, comparada ao melhor modelo individual.
7. **Ranking** por F1.
8. **Scaffold x aleatório** em cada base.
9. **Diversidade** do ensemble guloso (mapas de calor).
10. **Estudo do limiar** entre 0,4 e 0,6.
11. **Base padronizada x não padronizada.**
12. **Validação externa:** o melhor individual e o melhor ensemble do split por scaffold, treinados com a base inteira, avaliados uma vez no conjunto externo (Precision@K, Recall@K e BEDROC), nas versões padronizada e não padronizada. O EX_033 **não lê o conjunto externo de novo**: os modelos finais não mudaram, então as predições do EX_032 são recarregadas do disco.
13. **Bootstrap por scaffold e teste de Nadeau–Bengio** na base padronizada (limitação i).
14. **Calibração do stacking e seleção gulosa sem limiar** (limitações vi e v).
15. **Colapso por InChIKey** das predições da base não padronizada (limitação iii).
16. **Ablação dos conflitos de rótulo** (limitação iv).

Etapas 3 a 8 rodam nas duas bases e em dois protocolos de divisão: **scaffold** (esqueletos de Bemis–Murcko nunca aparecem em treino e teste ao mesmo tempo) e **aleatório estratificado**. As análises da seleção gulosa, da diversidade e do limiar (6, 9 e 10) usam só o split por scaffold, e o melhor modelo individual é o de maior F1 médio de teste no split por scaffold.
""")

md("## Configuração")
code("""
import os
import sys
import json
import importlib
import subprocess

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.stats import spearmanr
from sklearn.metrics import f1_score

# ---- caminhos: este notebook roda de notebooks/ ------------------------------------------
ROOT = os.path.abspath("..")                            # desenvolvimento_modelos/
sys.path.insert(0, os.path.join(ROOT, "utils", "regressao_logistica"))
import ex033_utils as U
U = importlib.reload(U)   # kernel reaproveitado não pega edições do módulo sem o reload

BASE_ORIGINAL = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas",
                             "Base_GeroLLM_reconstruida", "primary_dataset_dedup.csv")
DIR_BASE = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas",
                        "Base_GeroLLM_padronizada")
BASE_PADRONIZADA = os.path.join(DIR_BASE, "primary_padronizada.csv")
QUIMICA_ORIGINAL = os.path.join(DIR_BASE, "padronizacao_linhas_originais.csv")
QUIMICA_PADRONIZADA = os.path.join(DIR_BASE, "padronizacao_base_padronizada.csv")

# Pastas dos embeddings: um .npy por encoder. A base não padronizada usa os do EX_027.
DIR_EMBEDDINGS = {
    "padronizada": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM_padronizada"),
    "original": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM"),
}
SCRIPT_EMBEDDINGS = os.path.join(ROOT, "utils", "embeddings", "gerar_embeddings_base.py")
# Os encoders rodam no env `embeddings` (torch, transformers, torch_geometric, gensim,
# selfies, unimol_tools); este notebook roda no env PIBIC.
PYTHON_EMBEDDINGS = os.path.expanduser("~/miniconda3/envs/embeddings/bin/python")

RESULTS_DIR = os.path.join(ROOT, "resultados")
checkpoints = U.Checkpoints(os.path.join(RESULTS_DIR, "checkpoints", "checkpoints_ex032"))
MODELOS_FINAIS = os.path.join(RESULTS_DIR, "modelos", "ex032_modelos_finais.joblib")


def out(nome):
    pasta = os.path.join(RESULTS_DIR, {".csv": "csv", ".png": "figuras"}[os.path.splitext(nome)[1]])
    os.makedirs(pasta, exist_ok=True)
    return os.path.join(pasta, nome)


# ---- protocolo ---------------------------------------------------------------------------
SEEDS = (42, 43, 44)
N_SPLITS = 5
C_GRID = np.logspace(-4, 2, 7)          # mesmo grid do EX_027
BASES = ["padronizada", "original"]
NOME_BASE = {"padronizada": "base padronizada", "original": "base não padronizada"}
PROTOCOLOS = ["scaffold", "aleatorio"]
NOME_PROTOCOLO = {"scaffold": "split por scaffold", "aleatorio": "split aleatório estratificado"}
CENARIOS = [(base, protocolo) for base in BASES for protocolo in PROTOCOLOS]
N_JOBS = 5                               # folds internos em paralelo

ENCODERS = {  # nome no estudo -> chave do arquivo de embedding
    "MolFormer-XL": "molformer", "SELFormer": "selformer", "ChemBERTa-2": "chembert",
    "ChemBERTa-3": "chemberta3", "MolDeBERTa": "moldeberta", "MolCLR": "molclr",
    "Mole-BERT": "molebert", "GIN": "gin", "GraphMVP": "graphmvp", "Uni-Mol": "unimol",
    "ECFP4-2048": "ecfp4", "Atom-Pair": "atompair", "Graph-FP": "graphfp", "TT": "tt",
    "Mol2Vec": "mol2vec", "ChemGPT": "chemgpt", "ChemFM": "chemfm", "CLAMP": "clamp",
}
# MAT e R-MAT (EX_027) ficam de fora: dependem do pacote huggingmolecules, que só existe no
# GitHub e não está instalado no env `embeddings`. Para incluí-los, instale o pacote e
# acrescente "MAT": "mat" e "R-MAT": "rmat" acima.
BINARIOS = {"ECFP4-2048", "Atom-Pair", "Graph-FP", "TT"}


def arquivos_embedding(base, chave):
    \"\"\"Caminhos do embedding e dos rótulos de um encoder em uma base.\"\"\"
    pasta = DIR_EMBEDDINGS[base]
    if base == "original" and chave == "ecfp4":      # no EX_027 o ECFP4 tem outro nome
        return os.path.join(pasta, "fingerprints.npy"), os.path.join(pasta, "rotulos_fingerprints.npy")
    return os.path.join(pasta, f"Embeddings_{chave}.npy"), os.path.join(pasta, f"rotulos_{chave}.npy")
""")

# ---------------------------------------------------------------------------- 1
md("""
## 1. Obtenção da base e padronização química

Cada SMILES passa pela padronização do EX_027 com RDKit: limpeza, remoção de sais e solventes (fragmento pai), neutralização e tautômero canônico. Linhas com o mesmo InChIKey padronizado são a mesma substância e viram **uma linha só**:

- o SMILES passa a ser o tautômero canônico padronizado;
- o rótulo é positivo se qualquer forma for positiva (mesma regra que a base usa para conflitos entre fontes).

A base não padronizada é a base original do EX_027, com as duplicatas estruturais. O resultado da padronização fica salvo em CSV; rodar de novo carrega o arquivo.
""")
code("""
os.makedirs(DIR_BASE, exist_ok=True)
original = pd.read_csv(BASE_ORIGINAL)
if os.path.exists(BASE_PADRONIZADA):
    padronizada = pd.read_csv(BASE_PADRONIZADA)
    quimica_original = pd.read_csv(QUIMICA_ORIGINAL)
    print(f"[checkpoint] base padronizada carregada de {os.path.basename(BASE_PADRONIZADA)}")
else:
    padronizada, quimica_original, _ = U.padronizar_base(original)
    padronizada.to_csv(BASE_PADRONIZADA, index=False)
    quimica_original.to_csv(QUIMICA_ORIGINAL, index=False)

resumo = {
    "linhas na base original": len(original),
    "substâncias distintas (base padronizada)": len(padronizada),
    "linhas removidas como duplicata": len(original) - len(padronizada),
    "substâncias que reuniam mais de uma linha": int((padronizada["n_formas"] > 1).sum()),
    "substâncias com rótulos conflitantes (viraram positivas)": int(padronizada["rotulo_conflitante"].sum()),
    "positivos antes / depois": f"{int(original['label'].sum())} / {int(padronizada['label'].sum())}",
    "prevalência depois": f"{padronizada['label'].mean():.2%}",
}
display(pd.Series(resumo, name="valor").to_frame())
""")
code("""
# Quantas duplicatas cada etapa da padronização revelou (mesma auditoria do EX_027)
display(U.duplicate_audit(quimica_original))
exemplos = padronizada.query("n_formas > 1").head(8)[["smiles", "smiles_original", "label", "n_formas",
                                                        "rotulo_conflitante", "name"]]
display(exemplos)
""")
code("""
# Grupos de scaffold, usados pelo split por scaffold. Na base não padronizada são os mesmos
# grupos do EX_027 (calculados linha a linha sobre a base original).
if os.path.exists(QUIMICA_PADRONIZADA):
    quimica = pd.read_csv(QUIMICA_PADRONIZADA)
else:
    quimica = U.chemistry_table(padronizada["smiles"].tolist())
    quimica.to_csv(QUIMICA_PADRONIZADA, index=False)
assert len(quimica) == len(padronizada) and len(quimica_original) == len(original)
repetidos = int(quimica["L5_inchikey"].duplicated().sum())
if repetidos:
    print(f"Aviso: {repetidos} compostos ainda colidem ao repadronizar o SMILES padronizado.")

Y = {"padronizada": padronizada["label"].astype(int).to_numpy(),
     "original": original["label"].astype(int).to_numpy()}
GRUPOS = {"padronizada": quimica["group"].to_numpy(),
          "original": quimica_original["group"].to_numpy()}
for base in BASES:
    print(f"{NOME_BASE[base]}: {len(Y[base])} compostos, {Y[base].sum()} positivos "
          f"({Y[base].mean():.2%}), {len(np.unique(GRUPOS[base]))} grupos de scaffold")
""")

# ---------------------------------------------------------------------------- 2
md("""
## 2. Embeddings

Um arquivo `.npy` por encoder na pasta de checkpoints da base padronizada. Ao rodar:

- se a pasta estiver vazia ou incompleta, os embeddings que faltam são gerados;
- se estiver completa, o notebook **pergunta** se os embeddings devem ser gerados de novo (`s`) ou reaproveitados (Enter).

Para rodar sem pergunta (por exemplo com `nbconvert`), defina `REGERAR_EMBEDDINGS` como `True` ou `False`, ou a variável de ambiente `EX033_REGERAR=1`/`0`.

A geração roda no env `embeddings`, por um script separado (`utils/embeddings/gerar_embeddings_base.py`), com as mesmas funções que geraram os embeddings do EX_027. A base não padronizada usa esses embeddings do EX_027, que já estão em disco.
""")
code("""
REGERAR_EMBEDDINGS = None      # None = perguntar; True = gerar tudo de novo; False = reaproveitar

if os.environ.get("EX033_REGERAR") in ("0", "1"):
    REGERAR_EMBEDDINGS = os.environ["EX033_REGERAR"] == "1"


def embedding_pronto(base, chave):
    arquivo = arquivos_embedding(base, chave)[0]
    return os.path.exists(arquivo) and np.load(arquivo, mmap_mode="r").shape[0] == len(Y[base])


prontos = [n for n, chave in ENCODERS.items() if embedding_pronto("padronizada", chave)]
faltando = [n for n in ENCODERS if n not in prontos]
print(f"Pasta: {DIR_EMBEDDINGS['padronizada']}")
print(f"{len(prontos)} de {len(ENCODERS)} encoders com embeddings completos para a base padronizada.")

if REGERAR_EMBEDDINGS is None and prontos:
    resposta = input(f"Gerar os embeddings de novo? {len(prontos)} prontos, "
                     f"{len(faltando)} faltando. [s = gerar tudo / Enter = reaproveitar] ")
    REGERAR_EMBEDDINGS = resposta.strip().lower() in ("s", "sim", "y", "yes")

a_gerar = list(ENCODERS) if REGERAR_EMBEDDINGS else faltando
if a_gerar:
    comando = [PYTHON_EMBEDDINGS, SCRIPT_EMBEDDINGS, "--csv", BASE_PADRONIZADA,
               "--coluna", "smiles", "--rotulo", "label", "--saida", DIR_EMBEDDINGS["padronizada"],
               "--encoders", *a_gerar] + (["--forcar"] if REGERAR_EMBEDDINGS else [])
    print("Gerando:", ", ".join(a_gerar))
    processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for linha in processo.stdout:
        if linha.startswith(("[ok]", "[erro]", "[checkpoint]", "Resumo")):
            print(linha, end="")
    processo.wait()
else:
    print("Reaproveitando os embeddings da pasta.")
""")
code("""
# Só entram os encoders com embedding nas duas bases, para que a comparação use as mesmas
# representações.
NOMES = [n for n, chave in ENCODERS.items() if all(embedding_pronto(b, chave) for b in BASES)]
ausentes = [n for n in ENCODERS if n not in NOMES]
if ausentes:
    print(f"ATENÇÃO: {len(ausentes)} encoder(s) sem embedding e fora das próximas etapas: {ausentes}")

X = {base: {} for base in BASES}
for base in BASES:
    for n in NOMES:
        arquivo, arquivo_rotulos = arquivos_embedding(base, ENCODERS[n])
        X[base][n] = np.load(arquivo).astype(np.float32)
        rotulos = np.load(arquivo_rotulos).astype(int)
        assert X[base][n].shape[0] == len(Y[base]) and (rotulos == Y[base]).all(), (base, n)
display(pd.DataFrame({NOME_BASE[b]: {n: X[b][n].shape[1] for n in NOMES} for b in BASES}).T)
""")

# ---------------------------------------------------------------------------- 3
md("""
## 3. Regressão logística L2 por representação

- **Modelo:** regressão logística com penalidade L2 (soma dos quadrados dos coeficientes), `class_weight="balanced"` e padronização dos dados; nos fingerprints, bits presentes em menos de 1% dos compostos são removidos antes.
- **Escolha de C:** dentro de cada fold externo, uma CV interna de 5 folds (só com a parte de treino) escolhe `C` no grid 10⁻⁴ a 10² pelo ROC-AUC.
- **Validação:** 5 folds × 3 seeds = 15 folds, nos dois protocolos e nas duas bases.
- **Métricas:** accuracy, recall, precision, F1, balanced accuracy e MCC no limiar 0,5, mais ROC-AUC e AP (área sob a curva precision-recall), que não dependem de limiar, em treino e teste.

Antes dos ajustes, uma verificação explícita do split por scaffold: em cada fold, conta quantos grupos de scaffold aparecem no treino e no teste ao mesmo tempo. No scaffold esse número tem de ser zero (o notebook para se não for); no aleatório ele mostra quantos esqueletos o teste compartilha com o treino.

Cada ajuste fica salvo em checkpoint.
""")
code("""
folds = {}
for base in BASES:
    folds[(base, "aleatorio")] = U.repeated_random_folds(Y[base], n_splits=N_SPLITS, seeds=SEEDS)
    folds[(base, "scaffold")] = U.repeated_scaffold_folds(Y[base], GRUPOS[base], n_splits=N_SPLITS,
                                                          seeds=SEEDS)
indices = {c: {(s, k): (tr, te) for s, k, tr, te in folds[c]} for c in CENARIOS}

# Verificação de vazamento de scaffold: nenhum esqueleto pode estar em treino e teste no scaffold
verificacao = pd.concat([U.scaffolds_compartilhados(folds[(b, p)], GRUPOS[b]).assign(
    base=NOME_BASE[b], protocolo=NOME_PROTOCOLO[p]) for b, p in CENARIOS], ignore_index=True)
verificacao.to_csv(out("ex033_verificacao_scaffold.csv"), index=False)
resumo = verificacao.groupby(["base", "protocolo"])[["scaffolds no teste", "scaffolds em treino e teste"]].agg(["mean", "std", "max"])
print("Scaffolds compartilhados entre treino e teste, por fold (15 folds)")
display(resumo.round(2))
no_scaffold = verificacao[verificacao["protocolo"] == NOME_PROTOCOLO["scaffold"]]
assert no_scaffold["scaffolds em treino e teste"].eq(0).all(), "vazamento de scaffold"
print("OK: no split por scaffold, nenhum esqueleto aparece em treino e teste ao mesmo tempo.")

membros = {}
for base, protocolo in CENARIOS:
    membros[(base, protocolo)] = U.rodar_membros(
        X[base], Y[base], folds[(base, protocolo)], NOMES, BINARIOS, C_GRID, checkpoints, protocolo,
        grupos=GRUPOS[base] if protocolo == "scaffold" else None, n_jobs=N_JOBS,
        prefixo=f"{base}_{protocolo}")
    print(f"{NOME_BASE[base]}, {protocolo}: {len(membros[(base, protocolo)])} modelos "
          f"(15 folds x {len(NOMES)} representações)")

metricas_individuais = pd.concat(
    [U.tabela_metricas(Y[b], indices[(b, p)], membros[(b, p)], p, b) for b, p in CENARIOS],
    ignore_index=True)
metricas_individuais.to_csv(out("ex033_metricas_individuais_por_fold.csv"), index=False)
""")
code("""
def mostrar_por_fold(tabela, modelo):
    \"\"\"Métricas de um modelo em cada fold, treino e teste, e a média e o desvio-padrão das 15 folds.\"\"\"
    for base, protocolo in CENARIOS:
        parte = tabela.query("base == @base and protocolo == @protocolo and modelo == @modelo")
        largo = parte.pivot_table(index=["seed", "fold"], columns="parte", values=U.METRICAS)
        largo.columns = [f"{U.NOMES_METRICAS[m]} ({p})" for m, p in largo.columns]
        largo = largo[[f"{U.NOMES_METRICAS[m]} ({p})" for m in U.METRICAS for p in ("treino", "teste")]]
        media, desvio = largo.mean(), largo.std()
        largo.loc[("média", ""), :] = media
        largo.loc[("desvio-padrão", ""), :] = desvio
        print(f"{modelo} — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]}")
        display(largo.round(4))


for nome in NOMES:
    mostrar_por_fold(metricas_individuais, nome)
""")
code("""
media_individuais = U.media_por_modelo(metricas_individuais)
media_individuais.to_csv(out("ex033_metricas_individuais_media.csv"), index=False)
for base, protocolo in CENARIOS:
    for parte in ("teste", "treino"):
        selecao = (media_individuais.query("base == @base and protocolo == @protocolo and parte == @parte")
                   .set_index("modelo").sort_values("f1_mean", ascending=False))
        tabela = pd.DataFrame({U.NOMES_METRICAS[m]: U.media_dp(selecao[f"{m}_mean"], selecao[f"{m}_std"])
                               for m in U.METRICAS})
        print(f"Média ± desvio-padrão das 15 folds — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]} — {parte}")
        display(tabela)
""")
md("""
### 3.1 Modelos finais com todos os dados

Cada representação é treinada com a base padronizada inteira (a base não padronizada serve só para comparação e não ganha modelos finais). O `C` e as probabilidades out-of-fold que alimentam os ensembles finais (pesos, meta-modelo e seleção gulosa) vêm de uma CV de 5 folds por scaffold sobre a base inteira. Os modelos ficam salvos para uso nos ensembles e em predições futuras.
""")
code("""
if os.path.exists(MODELOS_FINAIS):
    finais = joblib.load(MODELOS_FINAIS)
    print(f"[checkpoint] modelos finais carregados de {os.path.basename(MODELOS_FINAIS)}")
else:
    finais = U.treinar_finais(X["padronizada"], Y["padronizada"], NOMES, BINARIOS, C_GRID,
                              GRUPOS["padronizada"], n_jobs=N_JOBS)
    os.makedirs(os.path.dirname(MODELOS_FINAIS), exist_ok=True)
    joblib.dump(finais, MODELOS_FINAIS)
display(pd.DataFrame({"C": finais["C"], "peso no soft ponderado": finais["pesos_roc_auc"]}).round(4))
print("Membros da seleção gulosa final:", finais["membros_gulosos"])
""")

# ---------------------------------------------------------------------------- 4
md("""
## 4. Ensembles com as 18 representações

1. **Soft voting:** média das probabilidades dos modelos.
2. **Soft voting ponderado por ROC-AUC:** média ponderada pelo ROC-AUC de cada modelo na CV interna (parte de treino), com os pesos somando 1.
3. **Stacking:** meta-modelo (regressão logística L2, `C=1`, `class_weight="balanced"`) treinado nas probabilidades out-of-fold internas dos modelos, ou seja, em predições de compostos que cada modelo não viu.

O EX_033 acrescenta três ensembles:

4. **Stacking (isotônica)** e **5. Stacking (Platt):** o mesmo meta-modelo, com um calibrador ajustado **dentro do fold**, nas probabilidades out-of-fold internas do próprio meta-modelo. O `class_weight="balanced"` desloca as probabilidades do stacking para cima, e no limiar 0,5 ele classifica positivo demais: tem o maior ROC-AUC entre os ensembles e o menor F1. A calibração é uma transformação monótona, então **ROC-AUC e PR-AUC não mudam** — o que muda é onde o limiar 0,5 cai. A isotônica não impõe forma funcional; a de Platt ajusta uma sigmoide de um parâmetro e é mais estável com poucos positivos.
6. **Seleção gulosa (PR-AUC):** a mesma busca gulosa, mas maximizando a PR-AUC das probabilidades médias em vez do F1 no limiar 0,5. A versão por F1 deixa o escore interno otimizado em 0,5, o que torna o estudo do limiar circular para esse ensemble (seção 10). A PR-AUC não depende de limiar e é a métrica de ranking mais próxima do uso em triagem, com prevalência de 17%.

Em todos, pesos, meta-modelo, calibrador e subconjunto são definidos só com a parte de treino de cada fold.
""")

# ---------------------------------------------------------------------------- 5
md("""
## 5. Avaliação dos ensembles nas 15 folds

As mesmas 15 folds e as mesmas métricas dos modelos individuais, no limiar 0,5, em treino e teste, nas duas bases e nos dois protocolos. A seleção gulosa (seção 6) é calculada junto.
""")
code("""
ensembles = {}
for base, protocolo in CENARIOS:
    cenario = (base, protocolo)
    ensembles[cenario] = {}
    for (seed, k), (tr, te) in indices[cenario].items():
        registros = {n: membros[cenario][(seed, k, n)] for n in NOMES}
        for nome, registro in U.ensembles_do_fold(Y[base], tr, registros, NOMES, seed).items():
            ensembles[cenario][(seed, k, nome)] = registro

metricas_ensembles = pd.concat(
    [U.tabela_metricas(Y[b], indices[(b, p)], ensembles[(b, p)], p, b) for b, p in CENARIOS],
    ignore_index=True)
metricas_ensembles.to_csv(out("ex033_metricas_ensembles_por_fold.csv"), index=False)

for nome in U.ENSEMBLES_033:
    mostrar_por_fold(metricas_ensembles, nome)
""")
code("""
media_ensembles = U.media_por_modelo(metricas_ensembles)
media_ensembles.to_csv(out("ex033_metricas_ensembles_media.csv"), index=False)
for base, protocolo in CENARIOS:
    selecao = media_ensembles.query("base == @base and protocolo == @protocolo").set_index(["parte", "modelo"])
    tabela = pd.DataFrame({f"{U.NOMES_METRICAS[m]} ({p})": U.media_dp(selecao.loc[p, f"{m}_mean"],
                                                                     selecao.loc[p, f"{m}_std"])
                           for m in U.METRICAS for p in ("teste", "treino")})
    print(f"Média ± desvio-padrão das 15 folds — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]}")
    display(tabela)
""")

# ---------------------------------------------------------------------------- 6
md("""
## 6. Seleção gulosa (split por scaffold)

Dentro de cada fold, a busca adiciona uma representação por vez ao soft voting, escolhendo a que mais aumenta o F1 (limiar 0,5) nas probabilidades out-of-fold internas da parte de treino, e fica com o melhor passo. O teste do fold não é usado na escolha.

A análise usa o split por scaffold. A referência é o **melhor modelo individual no split por scaffold** (maior F1 médio de teste nas 15 folds), em cada base:

- a tabela compara o ensemble guloso com esse modelo no teste;
- no gráfico do caminho, a linha tracejada é o F1 interno médio desse modelo, medido nas mesmas probabilidades out-of-fold internas que guiam a busca.
""")
code("""
melhor_individual = {}
for base in BASES:
    teste = media_individuais.query("base == @base and protocolo == 'scaffold' and parte == 'teste'")
    melhor_individual[base] = teste.set_index("modelo")["f1_mean"].idxmax()
    print(f"Melhor individual no split por scaffold — {NOME_BASE[base]}: {melhor_individual[base]}")

frequencias, consenso = {}, {}
for base in BASES:
    cenario = (base, "scaffold")
    frequencias[base] = U.frequencia_selecao(ensembles[cenario], NOMES)
    tamanhos = [len(r["membros"]) for (s, k, m), r in ensembles[cenario].items() if m == "Seleção gulosa"]
    print(f"{NOME_BASE[base]}: {len(tamanhos)} folds, tamanho mediano {np.median(tamanhos):.0f} "
          f"(de {min(tamanhos)} a {max(tamanhos)} membros)")
    consenso[base] = frequencias[base].query("frequencia >= 0.5").index.tolist()
tabela = pd.concat({NOME_BASE[b]: frequencias[b]["frequencia"] for b in BASES}, axis=1)
tabela.to_csv(out("ex033_frequencia_gulosa_scaffold.csv"))
display(tabela.sort_values(NOME_BASE["padronizada"], ascending=False).round(3))
for base in BASES:
    print(f"Membros de consenso ({NOME_BASE[base]}, escolhidos em >= 50% das folds): {consenso[base]}")
""")
code("""
linhas = []
for base in BASES:
    for modelo, tipo in ((melhor_individual[base], "melhor individual"), ("Seleção gulosa", "ensemble guloso")):
        fonte = media_ensembles if modelo == "Seleção gulosa" else media_individuais
        linha = fonte.query("base == @base and protocolo == 'scaffold' and parte == 'teste' "
                            "and modelo == @modelo").iloc[0]
        linhas.append({"base": NOME_BASE[base], "tipo": tipo, "modelo": modelo,
                       **{U.NOMES_METRICAS[m]: f"{linha[f'{m}_mean']:.4f} ± {linha[f'{m}_std']:.4f}"
                          for m in U.METRICAS}})
guloso_x_individual = pd.DataFrame(linhas)
guloso_x_individual.to_csv(out("ex033_guloso_x_melhor_individual_scaffold.csv"), index=False)
print("Teste, split por scaffold (média ± desvio-padrão das 15 folds)")
display(guloso_x_individual)
""")
code("""
fig, axes = plt.subplots(1, len(BASES), figsize=(7 * len(BASES), 4), sharey=True)
for ax, base in zip(axes, BASES):
    cenario = (base, "scaffold")
    for (seed, k, nome), registro in ensembles[cenario].items():
        if nome == "Seleção gulosa":
            caminho = registro["caminho"]
            ax.plot(caminho["n_membros"], caminho["f1_interno"], color="tab:blue", alpha=0.35)
    f1_interno_melhor = np.mean([
        f1_score(Y[base][tr], (membros[cenario][(s, k, melhor_individual[base])]["inner_oof"]
                               >= U.LIMIAR_PADRAO).astype(int), zero_division=0)
        for (s, k), (tr, te) in indices[cenario].items()])
    ax.axhline(f1_interno_melhor, color="tab:red", ls="--", lw=1.5,
               label=f"melhor individual ({melhor_individual[base]})")
    ax.set_title(f"Caminho da seleção gulosa — {NOME_BASE[base]}, scaffold")
    ax.set_xlabel("número de membros")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
axes[0].set_ylabel("F1 interno (parte de treino)")
fig.tight_layout()
fig.savefig(out("ex033_caminho_guloso_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------- 7
md("""
## 7. Ranking

Modelos individuais, os três ensembles com todas as representações e os ensembles gulosos, ordenados pelo F1 médio de teste das 15 folds (limiar 0,5), em cada base e protocolo.

A coluna **posição** é entre os 25 modelos; **posição (22 do artigo)** é entre os 22 do EX_032, que é a que o artigo cita. Elas divergem para quem está abaixo de um dos três ensembles novos: o Graph-FP, por exemplo, é 15º entre os 22 e 16º entre os 25.
""")
code("""
media_todos = pd.concat([media_individuais, media_ensembles], ignore_index=True)

# Conjuntos de modelos das agregações entre modelos (seções 7, 8 e 11). O artigo cita os
# números de 22 modelos, que é o conjunto do EX_032; o de 25 inclui os três ensembles que o
# EX_033 acrescentou. Os dois são calculados em toda agregação, e o conjunto vai na saída.
TODOS_OS_MODELOS = sorted(media_todos["modelo"].unique())
MODELOS_22 = [m for m in TODOS_OS_MODELOS if m not in U.ENSEMBLES_NOVOS]
CONJUNTOS = {"22 (do artigo)": MODELOS_22, "25 (completo)": TODOS_OS_MODELOS}
agregados = []        # preenchido nas seções 8 e 11, salvo no fim da seção 11
for rotulo, modelos in CONJUNTOS.items():
    print(f"conjunto {rotulo}: {len(modelos)} modelos")
assert len(MODELOS_22) == 22 and len(TODOS_OS_MODELOS) == 25, "conjunto de modelos inesperado"

rankings = {}
for base, protocolo in CENARIOS:
    teste = media_todos.query("base == @base and protocolo == @protocolo and parte == 'teste'").set_index("modelo")
    ranking = pd.DataFrame({U.NOMES_METRICAS[m]: U.media_dp(teste[f"{m}_mean"], teste[f"{m}_std"])
                            for m in U.METRICAS})
    ranking["tipo"] = ["ensemble" if m in U.ENSEMBLES_033 else "individual" for m in ranking.index]
    ranking = ranking.loc[teste["f1_mean"].sort_values(ascending=False).index]
    ranking.insert(0, "posição", range(1, len(ranking) + 1))
    # Posição dentro dos 22 do artigo: é a que o texto cita (o Graph-FP, por exemplo, é 15º
    # entre os 22 e 16º entre os 25).
    nos_22 = [m for m in ranking.index if m in MODELOS_22]
    ranking.insert(1, "posição (22 do artigo)",
                   [nos_22.index(m) + 1 if m in MODELOS_22 else None for m in ranking.index])
    rankings[(base, protocolo)] = ranking
    ranking.to_csv(out(f"ex033_ranking_{base}_{protocolo}.csv"))
    print(f"Ranking por F1 de teste — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]} "
          f"(média ± desvio-padrão das 15 folds)")
    display(ranking)
""")

# ---------------------------------------------------------------------------- 8
md("""
## 8. Scaffold x aleatório

F1 e ROC-AUC de teste de cada modelo nos dois protocolos (média ± desvio-padrão das 15 folds), em cada base, e a diferença aleatório − scaffold. No split aleatório, compostos com o mesmo esqueleto caem em treino e teste, então o teste é mais parecido com o treino; a diferença mede quanto do desempenho depende de já ter visto esqueletos parecidos. As folds dos dois protocolos são diferentes, então a comparação é das médias, não fold a fold, e o desvio do Δ é √(dp_scaffold² + dp_aleatório²). O Δ médio entre modelos vem com o desvio-padrão entre os modelos.

A correlação de Spearman entre os F1 dos dois protocolos diz se a ordem dos modelos se mantém. O Δ médio e o Spearman são agregações entre modelos, então saem nos dois conjuntos (22 e 25); o artigo cita os de 22.
""")
code("""
comparacao_protocolos = {}
for base in BASES:
    tabela = U.comparar(media_todos.query("base == @base"), "protocolo", "scaffold", "aleatorio")
    tabela = tabela.sort_values("F1 (scaffold)", ascending=False)
    tabela.insert(0, "tipo", ["ensemble" if m in U.ENSEMBLES_033 else "individual" for m in tabela.index])
    comparacao_protocolos[base] = tabela
    tabela.to_csv(out(f"ex033_scaffold_x_aleatorio_{base}.csv"))
    print(f"{NOME_BASE[base]} — teste, média ± desvio-padrão das 15 folds")
    for rotulo, modelos in CONJUNTOS.items():
        sub = tabela.loc[[m for m in tabela.index if m in modelos]]
        rho = spearmanr(sub["F1 (scaffold)"], sub["F1 (aleatorio)"]).statistic
        delta_f1 = sub["Δ F1 (aleatorio − scaffold)"]
        delta_auc = sub["Δ ROC-AUC (aleatorio − scaffold)"]
        print(f"  [{len(sub)} modelos, conjunto {rotulo}] "
              f"Δ F1 médio entre modelos (aleatório − scaffold): {delta_f1.mean():+.4f} ± {delta_f1.std():.4f}; "
              f"Δ ROC-AUC médio: {delta_auc.mean():+.4f} ± {delta_auc.std():.4f}; "
              f"Spearman entre os F1: {rho:.3f}")
        agregados.append({"comparação": "aleatório − scaffold", "fixo": NOME_BASE[base],
                          "conjunto": rotulo, "modelos": len(sub),
                          "Δ F1 médio": delta_f1.mean(), "Δ F1 dp entre modelos": delta_f1.std(),
                          "Δ ROC-AUC médio": delta_auc.mean(),
                          "Δ ROC-AUC dp entre modelos": delta_auc.std(),
                          "modelos com Δ F1 < 0": int((delta_f1 < 0).sum()), "Spearman": rho})
    display(U.formatar_dp(tabela))
""")
code("""
fig, axes = plt.subplots(1, len(BASES), figsize=(8 * len(BASES), 9), sharex=True)
for ax, base in zip(axes, BASES):
    teste = media_todos.query("base == @base and parte == 'teste'")
    ordem = comparacao_protocolos[base].index[::-1]
    posicoes = np.arange(len(ordem))
    for deslocamento, protocolo, cor in ((-0.2, "scaffold", "tab:blue"), (0.2, "aleatorio", "tab:orange")):
        valores = teste.query("protocolo == @protocolo").set_index("modelo").loc[ordem]
        ax.barh(posicoes + deslocamento, valores["f1_mean"], height=0.4, xerr=valores["f1_std"],
                color=cor, label=NOME_PROTOCOLO[protocolo], error_kw={"lw": 0.8})
    ax.set_yticks(posicoes)
    ax.set_yticklabels(ordem, fontsize=8)
    ax.set_xlabel("F1 de teste (média ± desvio-padrão das 15 folds)")
    ax.set_title(f"Scaffold x aleatório — {NOME_BASE[base]}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out("ex033_scaffold_x_aleatorio.png"), dpi=150, bbox_inches="tight")
plt.show()
""")

# ---------------------------------------------------------------------------- 9
md("""
## 9. Diversidade do ensemble guloso (split por scaffold)

Medidas par a par entre os membros de consenso do ensemble guloso no split por scaffold (escolhidos em pelo menos metade das folds), calculadas nas predições de teste da repetição de seed 42, no limiar 0,5 (cada composto é previsto uma vez, por um modelo que não o viu):

- **Double-fault:** fração de compostos que os dois erram (menor = mais complementares).
- **Q-statistic:** 0 = erros independentes, 1 = erros coincidentes.
- **Disagreement:** fração de compostos em que os dois discordam (maior = mais diversos).
- **Correlation coefficient:** correlação entre os vetores de acerto.
""")
code("""
for base in BASES:
    cenario = (base, "scaffold")
    membros_div = consenso[base]
    if len(membros_div) < 2:
        print(f"{NOME_BASE[base]}: menos de 2 membros de consenso, sem mapa de diversidade.")
        continue
    acertos = U.acertos_oof(Y[base], indices[cenario], membros[cenario], membros_div, seed=SEEDS[0])
    fig, axes = plt.subplots(1, 4, figsize=(5 * 4, 4.4))
    for ax, (titulo, medida) in zip(axes, U.MEDIDAS_DIVERSIDADE.items()):
        matriz = U.pairwise_matrix(medida, acertos, membros_div)
        imagem = ax.imshow(matriz, cmap="viridis")
        ax.set_xticks(range(len(membros_div)))
        ax.set_xticklabels(membros_div, rotation=90, fontsize=8)
        ax.set_yticks(range(len(membros_div)))
        ax.set_yticklabels(membros_div, fontsize=8)
        for i in range(len(membros_div)):
            for j in range(len(membros_div)):
                if i != j:
                    ax.text(j, i, f"{matriz[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w")
        ax.set_title(titulo)
        fig.colorbar(imagem, ax=ax, fraction=0.046)
    fig.suptitle(f"Diversidade do ensemble guloso — {NOME_BASE[base]}, split por scaffold")
    fig.tight_layout()
    fig.savefig(out(f"ex033_diversidade_{base}_scaffold.png"), dpi=150, bbox_inches="tight")
    plt.show()
""")

# ---------------------------------------------------------------------------- 10
md("""
## 10. Estudo do limiar (split por scaffold)

Até aqui todas as métricas usaram o limiar 0,5. Aqui o melhor modelo individual e o melhor ensemble **no split por scaffold** (maior F1 médio de teste) são avaliados na grade de limiares de 0,40 a 0,60 (passo 0,01), em cada base:

- **Curva OOF (linha cheia):** F1 médio das 15 folds nas probabilidades out-of-fold internas da parte de treino, em cada limiar.
- **Curva de teste (linha tracejada):** F1 médio das 15 folds no teste, em cada limiar. Só para visualização; não entra na escolha.
- **Limiar escolhido:** um por modelo, o pico da curva OOF.
- **Círculos:** as duas curvas em 0,5. **Estrelas:** as duas curvas no limiar escolhido.

A tabela traz exatamente os valores dos círculos e das estrelas, com o desvio-padrão das 15 folds. As folds são as mesmas nos dois limiares, então o desvio do Δ é o desvio da diferença fold a fold. O teste não entra na escolha, mas o OOF de uma fold vem de um treino que contém compostos de teste de outras folds. Como o limiar é um único escalar, esse vazamento é leve.
""")
code("""
resultados_limiar = []
fig, axes = plt.subplots(1, len(BASES), figsize=(8 * len(BASES), 4.5), sharey=True)
for ax, base in zip(axes, BASES):
    cenario = (base, "scaffold")
    teste = media_todos.query("base == @base and protocolo == 'scaffold' and parte == 'teste'")
    teste = teste.set_index("modelo")["f1_mean"]
    melhor_ensemble = teste[[m for m in U.ENSEMBLES_033 if m in teste.index]].idxmax()

    for modelo, cor in ((melhor_individual[base], "tab:blue"), (melhor_ensemble, "tab:red")):
        fonte = ensembles[cenario] if modelo in U.ENSEMBLES_033 else {
            (s, k, n): {**r, "inner_score": r["inner_oof"]} for (s, k, n), r in membros[cenario].items()}
        probas = {(s, k): r for (s, k, n), r in fonte.items() if n == modelo}
        por_fold, curva, limiar = U.estudo_limiar(Y[base], indices[cenario], probas)
        ax.plot(curva.index, curva["f1_oof_mean"], ls="-", color=cor, label=f"{modelo} — OOF")
        ax.plot(curva.index, curva["f1_teste_mean"], ls="--", color=cor,
                label=f"{modelo} — teste (só visualização)")
        ax.scatter([0.5, 0.5], curva.loc[0.5, ["f1_oof_mean", "f1_teste_mean"]], s=70,
                   facecolors="none", edgecolors=cor, linewidths=2, zorder=3)
        ax.scatter([limiar, limiar], curva.loc[limiar, ["f1_oof_mean", "f1_teste_mean"]], s=160,
                   marker="*", color=cor, edgecolors="black", linewidths=0.6, zorder=4)
        por_limiar = por_fold.set_index(["limiar", "seed", "fold"]).sort_index()
        linha = {"base": NOME_BASE[base], "modelo": modelo, "limiar escolhido (pico OOF)": limiar}
        for coluna, nome in (("f1_oof", "F1 OOF"), ("f1_teste", "F1 teste")):
            delta = por_limiar.loc[limiar, coluna] - por_limiar.loc[0.5, coluna]
            linha.update({
                f"{nome} em 0,5": curva.loc[0.5, f"{coluna}_mean"],
                f"{nome} em 0,5 dp": curva.loc[0.5, f"{coluna}_std"],
                f"{nome} no limiar escolhido": curva.loc[limiar, f"{coluna}_mean"],
                f"{nome} no limiar escolhido dp": curva.loc[limiar, f"{coluna}_std"],
                f"Δ {nome}": delta.mean(), f"Δ {nome} dp": delta.std()})
        resultados_limiar.append(linha)
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.scatter([], [], s=70, facecolors="none", edgecolors="gray", linewidths=2,
               label="limiar 0,5")
    ax.scatter([], [], s=160, marker="*", color="gray", edgecolors="black", linewidths=0.6,
               label="limiar escolhido (pico OOF)")
    ax.set_xlabel("limiar")
    ax.set_title(f"F1 em função do limiar — {NOME_BASE[base]}, scaffold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("F1 médio (15 folds)")
fig.tight_layout()
fig.savefig(out("ex033_estudo_limiar_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()

tabela_limiar = pd.DataFrame(resultados_limiar)
tabela_limiar.to_csv(out("ex033_estudo_limiar_scaffold.csv"), index=False)
print("Média ± desvio-padrão das 15 folds")
display(U.formatar_dp(tabela_limiar))
""")

# ---------------------------------------------------------------------------- 11
md("""
## 11. Base padronizada x não padronizada

Primeiro, as oito métricas de teste de todos os modelos em cada base e protocolo (quatro tabelas, média ± desvio-padrão das 15 folds, ordenadas por F1). Depois, F1 e ROC-AUC de cada modelo nas duas bases, lado a lado, e a diferença padronizada − não padronizada. As bases não têm os mesmos compostos (a padronizada une as duplicatas estruturais), então a comparação é das médias, não composto a composto, e o desvio do Δ é √(dp_padronizada² + dp_original²). O Δ médio entre modelos vem com o desvio-padrão entre os modelos.

Como na seção 8, o Δ médio entre modelos e o Spearman saem nos dois conjuntos de modelos, e o artigo cita os de 22 — inclusive a contagem de quantos modelos pioram com a padronização, que é 19 de 22 (e 22 de 25).

Na base não padronizada, formas da mesma substância (sal e base livre, formas carregada e neutra, tautômeros) são linhas diferentes. No split aleatório elas podem cair uma no treino e outra no teste, e o modelo é avaliado em um composto que já viu. A tabela de vazamento conta esses casos nos dois protocolos; no split por scaffold ele quase não acontece, porque as formas da mesma substância quase sempre têm o mesmo esqueleto e ficam no mesmo grupo.
""")
code("""
for base, protocolo in CENARIOS:
    print(f"Teste — {NOME_BASE[base]} — {NOME_PROTOCOLO[protocolo]} (média ± desvio-padrão das 15 folds)")
    display(rankings[(base, protocolo)])
""")
code("""
chaves = quimica_original["L5_inchikey"].to_numpy()
validas = pd.notna(chaves)
vazamento = []
for protocolo in PROTOCOLOS:
    for seed, k, tr, te in folds[("original", protocolo)]:
        no_treino = set(chaves[tr][validas[tr]])
        vazamento.append({"protocolo": NOME_PROTOCOLO[protocolo], "seed": seed, "fold": k,
                          "compostos de teste com a mesma substância no treino":
                              int(sum(c in no_treino for c in chaves[te][validas[te]])),
                          "compostos de teste": len(te)})
vazamento = pd.DataFrame(vazamento)
vazamento["fração"] = vazamento["compostos de teste com a mesma substância no treino"] / vazamento["compostos de teste"]
colunas = ["compostos de teste com a mesma substância no treino", "compostos de teste", "fração"]
resumo = vazamento.groupby("protocolo")[colunas].agg(["mean", "std"])
print("Vazamento de substâncias duplicadas na base não padronizada (média ± desvio-padrão das 15 folds)")
display(pd.DataFrame({c: U.media_dp(resumo[(c, "mean")], resumo[(c, "std")]) for c in colunas}))
""")
code("""
comparacao_bases = {}
for protocolo in PROTOCOLOS:
    tabela = U.comparar(media_todos.query("protocolo == @protocolo"), "base", "original", "padronizada")
    tabela = tabela.sort_values("F1 (padronizada)", ascending=False)
    tabela.insert(0, "tipo", ["ensemble" if m in U.ENSEMBLES_033 else "individual" for m in tabela.index])
    comparacao_bases[protocolo] = tabela
    tabela.to_csv(out(f"ex033_padronizada_x_original_{protocolo}.csv"))
    print(f"{NOME_PROTOCOLO[protocolo]} — teste, média ± desvio-padrão das 15 folds")
    for rotulo, modelos in CONJUNTOS.items():
        sub = tabela.loc[[m for m in tabela.index if m in modelos]]
        rho = spearmanr(sub["F1 (original)"], sub["F1 (padronizada)"]).statistic
        delta_f1 = sub["Δ F1 (padronizada − original)"]
        delta_auc = sub["Δ ROC-AUC (padronizada − original)"]
        print(f"  [{len(sub)} modelos, conjunto {rotulo}] "
              f"Δ F1 médio entre modelos (padronizada − original): {delta_f1.mean():+.4f} ± {delta_f1.std():.4f}; "
              f"Δ ROC-AUC médio: {delta_auc.mean():+.4f} ± {delta_auc.std():.4f}; "
              f"Spearman entre os F1: {rho:.3f}; "
              f"modelos que caem: {int((delta_f1 < 0).sum())}/{len(sub)}")
        agregados.append({"comparação": "padronizada − original", "fixo": NOME_PROTOCOLO[protocolo],
                          "conjunto": rotulo, "modelos": len(sub),
                          "Δ F1 médio": delta_f1.mean(), "Δ F1 dp entre modelos": delta_f1.std(),
                          "Δ ROC-AUC médio": delta_auc.mean(),
                          "Δ ROC-AUC dp entre modelos": delta_auc.std(),
                          "modelos com Δ F1 < 0": int((delta_f1 < 0).sum()), "Spearman": rho})
    display(U.formatar_dp(tabela))

# As quatro comparações nos dois conjuntos, na ordem em que o artigo as cita.
agregados = pd.DataFrame(agregados)
agregados.to_csv(out("ex033_agregados_por_conjunto.csv"), index=False)
print("\nAgregações entre modelos nos dois conjuntos (o artigo cita as linhas de 22 modelos)")
display(agregados.round(4))
""")
code("""
fig, axes = plt.subplots(1, len(PROTOCOLOS), figsize=(8 * len(PROTOCOLOS), 9), sharex=True)
for ax, protocolo in zip(axes, PROTOCOLOS):
    teste = media_todos.query("protocolo == @protocolo and parte == 'teste'")
    ordem = comparacao_bases[protocolo].index[::-1]
    posicoes = np.arange(len(ordem))
    for deslocamento, base, cor in ((-0.2, "padronizada", "tab:green"), (0.2, "original", "tab:gray")):
        valores = teste.query("base == @base").set_index("modelo").loc[ordem]
        ax.barh(posicoes + deslocamento, valores["f1_mean"], height=0.4, xerr=valores["f1_std"],
                color=cor, label=NOME_BASE[base], error_kw={"lw": 0.8})
    ax.set_yticks(posicoes)
    ax.set_yticklabels(ordem, fontsize=8)
    ax.set_xlabel("F1 de teste (média ± desvio-padrão das 15 folds)")
    ax.set_title(f"Padronizada x não padronizada — {NOME_PROTOCOLO[protocolo]}")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="lower right")
fig.tight_layout()
fig.savefig(out("ex033_padronizada_x_original.png"), dpi=150, bbox_inches="tight")
plt.show()
""")
code("""
# Resumo: melhor individual e melhor ensemble de cada cenário (F1 de teste, média ± desvio-padrão das 15 folds)
linhas = []
for base, protocolo in CENARIOS:
    teste = media_todos.query("base == @base and protocolo == @protocolo and parte == 'teste'").set_index("modelo")
    individual = teste.drop(U.ENSEMBLES_033, errors="ignore")["f1_mean"].idxmax()
    ensemble = teste.loc[[m for m in U.ENSEMBLES_033 if m in teste.index], "f1_mean"].idxmax()
    linhas.append({"base": NOME_BASE[base], "protocolo": NOME_PROTOCOLO[protocolo],
                   "melhor individual": individual,
                   "F1 do melhor individual": teste.loc[individual, "f1_mean"],
                   "F1 do melhor individual dp": teste.loc[individual, "f1_std"],
                   "melhor ensemble": ensemble,
                   "F1 do melhor ensemble": teste.loc[ensemble, "f1_mean"],
                   "F1 do melhor ensemble dp": teste.loc[ensemble, "f1_std"]})
resumo_cenarios = pd.DataFrame(linhas)
resumo_cenarios.to_csv(out("ex033_resumo_cenarios.csv"), index=False)
display(U.formatar_dp(resumo_cenarios))
""")

# ---------------------------------------------------------------------------- 12
md("""
## 12. Validação externa

O conjunto externo (1.557 compostos, 35 positivos) não entra em nenhum treino nem em nenhuma escolha deste experimento. Ele é avaliado **uma vez**, sem k-fold, com modelos treinados com a base de treino inteira:

- **Modelos:** o melhor individual e o melhor ensemble do split por scaffold (maior F1 médio de teste), em cada base. Os modelos finais seguem a seção 3.1: `C`, membros da seleção gulosa, pesos e meta-modelo vêm de uma CV de 5 folds por scaffold sobre a base inteira. A base não padronizada ganha aqui os seus modelos finais, treinados do mesmo jeito.
- **Duas versões do externo:**
  - *não padronizada*: os SMILES originais, avaliados pelos modelos da base não padronizada, com os embeddings do EX_027;
  - *padronizada*: os SMILES passam pela mesma padronização da seção 1 (formas da mesma substância viram uma linha, rótulo positivo se qualquer forma for positiva), com embeddings gerados agora, e são avaliados pelos modelos da base padronizada.
- **Sobreposição removida:** compostos do externo que são a mesma substância (mesmo InChIKey padronizado) de algum composto de treino saem da avaliação, porque o modelo já os viu.
- **Cortes:** *completo* (tudo menos a sobreposição) e *scaffold inédito* (só compostos cujo esqueleto de Bemis–Murcko não aparece na base de treino).
- **Métricas de triagem**, livres de limiar: Precision@K e Recall@K para K = 50, 100 e 200, e BEDROC (α = 20). A prevalência do externo (≈ 2%) é muito menor que a do treino (17%), então essas métricas comparam a ordem dos compostos, não o corte em 0,5.

As predições ficam salvas em CSV. Rodar de novo carrega o arquivo e não lê o externo outra vez; cada leitura nova é registrada no log de leituras do conjunto externo (`resultados/json/external_read_log_ex026.json`). O EX_025, o EX_026 e o EX_027 já leram esse conjunto, então ele não é mais um lock-box virgem.
""")
code("""
BASE_EXTERNA = os.path.join(ROOT, "dados", "Bases de dados", "Treinamento - limpas",
                            "Base_GeroLLM_reconstruida", "external_test_set.csv")
EXTERNA_PADRONIZADA = os.path.join(DIR_BASE, "externo_padronizado.csv")
DIR_EMBEDDINGS_EXTERNO = {
    "original": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM_testeExterno"),
    "padronizada": os.path.join(ROOT, "dados", "Embeddings_salvos", "Embeddings_GeroLLM_testeExterno_padronizada"),
}
MODELOS_FINAIS_ORIGINAL = os.path.join(RESULTS_DIR, "modelos", "ex032_modelos_finais_original.joblib")
PREDICOES_EXTERNO = out("ex032_externo_predicoes.csv")   # mesma leitura do EX_032: os modelos finais não mudaram
LOG_EXTERNO = os.path.join(RESULTS_DIR, "json", "external_read_log_ex026.json")

# Versões do externo e a química de cada uma (InChIKey e grupo de scaffold)
externo = {"original": pd.read_csv(BASE_EXTERNA)}
if os.path.exists(EXTERNA_PADRONIZADA):
    externo["padronizada"] = pd.read_csv(EXTERNA_PADRONIZADA)
else:
    externo["padronizada"], _, _ = U.padronizar_base(externo["original"])
    externo["padronizada"].to_csv(EXTERNA_PADRONIZADA, index=False)
quimica_treino = {"original": quimica_original, "padronizada": quimica}
quimica_externo = {b: U.chemistry_table(externo[b]["smiles"].astype(str).tolist()) for b in BASES}

cortes = {b: U.external_cuts(quimica_treino[b], quimica_externo[b]) for b in BASES}
display(pd.DataFrame({NOME_BASE[b]: {
    "compostos no externo": len(externo[b]),
    "positivos no externo": int(externo[b]["label"].sum()),
    "mesma substância do treino (removidos)": cortes[b]["n_same_substance_removed"],
    "corte completo": int(cortes[b]["full"].sum()),
    "positivos no corte completo": int(externo[b]["label"].to_numpy()[cortes[b]["full"]].sum()),
    "corte scaffold inédito": int(cortes[b]["novel_scaffold"].sum()),
    "positivos no corte scaffold inédito": int(externo[b]["label"].to_numpy()[cortes[b]["novel_scaffold"]].sum()),
} for b in BASES}))
""")
code("""
# Modelos finais da base não padronizada (a padronizada já tem os seus, seção 3.1)
if os.path.exists(MODELOS_FINAIS_ORIGINAL):
    finais_original = joblib.load(MODELOS_FINAIS_ORIGINAL)
    print(f"[checkpoint] modelos finais carregados de {os.path.basename(MODELOS_FINAIS_ORIGINAL)}")
else:
    finais_original = U.treinar_finais(X["original"], Y["original"], NOMES, BINARIOS, C_GRID,
                                       GRUPOS["original"], n_jobs=N_JOBS)
    joblib.dump(finais_original, MODELOS_FINAIS_ORIGINAL)
modelos_finais = {"padronizada": finais, "original": finais_original}

# Melhor individual e melhor ensemble do split por scaffold, em cada base
avaliados = {}
for base in BASES:
    teste = media_todos.query("base == @base and protocolo == 'scaffold' and parte == 'teste'").set_index("modelo")["f1_mean"]
    avaliados[base] = [melhor_individual[base], teste[[m for m in U.ENSEMBLES if m in teste.index]].idxmax()]
    print(f"{NOME_BASE[base]}: {avaliados[base]}; membros da seleção gulosa final: "
          f"{modelos_finais[base]['membros_gulosos']}")
""")
code("""
# Embeddings do externo: só as representações que os modelos avaliados usam
necessarios = {b: sorted({n for m in avaliados[b] for n in U.encoders_do_modelo(modelos_finais[b], m)}) for b in BASES}


def arquivo_externo(base, chave):
    if base == "original" and chave == "ecfp4":      # nome do EX_027
        return os.path.join(DIR_EMBEDDINGS_EXTERNO[base], "fingerprints.npy")
    return os.path.join(DIR_EMBEDDINGS_EXTERNO[base], f"Embeddings_{chave}.npy")


def externo_pronto(base, nome):
    arquivo = arquivo_externo(base, ENCODERS[nome])
    return os.path.exists(arquivo) and np.load(arquivo, mmap_mode="r").shape[0] == len(externo[base])


faltando = [n for n in necessarios["padronizada"] if not externo_pronto("padronizada", n)]
if faltando:
    comando = [PYTHON_EMBEDDINGS, SCRIPT_EMBEDDINGS, "--csv", EXTERNA_PADRONIZADA, "--coluna", "smiles",
               "--rotulo", "label", "--saida", DIR_EMBEDDINGS_EXTERNO["padronizada"], "--encoders", *faltando]
    print("Gerando embeddings do externo padronizado:", ", ".join(faltando))
    processo = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for linha in processo.stdout:
        if linha.startswith(("[ok]", "[erro]", "[checkpoint]", "Resumo")):
            print(linha, end="")
    processo.wait()

X_externo = {}
for base in BASES:
    ausentes = [n for n in necessarios[base] if not externo_pronto(base, n)]
    assert not ausentes, f"embeddings do externo faltando ({base}): {ausentes}"
    X_externo[base] = {n: np.load(arquivo_externo(base, ENCODERS[n])).astype(np.float32) for n in necessarios[base]}
    for n in necessarios[base]:        # embeddings na mesma ordem e com os mesmos rótulos do CSV
        pasta, nome_arquivo = os.path.split(arquivo_externo(base, ENCODERS[n]))
        nome_rotulos = ("rotulos_fingerprints.npy" if nome_arquivo == "fingerprints.npy"
                        else nome_arquivo.replace("Embeddings_", "rotulos_"))
        rotulos = np.load(os.path.join(pasta, nome_rotulos)).astype(int)
        assert (rotulos == externo[base]["label"].to_numpy()).all(), (base, n)
    print(f"{NOME_BASE[base]}: {len(necessarios[base])} representações do externo carregadas")
""")
code("""
# Leitura única do externo: as predições ficam salvas e não são refeitas
if os.path.exists(PREDICOES_EXTERNO):
    predicoes_externo = pd.read_csv(PREDICOES_EXTERNO)
    print(f"[checkpoint] predições do externo carregadas de {os.path.basename(PREDICOES_EXTERNO)} (sem nova leitura)")
else:
    partes = []
    for base in BASES:
        parte = pd.DataFrame({"base": base, "linha": np.arange(len(externo[base])),
                              "label": externo[base]["label"].astype(int).to_numpy(),
                              "corte_completo": cortes[base]["full"],
                              "corte_scaffold_inedito": cortes[base]["novel_scaffold"]})
        for modelo in avaliados[base]:
            parte[modelo] = U.prever_modelo(modelos_finais[base], X_externo[base], modelo)
        partes.append(parte)
    predicoes_externo = pd.concat(partes, ignore_index=True)
    predicoes_externo.to_csv(PREDICOES_EXTERNO, index=False)
    U.log_external_read(LOG_EXTERNO, "ex032-modelos-finais",
                        note="EX_032: melhor individual e melhor ensemble do split por scaffold, "
                             "modelos finais das bases padronizada e não padronizada; externo nas "
                             "versões padronizada e não padronizada; leitura única, métricas de triagem")

linhas = []
for base in BASES:
    parte = predicoes_externo.query("base == @base")
    for corte, coluna in (("completo", "corte_completo"), ("scaffold inédito", "corte_scaffold_inedito")):
        selecao = parte[parte[coluna]]
        for modelo in avaliados[base]:
            linhas.append({"externo": NOME_BASE[base].replace("base", "externo"), "corte": corte,
                           "modelo": modelo, **U.metricas_triagem(selecao["label"].to_numpy(), selecao[modelo].to_numpy())})
externo_triagem = pd.DataFrame(linhas)
externo_triagem.to_csv(out("ex033_externo_triagem.csv"), index=False)
print("Conjunto externo, leitura única (sem k-fold)")
display(externo_triagem.round(4))
""")

# ---------------------------------------------------------------------------- 13
md("""
## 13. Intervalos de confiança e testes pareados (limitação i)

Até aqui tudo foi reportado como média ± desvio-padrão das 15 folds. Esse desvio descreve a **variabilidade entre partições**, não a incerteza da estimativa: as folds compartilham dados de treino e não são independentes, então ele não é intervalo de confiança e não sustenta afirmação de significância. Esta seção produz as duas peças que faltavam, com as funções do EX_027:

- **Bootstrap por scaffold.** Em uma repetição (seed 42), as 5 folds cobrem a base inteira e cada composto é previsto exatamente uma vez, por um modelo que não o viu. O bootstrap reamostra **grupos de scaffold inteiros**, não compostos: compostos do mesmo esqueleto não são observações independentes, e reamostrar linha a linha trata uma série congênere como vários sorteios independentes, o que estreita o intervalo sem informação nova. São 2.000 réplicas.
- **Bootstrap pareado.** Dois intervalos que se sobrepõem **não** implicam que a diferença seja nula, quando os modelos são correlacionados — e modelos avaliados nos mesmos compostos sempre são. A pergunta "A supera B" se responde no intervalo da diferença, com o mesmo reamostramento aplicado aos dois.
- **Teste t corrigido de Nadeau–Bengio.** Sobre as 15 folds, para as afirmações que só existem em forma de valores por fold. A correção troca o fator de variância 1/n por 1/n + n_teste/n_treino = 0,3167. O p ingênuo vai impresso ao lado, para mostrar o tamanho da diferença.

O cenário é o principal: **base padronizada, split por scaffold**. A referência é o melhor modelo individual.
""")
code("""
N_BOOT = 2000
MODELOS_IC = U.ENSEMBLES_033 + [melhor_individual["padronizada"]]
cenario_ic = ("padronizada", "scaffold")

proba_oof = U.oof_de_uma_repeticao(
    Y["padronizada"], indices[cenario_ic],
    {**membros[cenario_ic], **ensembles[cenario_ic]}, MODELOS_IC, seed=42)

intervalos = U.tabela_intervalos(Y["padronizada"], proba_oof, GRUPOS["padronizada"],
                                 MODELOS_IC, n_boot=N_BOOT)
intervalos.to_csv(out("ex033_intervalos_scaffold_padronizada.csv"), index=False)
largo = intervalos.assign(
    valor=lambda d: d.apply(lambda r: f"{r['point']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]", axis=1)
).pivot(index="modelo", columns="metric", values="valor")
print(f"IC95% por bootstrap de scaffold ({N_BOOT} réplicas), base padronizada, scaffold, seed 42")
display(largo)
""")
code("""
# Cada ensemble contra o melhor individual, com o mesmo reamostramento nos dois
pareado = U.tabela_pareada(Y["padronizada"], proba_oof, GRUPOS["padronizada"],
                           U.ENSEMBLES_033, melhor_individual["padronizada"], n_boot=N_BOOT)
pareado.to_csv(out("ex033_pareado_scaffold_padronizada.csv"), index=False)
resumo = pareado.assign(
    valor=lambda d: d.apply(lambda r: f"{r['difference']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]"
                                      + ("*" if r["excludes_zero"] else ""), axis=1)
).pivot(index="modelo", columns="metric", values="valor")
print(f"Ensemble − {melhor_individual['padronizada']}, bootstrap pareado por scaffold. "
      f"* = o intervalo exclui zero.")
display(resumo)
""")
code("""
# Nadeau-Bengio sobre as 15 folds, nos quatro cenários
por_fold = pd.concat([metricas_individuais, metricas_ensembles], ignore_index=True)
linhas = []
for base, protocolo in CENARIOS:
    parte = por_fold.query("base == @base and protocolo == @protocolo")
    tabela = U.tabela_nadeau_bengio(parte, U.ENSEMBLES_033, melhor_individual[base])
    linhas.append(tabela.assign(base=NOME_BASE[base], protocolo=NOME_PROTOCOLO[protocolo]))
nadeau = pd.concat(linhas, ignore_index=True)
nadeau.to_csv(out("ex033_nadeau_bengio.csv"), index=False)
print("Teste t corrigido de Nadeau-Bengio sobre as 15 folds (p ingênuo entre parênteses)")
display(nadeau.assign(
    p=lambda d: d.apply(lambda r: f"{r['p corrigido']:.3f} ({r['p ingênuo']:.3f})", axis=1)
).pivot_table(index=["base", "protocolo", "modelo"], columns="metrica", values="p", aggfunc="first"))
""")

# ---------------------------------------------------------------------------- 14
md("""
## 14. Calibração do stacking e seleção sem limiar (limitações vi e v)

### 14.1 O que a calibração muda

A calibração é monótona, então ROC-AUC e PR-AUC do stacking **não mudam**: a ordem dos compostos é a mesma. O que muda é onde o limiar 0,5 cai na distribuição de escores, e portanto todas as métricas que dependem do corte.

Aqui está o ponto que o diagnóstico do EX_032 não antecipou: **calibrar e continuar lendo o modelo em 0,5 piora o F1**, em vez de melhorar. O stacking descalibrado empurra as probabilidades para cima, então em 0,5 ele prevê positivo demais (recall alto, precision baixa). A calibração encolhe as probabilidades na direção da prevalência real, 17%; com isso quase nenhum composto passa de 0,5, o recall desaba e o F1 cai. As duas leituras são sintomas do mesmo problema: **0,5 não é o corte certo para esse modelo**, antes nem depois.

Por isso a seção avalia as duas coisas juntas. Para cada variante do stacking, o F1 é medido em 0,5 e no **limiar escolhido dentro do fold** — o que maximiza o F1 nas probabilidades out-of-fold internas da parte de treino, sobre uma grade larga de 0,05 a 0,95. A grade de 0,40 a 0,60 da seção 10 não serve aqui: o ótimo de um modelo calibrado fica muito abaixo dela. O teste do fold não entra na escolha.

A previsão a conferir na tabela: como a calibração é monótona, **os três stackings devem chegar ao mesmo F1 depois que o limiar é escolhido no fold**. Se isso acontecer, o diagnóstico "stacking descalibrado" se resolve em "stacking lido no limiar errado", e a calibração passa a valer pelo que ela de fato entrega — uma probabilidade interpretável e um limiar bem posto — e não por desempenho extra.

### 14.2 Seleção gulosa: F1 em 0,5 contra PR-AUC

A versão por F1 escolhe os membros maximizando o F1 no limiar 0,5, nas mesmas probabilidades out-of-fold que depois alimentam o estudo do limiar — por isso a curva OOF daquele ensemble tem um pico exatamente em 0,5 e o estudo não consegue avaliar outro limiar para ele. A versão por PR-AUC não usa limiar nenhum na escolha. Comparar as duas diz se a circularidade estava custando desempenho e se os membros escolhidos mudam.
""")
code("""
cenario_ic = ("padronizada", "scaffold")
alvos = ["Stacking", "Stacking (isotônica)", "Stacking (Platt)",
         "Seleção gulosa", "Seleção gulosa (PR-AUC)"]
tabela = media_ensembles.query("base == 'padronizada' and protocolo == 'scaffold' and parte == 'teste'").set_index("modelo")
display(pd.DataFrame({U.NOMES_METRICAS[m]: U.media_dp(tabela.loc[alvos, f"{m}_mean"],
                                                      tabela.loc[alvos, f"{m}_std"])
                      for m in U.METRICAS}))

# Fração de compostos classificados como positivos no limiar 0,5: o sintoma da descalibração
linhas = []
for nome in alvos:
    fracoes = [( ensembles[cenario_ic][(s, k, nome)]["test_proba"] >= U.LIMIAR_PADRAO).mean()
               for (s, k) in indices[cenario_ic]]
    linhas.append({"modelo": nome, "fração prevista positiva (teste)": np.mean(fracoes),
                   "fração prevista positiva dp": np.std(fracoes, ddof=1)})
fracao = pd.DataFrame(linhas)
fracao.to_csv(out("ex033_calibracao_scaffold_padronizada.csv"), index=False)
print(f"Prevalência real: {Y['padronizada'].mean():.2%}")
display(U.formatar_dp(fracao))
""")
code("""
# Limiar escolhido dentro do fold, grade larga: é isso que resolve a descalibração
por_fold_limiar = U.limiar_in_fold(Y["padronizada"], indices[cenario_ic],
                                   ensembles[cenario_ic], alvos)
por_fold_limiar.to_csv(out("ex033_limiar_in_fold_scaffold_padronizada.csv"), index=False)

agregado = por_fold_limiar.groupby("modelo").agg(["mean", "std"])
colunas = ["limiar escolhido"] + [f"{m}: {quando}" for m in ("f1", "precision", "recall")
                                  for quando in ("0,5", "limiar do fold")]
tabela = pd.DataFrame({c: U.media_dp(agregado[(c, "mean")], agregado[(c, "std")])
                       for c in colunas}).loc[alvos]
print("Limiar 0,5 contra o limiar escolhido no fold (média ± desvio-padrão das 15 folds)")
display(tabela)
print("Ganho de F1 ao escolher o limiar no fold:")
display((por_fold_limiar.assign(ganho=lambda d: d["f1: limiar do fold"] - d["f1: 0,5"])
         .groupby("modelo")["ganho"].agg(["mean", "std"]).loc[alvos].round(4)))
""")
code("""
# Membros escolhidos por cada critério, nas 15 folds
for base in BASES:
    cenario = (base, "scaffold")
    freq = pd.concat({
        "F1 em 0,5": U.frequencia_selecao(ensembles[cenario], NOMES, "Seleção gulosa")["frequencia"],
        "PR-AUC": U.frequencia_selecao(ensembles[cenario], NOMES, "Seleção gulosa (PR-AUC)")["frequencia"],
    }, axis=1)
    freq.to_csv(out(f"ex033_frequencia_gulosa_criterio_{base}.csv"))
    tamanhos = {c: [len(r["membros"]) for (s, k, m), r in ensembles[cenario].items() if m == c]
                for c in ("Seleção gulosa", "Seleção gulosa (PR-AUC)")}
    print(f"{NOME_BASE[base]}, scaffold — tamanho mediano: "
          + "; ".join(f"{c} {np.median(v):.0f} (de {min(v)} a {max(v)})" for c, v in tamanhos.items()))
    display(freq.sort_values("PR-AUC", ascending=False).round(3))
""")
code("""
# O estudo do limiar refeito para a seleção gulosa por PR-AUC, que não tem o pico forçado em 0,5
linhas = []
fig, axes = plt.subplots(1, len(BASES), figsize=(8 * len(BASES), 4.5), sharey=True)
for ax, base in zip(axes, BASES):
    cenario = (base, "scaffold")
    for modelo, cor in (("Seleção gulosa", "tab:red"), ("Seleção gulosa (PR-AUC)", "tab:green")):
        probas = {(s, k): r for (s, k, n), r in ensembles[cenario].items() if n == modelo}
        curva, limiar = U.estudo_limiar(Y[base], indices[cenario], probas)[1:]
        ax.plot(curva.index, curva["f1_oof_mean"], ls="-", color=cor, label=f"{modelo} — OOF")
        ax.plot(curva.index, curva["f1_teste_mean"], ls="--", color=cor,
                label=f"{modelo} — teste (só visualização)")
        ax.scatter([limiar], [curva.loc[limiar, "f1_oof_mean"]], s=160, marker="*", color=cor,
                   edgecolors="black", linewidths=0.6, zorder=4)
        linhas.append({"base": NOME_BASE[base], "critério": modelo,
                       "limiar escolhido (pico OOF)": limiar,
                       "F1 teste em 0,5": curva.loc[0.5, "f1_teste_mean"],
                       "F1 teste no limiar escolhido": curva.loc[limiar, "f1_teste_mean"]})
    ax.axvline(0.5, color="gray", ls=":", lw=1)
    ax.set_xlabel("limiar")
    ax.set_title(f"Limiar por critério de seleção — {NOME_BASE[base]}, scaffold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("F1 médio (15 folds)")
fig.tight_layout()
fig.savefig(out("ex033_limiar_por_criterio_scaffold.png"), dpi=150, bbox_inches="tight")
plt.show()

limiar_criterio = pd.DataFrame(linhas)
limiar_criterio.to_csv(out("ex033_limiar_por_criterio.csv"), index=False)
display(limiar_criterio.round(4))
""")

# ---------------------------------------------------------------------------- 15
md("""
## 15. Colapso por InChIKey (limitação iii)

No split por scaffold o vazamento de duplicatas já é quase zero (0,4 composto por fold), e mesmo assim a base padronizada perde cerca de 0,016 de F1 para a não padronizada. O EX_032 não isolou a causa. A hipótese mais plausível é **contagem**: na base não padronizada, uma substância presente em várias formas é pontuada várias vezes no mesmo fold de teste, e se essas substâncias forem as mais estudadas, e portanto mais fáceis, o F1 sobe sem que o modelo seja melhor.

O teste é direto e não exige reajustar nada: pegar as predições de teste da base **não padronizada** e agregá-las por InChIKey padronizado, com uma predição por substância (média das probabilidades) e o rótulo pela mesma regra "qualquer forma positiva" da padronização. Se o Δ para a base padronizada encolher, a queda era efeito de contagem. Se continuar, a causa é outra.
""")
code("""
chaves_original = quimica_original["L5_inchikey"].to_numpy()
MODELOS_COLAPSO = U.ENSEMBLES_033 + [melhor_individual["original"], melhor_individual["padronizada"]]
MODELOS_COLAPSO = list(dict.fromkeys(MODELOS_COLAPSO))

linhas = []
for seed in SEEDS:
    proba_orig = U.oof_de_uma_repeticao(
        Y["original"], indices[("original", "scaffold")],
        {**membros[("original", "scaffold")], **ensembles[("original", "scaffold")]},
        MODELOS_COLAPSO, seed=seed)
    y_col, proba_col = U.colapsar_por_inchikey(Y["original"], proba_orig, chaves_original)
    proba_pad = U.oof_de_uma_repeticao(
        Y["padronizada"], indices[("padronizada", "scaffold")],
        {**membros[("padronizada", "scaffold")], **ensembles[("padronizada", "scaffold")]},
        MODELOS_COLAPSO, seed=seed)
    for modelo in MODELOS_COLAPSO:
        for cenario, y_c, p_c in (("não padronizada (linhas)", Y["original"], proba_orig[modelo]),
                                  ("não padronizada (colapsada)", y_col, proba_col[modelo]),
                                  ("padronizada", Y["padronizada"], proba_pad[modelo])):
            linhas.append({"seed": seed, "modelo": modelo, "avaliação": cenario,
                           "compostos": len(y_c), **U.metricas(y_c, p_c)})
colapso = pd.DataFrame(linhas)
colapso.to_csv(out("ex033_colapso_inchikey_por_seed.csv"), index=False)

resumo_colapso = colapso.groupby(["modelo", "avaliação"])[["compostos"] + U.METRICAS].agg(["mean", "std"])
tabela = pd.DataFrame({c: U.media_dp(resumo_colapso[(c, "mean")], resumo_colapso[(c, "std")])
                       for c in ["f1", "roc_auc", "average_precision"]})
tabela.columns = [U.NOMES_METRICAS[c] for c in ["f1", "roc_auc", "average_precision"]]
tabela.insert(0, "compostos", resumo_colapso[("compostos", "mean")].round(0).astype(int))
print("Predições OOF do split por scaffold, média ± desvio-padrão das 3 repetições")
display(tabela)
""")
code("""
# O Δ padronizada − não padronizada, antes e depois do colapso
largo = colapso.pivot_table(index=["modelo", "seed"], columns="avaliação", values="f1")
deltas = pd.DataFrame({
    "Δ F1 (padronizada − linhas)": largo["padronizada"] - largo["não padronizada (linhas)"],
    "Δ F1 (padronizada − colapsada)": largo["padronizada"] - largo["não padronizada (colapsada)"],
}).groupby("modelo").agg(["mean", "std"])
tabela = pd.DataFrame({c: U.media_dp(deltas[(c, "mean")], deltas[(c, "std")])
                       for c in ["Δ F1 (padronizada − linhas)", "Δ F1 (padronizada − colapsada)"]})
tabela.to_csv(out("ex033_colapso_inchikey_delta.csv"))
print("Se a segunda coluna estiver perto de zero, a queda da padronização era efeito de contagem.")
display(tabela)
""")

# ---------------------------------------------------------------------------- 16
md("""
## 16. Ablação dos conflitos de rótulo (limitação iv)

A fusão por InChIKey junta formas da mesma substância que vinham com rótulos diferentes. São **82 substâncias**, todas resolvidas como positivas pela mesma regra que a base já usava para conflitos entre fontes. Essa escolha nunca foi testada na base padronizada. Aqui as três políticas são comparadas:

| Política | Compostos | Positivos | Prevalência |
|---|---|---|---|
| positivo (padrão do EX_032) | 3.752 | 637 | 16,98% |
| removido | 3.670 | 555 | 15,12% |
| negativo | 3.752 | 555 | 14,79% |

Diferente das outras seções, esta **reajusta os modelos**: mudar rótulos invalida os checkpoints. Para manter o custo em cerca de uma hora, a ablação roda só na base padronizada e só no split por scaffold, que é o cenário principal; são 2 políticas novas × 18 representações × 15 folds = 540 ajustes, em checkpoints próprios. A ablação não existe para a base não padronizada, onde essas formas são linhas separadas com seus próprios rótulos e não há conflito a resolver.

O que interessa não é o valor absoluto, que muda só pela prevalência, mas se **a ordem das representações e a vantagem do ensemble sobrevivem** à política de rótulo.
""")
code("""
checkpoints_ablacao = U.Checkpoints(os.path.join(RESULTS_DIR, "checkpoints", "checkpoints_ex033_ablacao"))
ablacao = {}
for politica in U.POLITICAS_CONFLITO:
    base_pol, mantidas = U.aplicar_politica_conflito(padronizada, politica)
    y_pol = base_pol["label"].astype(int).to_numpy()
    grupos_pol = GRUPOS["padronizada"][mantidas]
    X_pol = {n: X["padronizada"][n][mantidas] for n in NOMES}
    folds_pol = U.repeated_scaffold_folds(y_pol, grupos_pol, n_splits=N_SPLITS, seeds=SEEDS)
    indices_pol = {(s, k): (tr, te) for s, k, tr, te in folds_pol}
    print(f"[{politica}] {len(y_pol)} compostos, {y_pol.sum()} positivos ({y_pol.mean():.2%})")
    # A política "positivo" é a do EX_032: mesmos rótulos, mesmas folds, mesmos checkpoints.
    # Só as outras duas precisam de ajustes novos (2 x 18 x 15 = 540).
    reaproveita = politica == "positivo"
    membros_pol = U.rodar_membros(
        X_pol, y_pol, folds_pol, NOMES, BINARIOS, C_GRID,
        checkpoints if reaproveita else checkpoints_ablacao, "scaffold", grupos=grupos_pol,
        n_jobs=N_JOBS, prefixo="padronizada_scaffold" if reaproveita else f"ablacao_{politica}")
    ensembles_pol = {}
    for (seed, k), (tr, te) in indices_pol.items():
        registros = {n: membros_pol[(seed, k, n)] for n in NOMES}
        for nome, registro in U.ensembles_do_fold(y_pol, tr, registros, NOMES, seed).items():
            ensembles_pol[(seed, k, nome)] = registro
    ablacao[politica] = {"y": y_pol, "indices": indices_pol,
                         "metricas": pd.concat([
                             U.tabela_metricas(y_pol, indices_pol, membros_pol, "scaffold", politica),
                             U.tabela_metricas(y_pol, indices_pol, ensembles_pol, "scaffold", politica)],
                             ignore_index=True)}

metricas_ablacao = pd.concat([ablacao[p]["metricas"] for p in U.POLITICAS_CONFLITO], ignore_index=True)
metricas_ablacao.to_csv(out("ex033_ablacao_conflitos_por_fold.csv"), index=False)
media_ablacao = U.media_por_modelo(metricas_ablacao)
media_ablacao.to_csv(out("ex033_ablacao_conflitos_media.csv"), index=False)
""")
code("""
# A ordem das representações muda com a política?
teste_ablacao = media_ablacao.query("parte == 'teste'")
ordens = {p: teste_ablacao.query("base == @p").set_index("modelo")["f1_mean"] for p in U.POLITICAS_CONFLITO}
linhas = []
for metrica in ("f1", "roc_auc", "average_precision"):
    valores = {p: teste_ablacao.query("base == @p").set_index("modelo")[f"{metrica}_mean"]
               for p in U.POLITICAS_CONFLITO}
    comum = sorted(set.intersection(*(set(v.index) for v in valores.values())))
    for a, b in (("positivo", "removido"), ("positivo", "negativo")):
        rho = spearmanr(valores[a].loc[comum], valores[b].loc[comum]).statistic
        linhas.append({"métrica": U.NOMES_METRICAS[metrica], "comparação": f"{a} x {b}",
                       "Spearman": rho})
spearman_ablacao = pd.DataFrame(linhas).pivot(index="métrica", columns="comparação", values="Spearman")
spearman_ablacao.to_csv(out("ex033_ablacao_spearman.csv"))
print(f"Correlação de Spearman entre os rankings dos {len(comum)} modelos sob cada política")
display(spearman_ablacao.round(3))
""")
code("""
# A vantagem do ensemble sobrevive a cada política?
linhas = []
for politica in U.POLITICAS_CONFLITO:
    t = teste_ablacao.query("base == @politica").set_index("modelo")
    individuais = t.drop(U.ENSEMBLES_033, errors="ignore")
    melhor_ind = individuais["f1_mean"].idxmax()
    for metrica in ("f1", "roc_auc", "average_precision", "mcc"):
        melhor_ens = t.loc[[m for m in U.ENSEMBLES_033 if m in t.index], f"{metrica}_mean"].idxmax()
        linhas.append({"política": politica, "métrica": U.NOMES_METRICAS[metrica],
                       "melhor individual": melhor_ind,
                       "melhor ensemble": melhor_ens,
                       "Δ (ensemble − individual)": t.loc[melhor_ens, f"{metrica}_mean"]
                                                    - individuais[f"{metrica}_mean"].max()})
ganho_ablacao = pd.DataFrame(linhas)
ganho_ablacao.to_csv(out("ex033_ablacao_ganho_ensemble.csv"), index=False)
print("Se o Δ for positivo nas três políticas, a vantagem do ensemble não depende da regra de rótulo.")
display(ganho_ablacao.pivot(index="política", columns="métrica", values="Δ (ensemble − individual)").round(4))
display(ganho_ablacao)
""")

# =========================================================================================
for posicao, cell in enumerate(cells):
    cell["id"] = f"ex033-{posicao:03d}"
    cell["source"] = cell["source"].splitlines(keepends=True)

notebook = {
    "cells": cells,
    "metadata": {"kernelspec": {"display_name": "Python 3 (PIBIC)", "language": "python",
                                "name": "python3"},
                 "language_info": {"name": "python", "version": "3.14"}},
    "nbformat": 4, "nbformat_minor": 5,
}
destino = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "notebooks", NOTEBOOK))
with open(destino, "w") as arquivo:
    json.dump(notebook, arquivo, indent=1, ensure_ascii=False)
print(f"escrito {destino} ({len(cells)} células)")
