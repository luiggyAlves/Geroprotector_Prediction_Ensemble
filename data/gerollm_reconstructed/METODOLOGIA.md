# Reconstrução das bases de dados — Gero-LLM (Seção III-A)

Este pacote contém a reconstrução, a partir de fontes públicas reais, das duas
bases descritas na Seção III-A ("Dataset") do artigo *Gero-LLM: A Multimodal
Large Language Model for Geroprotector Discovery via Cross-Modal Differentiated
Mutual Learning*.

## 1. Base primária (`primary_dataset_dedup.csv` e `primary_train/val/test.csv`)

**Meta do artigo:** 4.107 moléculas únicas (769 geroprotetores / 3.338 não-geroprotetores,
razão ≈ 1:4.34), integrando DrugAge, Geroprotective Compounds Database
(geroprotectors.org), LOPAC, MicroSource Discovery Systems e os estudos [22]
Barardo et al. 2017, [23] Arora et al. 2025 (AgeXtend), [29] Wong et al. 2023
(DNN de senolíticos) e [30] Ye et al. 2014 (rede farmacológica em *C. elegans*).

**Resultado obtido:** **4.233 moléculas únicas** (727 positivas / 3.506 negativas,
razão 4.82:1).

### Fontes efetivamente coletadas e integradas

| Fonte no artigo | Fonte usada | Registros brutos | Como foi obtida |
|---|---|---|---|
| [23] AgeXtend (Arora et al. 2025) | `agextend_holy_aging.csv` | 972 | Clonado do repositório oficial `the-ahuja-lab/AgeXtend` (GitHub), arquivo `HOLY_AgingDataset.tsv`. Rótulos de origem (DrugAge, PMID 24134630/Ye et al. 2014, Geroprotective Compounds) preservados na coluna `sources`. |
| [22] Barardo et al. 2017 | `barardo2017_tableS1.csv` | 1.395 | Material suplementar oficial do artigo (`aging-09-1721-s001.xlsx`, Tabela S1), com SMILES reais extraídos da planilha. |
| [29] Wong et al. 2023 (senolíticos) | `wong2023_senolytics.csv` | 2.352 | Clonado do repositório oficial `felixjwong/senolyticsai` (GitHub), `notebook_files/train.csv`. |
| Geroprotectors.org | `geroprotectors_2024.csv` | 206 | Planilha "Geroprotectors_Clean_Descriptors_2024" com SMILES reais, obtida do seu diretório de pesquisa local. |

Todas as SMILES foram canonicalizadas via RDKit (`Chem.MolToSmiles(canonical=True)`)
e deduplicadas por SMILES canônica. Quando um composto aparecia com rótulo positivo
em qualquer fonte, foi mantido como positivo (109 conflitos resolvidos dessa forma,
consistente com a lógica de "lista positiva curada + fundo negativo" do artigo).
2 SMILES brutas não puderam ser interpretadas pelo RDKit e foram descartadas.

### Divisão 8:1:1 (estratificada, preservando a razão de classes)

| Split | Total | Positivos | Negativos | Razão |
|---|---|---|---|---|
| train | 3.387 | 582 | 2.805 | 4.82:1 |
| val | 424 | 73 | 351 | 4.81:1 |
| test | 422 | 72 | 350 | 4.86:1 |

`random_state`/seed fixa em 42 para reprodutibilidade.

### Limitações conhecidas desta base

- **MicroSource Discovery Systems**: não foi possível localizar uma fonte pública
  gratuita e legítima para esta biblioteca comercial específica. Nenhum dado dela
  foi incluído — isso explica parte da diferença de contagem em relação ao artigo.
- **DrugAge (fonte [27]) direta**: o export oficial do DrugAge (`drugage.csv`,
  1.046 nomes únicos, todas as espécies) não possui SMILES nativamente. A tentativa
  de resolver nomes → SMILES via PubChem PUG REST foi limitada por instabilidade/
  rate-limiting severo da API (muitos erros 503). Por isso, a cobertura de DrugAge
  nesta reconstrução vem indiretamente, através dos subconjuntos que já incorporam
  compostos do DrugAge com SMILES prontas (AgeXtend e Barardo et al. 2017), e não
  do arquivo oficial completo do DrugAge.
- **LOPAC como fonte da base primária**: o artigo cita LOPAC como uma das fontes
  integradas à base primária. Neste trabalho, dados LOPAC (via a coleção Prestwick/
  LOPAC de Smer-Barreto et al. 2023) foram usados para a **base externa** (ver
  abaixo), não para a primária, por serem obtidos de um dataset rotulado
  especificamente para triagem de senolíticos (rótulo binário senolítico/não),
  e não como biblioteca química genérica.
- A diferença de 126 compostos (4.233 vs. 4.107) e de 42 positivos (727 vs. 769)
  reflete escolhas de curadoria não documentadas publicamente pelos autores originais
  (critérios exatos de inclusão/exclusão, versão exata de cada fonte, tratamento de
  duplicatas entre fontes).

## 2. Base de teste externa (`external_test_set.csv`)

**Meta do artigo:** conjunto independente, sem sobreposição com a base primária,
com 37 positivos e 1.707 negativos, construído a partir de literatura, patentes e
bibliotecas comerciais distintas, referenciando [31] Smer-Barreto et al. 2023
(*Nature Communications*, aprendizado de máquina para descoberta de senolíticos).

**Resultado obtido:** **1.557 compostos únicos** (35 positivos / 1.522 negativos)
após remoção estrita de qualquer sobreposição (por SMILES canônica) com a base
primária.

### Como foi construída

Os dados de Smer-Barreto et al. 2023 foram obtidos diretamente do repositório
Zenodo oficial do artigo (DOI `10.5281/zenodo.7870357`), arquivo
`list_of_compounds_for_training.csv` (2.523 compostos: bibliotecas LOPAC e
Prestwick, mais um conjunto de 58 senolíticos positivos vindos de 15 fontes da
literatura/patentes citadas no próprio artigo de Smer-Barreto — exatamente o tipo
de "literatura, patentes e bibliotecas comerciais" descrito na Seção III-A do
Gero-LLM). Após canonicalização e deduplicação (2.482 compostos únicos), foram
removidos 925 compostos que já apareciam na base primária, restando 1.557
compostos estritamente independentes — muito próximo, em proporção, da meta do
artigo (37/1.707 = 1.744 total; aqui 35/1.522 = 1.557 total).

### Limitação conhecida

Os 2 positivos "faltantes" (35 obtidos vs. 37 do artigo) provavelmente correspondem
a compostos que, nesta reconstrução, caíram simultaneamente na base primária (e
foram removidos da externa por sobreposição) ou a diferenças de curadoria exata
entre a lista de 58 senolíticos de Smer-Barreto e a lista de 37 usada pelos autores
do Gero-LLM (não publicada em detalhe).

## 3. Verificação de integridade

- Nenhuma sobreposição de SMILES canônica entre base primária e base externa (0 compostos em comum).
- Nenhuma sobreposição entre os splits train/val/test (partição estrita).
- União de train+val+test = base primária completa (4.233 compostos).
- Todas as SMILES nos arquivos finais são válidas e canonicalizáveis via RDKit.

## 4. Arquivos entregues

- `primary_dataset_dedup.csv` — base primária completa (4.233 compostos), colunas: `smiles,label,sources,name`
- `primary_dataset_split.csv` — mesma base primária com coluna adicional `split` (train/val/test)
- `primary_train.csv`, `primary_val.csv`, `primary_test.csv` — splits separados
- `external_test_set.csv` — base de teste externa (1.557 compostos), colunas: `smiles,label,sources,name`
- `METODOLOGIA.md` — este documento

`label`: 1 = geroprotetor/senolítico (positivo); 0 = não-geroprotetor (negativo/background).
