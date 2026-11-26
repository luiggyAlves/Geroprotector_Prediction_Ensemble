# How the datasets were reconstructed

This folder holds a reconstruction, from public sources, of the two datasets described in the
"Dataset" section of *Gero-LLM: A Multimodal Large Language Model for Geroprotector Discovery
via Cross-Modal Differentiated Mutual Learning*. The original files were not released, so each
source was collected again and consolidated here; the counts below therefore differ slightly
from the ones reported there, and every difference is stated.

## 1. Primary dataset (`primary_dataset_dedup.csv`)

**Target from that paper:** 4,107 unique molecules (769 geroprotectors / 3,338
non-geroprotectors, a ratio of about 1:4.34), consolidating DrugAge, the Geroprotective
Compounds Database (geroprotectors.org), LOPAC, MicroSource Discovery Systems and the studies
of Barardo et al. 2017, Arora et al. 2025 (AgeXtend), Wong et al. 2023 (a senolytics DNN) and
Ye et al. 2014 (a pharmacological network in *C. elegans*).

**What this reconstruction obtained:** **4,233 unique molecules** (727 positive / 3,506
negative, a ratio of 4.82:1).

### Sources actually collected and merged

| Source in that paper | File used here | Raw records | How it was obtained |
|---|---|---|---|
| AgeXtend (Arora et al. 2025) | `agextend_holy_aging.csv` | 972 | Cloned from the official `the-ahuja-lab/AgeXtend` repository on GitHub, file `HOLY_AgingDataset.tsv`. The provenance labels (DrugAge, PMID 24134630 / Ye et al. 2014, Geroprotective Compounds) are kept in the `sources` column. |
| Barardo et al. 2017 | `barardo2017_tableS1.csv` | 1,395 | Official supplementary material of the article (`aging-09-1721-s001.xlsx`, Table S1), with the real SMILES taken from the spreadsheet. |
| Wong et al. 2023 (senolytics) | `wong2023_senolytics.csv` | 2,352 | Cloned from the official `felixjwong/senolyticsai` repository on GitHub, `notebook_files/train.csv`. |
| Geroprotectors.org | `geroprotectors_2024.csv` | 206 | The "Geroprotectors_Clean_Descriptors_2024" spreadsheet, with real SMILES. |

Every SMILES was canonicalized with RDKit (`Chem.MolToSmiles(canonical=True)`) and deduplicated
by canonical SMILES. A compound labelled positive in any source was kept positive — 109
conflicts were resolved this way, which follows the "curated positive list plus negative
background" logic of the original. Two raw SMILES could not be parsed by RDKit and were
dropped.

### Known limitations of this dataset

- **MicroSource Discovery Systems:** no free, legitimate public source was found for this
  specific commercial library. None of its data is included, which explains part of the
  difference in counts.
- **DrugAge directly:** the official DrugAge export (`drugage.csv`, 1,046 unique names, all
  species) carries no SMILES. Resolving names to SMILES through the PubChem PUG REST API was
  defeated by severe rate limiting (many 503 responses). DrugAge coverage in this
  reconstruction therefore arrives indirectly, through the subsets that already carry DrugAge
  compounds with SMILES attached (AgeXtend and Barardo et al. 2017), rather than from the
  complete official DrugAge file.
- **LOPAC as a source of the primary dataset:** the original lists LOPAC among the sources
  merged into the primary dataset. Here, LOPAC data (through the Prestwick/LOPAC collection of
  Smer-Barreto et al. 2023) was used for the **external set** instead, because it was obtained
  from a dataset labelled specifically for senolytic screening (a binary senolytic/non-senolytic
  label) rather than as a generic chemical library.
- The difference of 126 compounds (4,233 vs 4,107) and of 42 positives (727 vs 769) reflects
  curation choices the original authors did not document publicly: exact inclusion and
  exclusion criteria, the exact version of each source, and the handling of duplicates across
  sources.

## 2. External test set (`external_test_set.csv`)

**Target from that paper:** an independent set with no overlap with the primary dataset, 37
positives and 1,707 negatives, assembled from literature, patents and different commercial
libraries, citing Smer-Barreto et al. 2023 (*Nature Communications*, machine learning for
senolytic discovery).

**What this reconstruction obtained:** **1,557 unique compounds** (35 positive / 1,522
negative) after strictly removing every overlap, by canonical SMILES, with the primary dataset.

### How it was built

The data of Smer-Barreto et al. 2023 was taken directly from the article's official Zenodo
record (DOI `10.5281/zenodo.7870357`), file `list_of_compounds_for_training.csv` (2,523
compounds: the LOPAC and Prestwick libraries, plus a set of 58 positive senolytics drawn from
15 literature and patent sources cited in that article — exactly the kind of "literature,
patents and commercial libraries" the original describes). After canonicalization and
deduplication (2,482 unique compounds), 925 compounds that already appeared in the primary
dataset were removed, leaving 1,557 strictly independent compounds — close in proportion to the
target (37/1,707 = 1,744 in total there; 35/1,522 = 1,557 here).

### Known limitation

The two "missing" positives (35 obtained against 37) most likely correspond either to compounds
that, in this reconstruction, also landed in the primary dataset and were therefore removed
from the external set, or to curation differences between Smer-Barreto's list of 58 senolytics
and the list of 37 the original authors used, which was not published in detail.

## 3. Integrity checks

- No canonical SMILES is shared between the primary dataset and the external set (zero
  compounds in common).
- Every SMILES in the delivered files is valid and canonicalizable with RDKit.

## 4. Delivered files

- `primary_dataset_dedup.csv` — the complete primary dataset (4,233 compounds), columns
  `smiles,label,sources,name`
- `external_test_set.csv` — the external test set (1,557 compounds), same columns
- `METHODOLOGY.md` — this document

`label`: 1 = geroprotector/senolytic (positive); 0 = non-geroprotector (negative background).
