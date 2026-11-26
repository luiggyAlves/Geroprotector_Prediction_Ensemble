import numpy as np

# Caches para modelos/tokenizers/device (começam vazios, carregados sob demanda)
_chembert_cache = {'tokenizer': None, 'model': None}
_chemberta3_cache = {'tokenizer': None, 'model': None}
_molformer_cache = {'tokenizer': None, 'model': None}
_unimol_cache = {'model': None}
_selformer_cache = {'tokenizer': None, 'model': None}
_moldeberta_cache = {'tokenizer': None, 'model': None}
_molclr_cache = {'model': None}
_molebert_cache = {'model': None}
_gin_pretrain_cache = {'model': None}
_graphmvp_cache = {'model': None}
_mol2vec_cache = {'model': None}
_chemgpt_cache = {'tokenizer': None, 'model': None}
_chemfm_cache = {'tokenizer': None, 'model': None}
_unimol2_cache = {'model': None}
_mat_cache = {'model': None, 'featurizer': None}
_rmat_cache = {'model': None, 'featurizer': None}
_clamp_cache = {'model': None}
_fingerprint_cache = {'generator': None}
_atompair_cache = {'generator': None}
_torsion_cache = {'generator': None}
_device_cache = {'value': None}

CHEMBERT_ID = "DeepChem/ChemBERTa-77M-MTR"
# ChemBERTa-3 (framework/paper 2026): pré-treinado no ZINC20, não no PubChem
# como o ChemBERT_ID acima (esse é o critério pra diferenciar v2 de v3).
CHEMBERTA3_ID = "DeepChem/ChemBERTa-100M-MLM"
# Atenção à grafia: é "MoL" (L maiúsculo), não "Mol"
MOLFORMER_ID = "ibm-research/MoLFormer-XL-both-10pct"
SELFORMER_ID = "HUBioDataLab/SELFormer"
MOLDEBERTA_ID = "SaeedLab/MolDeBERTa-base-123M-mtr"  # variante SOTA geral recomendada pelos autores
MOLDEBERTA_MAX_LEN = 128  # limite de positional embeddings do modelo (ver paper, secao de arquitetura)

# MolCLR e Mole-BERT nao tem porte no HF Hub: arquitetura (GIN) e featurizacao
# vendorizadas dos repos originais abaixo, e o checkpoint pre-treinado e
# baixado direto do GitHub (commitado no repo, nao e ponteiro Git LFS).
MOLCLR_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/yuyangw/MolCLR/master/"
    "ckpt/pretrained_gin/checkpoints/model.pth"
)  # https://github.com/yuyangw/MolCLR (models/ginet_finetune.py)
MOLEBERT_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/junxia97/Mole-BERT/main/"
    "model_gin/Mole-BERT.pth"
)  # https://github.com/junxia97/Mole-BERT (model.py, classe GNN)

# GIN "puro" (Hu et al. 2020, pretrain-gnns) - mesma arquitetura vendorizada
# do Mole-BERT (num_atom_type=120, num_chirality_tag=3, num_bond_type=6,
# num_bond_direction=3), so trocando o checkpoint - ver _construir_gnn_pretrain_gnns.
# ATENCAO: GraphMVP NAO compartilha essa arquitetura (ver GRAPHMVP_CHECKPOINT_URL
# abaixo) - apesar de tambem ser da familia "pretrain-gnns", o checkpoint
# publicado usa featurizacao OGB completa (9 features de atomo / 3 de ligacao),
# arquitetura diferente da simplificada usada por MolCLR/Mole-BERT/GIN.
GIN_PRETRAIN_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/master/"
    "chem/model_gin/contextpred.pth"
)  # https://github.com/snap-stanford/pretrain-gnns (chem/model.py, classe GNN)

# GraphMVP (Liu et al., ICLR 2022): pre-treino contrastivo 3D-2D + gerativo,
# variante GraphMVP-C. Checkpoint espelhado no HF Hub pelos proprios autores.
# Arquitetura = GNN estilo OGB (ogb.graphproppred.mol_encoder): AtomEncoder de
# 9 features (9 embeddings, dims [119,4,12,12,10,6,6,2,2]) + GINConv com
# epsilon aprendido e BondEncoder de 3 features (dims [5,6,2]) POR CAMADA, mais
# um batch_norms externo por camada - dims conferidos direto no state_dict do
# checkpoint baixado (ver _construir_gnn_ogb).
GRAPHMVP_CHECKPOINT_URL = (
    "https://huggingface.co/chao1224/MoleculeSTM/resolve/main/"
    "pretrained_GraphMVP/GraphMVP_C/model.pth"
)  # https://github.com/chao1224/GraphMVP

# ChemFM (Gao et al., "Neural Scaling Law...", Nature Communications Chemistry
# 2025): foundation model causal (GPT-like) pre-treinado em ~178M moleculas
# (UniChem). Menor variante publicada no HF Hub, mesmo padrao de uso do
# ChemGPT (AutoModel + mean pooling do ultimo hidden state).
CHEMFM_ID = "ChemFM/ChemFMv2-20M"

# Uni-Mol2 (versao mais recente do Uni-Mol, treinada em mais dados e com mais
# tamanhos de modelo) - mesmo pacote unimol_tools ja usado pro Uni-Mol(v1)
# em _get_unimol, so trocando model_name/model_size.
UNIMOL2_MODEL_SIZE = "84m"

# MAT (Maziarka et al., 2020) e R-MAT (Maziarka et al., 2024): transformers
# moleculares que incorporam a matriz de adjacencia/distancia (MAT) ou
# features relativas de estrutura (R-MAT) diretamente na atencao. Sem porte
# no HF Hub - pacote proprio "huggingmolecules" (github.com/gmum/huggingmolecules),
# instalado via `pip install -e ./src` do repo clonado (setup.py da pasta
# src/ so pede torch/sklearn/filelock/gdown - nao os pins antigos do
# requirements.txt raiz do repo, que e so pra reproduzir os experimentos de
# treino). Checkpoint baixado automaticamente do Google Drive (gdown) na
# primeira chamada a from_pretrained.
MAT_CHECKPOINT = "mat_masking_20M"
RMAT_CHECKPOINT = "rmat_4M"

# CLAMP (Seidl et al., 2023): modelo contrastivo composto-x-ensaio (CLIP-like).
# So usamos o COMPOUND ENCODER (MLP), sem o encoder de texto do ensaio -
# equivalente ao `encode_smiles` do pacote oficial (github.com/ml-jku/clamp),
# mas reimplementado direto aqui (sem instalar o pacote `clamp` nem sua
# dependencia `mhnreact`, que nao esta no PyPI) porque as duas pecas que
# precisamos sao simples: (1) fingerprint "morganc+rdkc" hasheado em 8192-dim
# (funcao `getFingerprint` de mhnreact/molutils.py - reproduzida bit a bit
# abaixo, incluindo o "bug" de sobrescrever em vez de somar em colisoes de
# hash, que faz parte do que o modelo foi treinado com) e (2) o MLP do
# compound_encoder (768-dim, 8192->4096->2048->768, LayerNorm SEM parametros
# aprendidos entre camadas - por isso nao aparecem pesos de LayerNorm no
# checkpoint). Confirmado batendo os hiperparametros publicados (hp.json) e
# as dimensoes reais dos pesos no checkpoint.pt baixado.
CLAMP_CHECKPOINT_URL = "https://cloud.ml.jku.at/s/7nxgpAQrTr69Rp2/download/checkpoint.pt"
CLAMP_FP_SIZE = 8192
CLAMP_FP_RADIUS = 2
CLAMP_HIDDEN_SIZES = [4096, 2048]
CLAMP_EMBEDDING_SIZE = 768

# Mol2Vec (Jaeger, Fulle & Turk 2018): Word2Vec treinado sobre "sentencas" de
# identificadores de sub-estruturas Morgan (radius 0 e 1). Checkpoint oficial
# do proprio repo do metodo, treinado em ~19M moleculas do ZINC.
MOL2VEC_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/samoturk/mol2vec/master/"
    "examples/models/model_300dim.pkl"
)  # https://github.com/samoturk/mol2vec
MOL2VEC_RADIUS = 1
MOL2VEC_UNSEEN = "UNK"  # token p/ identificadores fora do vocabulario, treinado no proprio modelo

# ChemGPT (Frey et al., "Neural Scaling of Deep Chemical Models" 2023):
# GPT-Neo pre-treinado sobre SELFIES do PubChem10M. Variante grande (1.2B
# params, 2048 dim); ncfrey/ChemGPT-4.7M e ncfrey/ChemGPT-19M tambem existem
# no HF Hub com a mesma interface, caso custo de download/inferencia pese.
CHEMGPT_ID = "ncfrey/ChemGPT-1.2B"


def _baixar_checkpoint(url, nome_arquivo):
    """Baixa (uma unica vez) e cacheia localmente um checkpoint que nao vive no HF Hub."""
    from pathlib import Path
    import urllib.request

    destino = Path.home() / ".cache" / "pibic_embeddings" / nome_arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        urllib.request.urlretrieve(url, destino)
    return destino


def _smiles_para_grafo_pyg(smile, add_hs):
    """
    Featurizacao "simple" da familia pretrain-gnns (Hu et al. 2020, ref [34]
    do artigo Gero-LLM), usada tanto pelo MolCLR quanto pelo Mole-BERT:
    atomo -> [indice do numero atomico (1-118), indice da quiralidade];
    ligacao -> [indice do tipo de ligacao, indice da direcao], espelhada nos
    dois sentidos (grafo nao-direcionado). Devolve None p/ SMILES invalido.

    add_hs=True adiciona hidrogenios explicitos (Chem.AddHs) antes de montar
    o grafo - e o que o MolCLR faz (dataset_test.py); o Mole-BERT (loader.py)
    nao adiciona, entao passe add_hs=False para ele. Usar o convencao errada
    faz o modelo rodar sobre uma distribuicao de entrada diferente da que
    ele foi pre-treinado, silenciosamente piorando o embedding.
    """
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    ATOM_LIST = list(range(1, 119))
    CHIRALITY_LIST = [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER,
    ]
    BOND_LIST = [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC,
    ]
    BONDDIR_LIST = [
        Chem.rdchem.BondDir.NONE, Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT,
    ]

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    if add_hs:
        mol = Chem.AddHs(mol)

    tipo_idx = [ATOM_LIST.index(atom.GetAtomicNum()) for atom in mol.GetAtoms()]
    quiral_idx = [CHIRALITY_LIST.index(atom.GetChiralTag()) for atom in mol.GetAtoms()]
    x = torch.tensor(list(zip(tipo_idx, quiral_idx)), dtype=torch.long)

    linhas, colunas, atributos = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
        linhas += [i, j]
        colunas += [j, i]
        atributos += [feat, feat]

    if linhas:
        edge_index = torch.tensor([linhas, colunas], dtype=torch.long)
        edge_attr = torch.tensor(atributos, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _safe_index(lista, valor):
    """Indice de `valor` em `lista`; se nao encontrado, cai no ultimo indice
    (categoria "misc"/catch-all) - mesma convencao do OGB (ogb.utils.features)."""
    try:
        return lista.index(valor)
    except ValueError:
        return len(lista) - 1


# Listas de categorias do esquema de featurizacao completo do OGB
# (ogb.utils.features.atom_to_feature_vector / bond_to_feature_vector),
# usado pelo GraphMVP - MAIS RICO que o esquema simples de _smiles_para_grafo_pyg
# (2 features de atomo / 2 de ligacao). Tamanhos de cada lista conferidos
# direto contra os embeddings do checkpoint baixado (ver GRAPHMVP_CHECKPOINT_URL):
# atomo = [119, 4, 12, 12, 10, 6, 6, 2, 2], ligacao = [5, 6, 2].
_OGB_ATOMIC_NUM = list(range(1, 119)) + ['misc']                          # 119
_OGB_CHIRALITY = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
                  'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']                     # 4 (sem 'misc')
_OGB_DEGREE = list(range(11)) + ['misc']                                 # 12
_OGB_FORMAL_CHARGE = list(range(-5, 6)) + ['misc']                       # 12
_OGB_NUM_H = list(range(9)) + ['misc']                                   # 10
_OGB_NUM_RADICAL_E = list(range(5)) + ['misc']                           # 6
_OGB_HYBRIDIZATION = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc']       # 6
_OGB_BOND_TYPE = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc']      # 5
_OGB_BOND_STEREO = ['STEREONONE', 'STEREOZ', 'STEREOE',
                    'STEREOCIS', 'STEREOTRANS', 'STEREOANY']            # 6 (sem 'misc')

OGB_ATOM_FEATURE_DIMS = [119, 4, 12, 12, 10, 6, 6, 2, 2]
OGB_BOND_FEATURE_DIMS = [5, 6, 2]


def _smiles_para_grafo_ogb(smile):
    """
    Featurizacao "completa" do OGB (ogb.graphproppred.mol_encoder), usada
    pelo GraphMVP: 9 features categoricas por atomo (numero atomico,
    quiralidade, grau, carga formal, num. hidrogenios, num. eletrons
    radicais, hibridizacao, aromaticidade, se esta em anel) e 3 por ligacao
    (tipo, estereoquimica, se e conjugada), espelhada nos dois sentidos
    (grafo nao-direcionado). Devolve None p/ SMILES invalido.
    """
    import torch
    from rdkit import Chem
    from torch_geometric.data import Data

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    features_atomos = []
    for atom in mol.GetAtoms():
        features_atomos.append([
            _safe_index(_OGB_ATOMIC_NUM, atom.GetAtomicNum()),
            _safe_index(_OGB_CHIRALITY, str(atom.GetChiralTag())),
            _safe_index(_OGB_DEGREE, atom.GetTotalDegree()),
            _safe_index(_OGB_FORMAL_CHARGE, atom.GetFormalCharge()),
            _safe_index(_OGB_NUM_H, atom.GetTotalNumHs()),
            _safe_index(_OGB_NUM_RADICAL_E, atom.GetNumRadicalElectrons()),
            _safe_index(_OGB_HYBRIDIZATION, str(atom.GetHybridization())),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),
        ])
    x = torch.tensor(features_atomos, dtype=torch.long)

    linhas, colunas, atributos = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [
            _safe_index(_OGB_BOND_TYPE, str(bond.GetBondType())),
            _safe_index(_OGB_BOND_STEREO, str(bond.GetStereo())),
            int(bond.GetIsConjugated()),
        ]
        linhas += [i, j]
        colunas += [j, i]
        atributos += [feat, feat]

    if linhas:
        edge_index = torch.tensor([linhas, colunas], dtype=torch.long)
        edge_attr = torch.tensor(atributos, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _get_device():
    """Detecta e cacheia o device (cuda/cpu). Só importa torch quando chamado."""
    if _device_cache['value'] is None:
        import torch
        _device_cache['value'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device_cache['value']


def _get_chembert():
    """Carrega o ChemBERTa sob demanda (apenas na primeira chamada)."""
    if _chembert_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(CHEMBERT_ID)
        model = AutoModel.from_pretrained(CHEMBERT_ID)
        model.eval()
        model.to(_get_device())
        _chembert_cache['tokenizer'] = tokenizer
        _chembert_cache['model'] = model
    return _chembert_cache['tokenizer'], _chembert_cache['model']


def _get_chemberta3():
    """Carrega o ChemBERTa-3 sob demanda (apenas na primeira chamada)."""
    if _chemberta3_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA3_ID)
        model = AutoModel.from_pretrained(CHEMBERTA3_ID)
        model.eval()
        model.to(_get_device())
        _chemberta3_cache['tokenizer'] = tokenizer
        _chemberta3_cache['model'] = model
    return _chemberta3_cache['tokenizer'], _chemberta3_cache['model']


def _limpar_cache_molformer():
    """Remove o cache de código dinâmico do MolFormer, caso esteja corrompido/incompleto."""
    import shutil
    from pathlib import Path

    cache_dir = (
        Path.home() / ".cache" / "huggingface" / "modules"
        / "transformers_modules" / "ibm_hyphen_research"
    )
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)

    hub_dir = (
        Path.home() / ".cache" / "huggingface" / "hub"
        / "models--ibm-research--MoLFormer-XL-both-10pct"
    )
    if hub_dir.exists():
        shutil.rmtree(hub_dir, ignore_errors=True)


def _get_molformer():
    """Carrega o MolFormer sob demanda (apenas na primeira chamada).

    Se o cache de código dinâmico estiver corrompido/incompleto
    (ModuleNotFoundError em transformers_modules...), limpa o cache
    e força um novo download automaticamente.
    """
    if _molformer_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModel
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                MOLFORMER_ID, trust_remote_code=True
            )
            model = AutoModel.from_pretrained(
                MOLFORMER_ID, deterministic_eval=True, trust_remote_code=True
            )
        except ModuleNotFoundError:
            _limpar_cache_molformer()
            tokenizer = AutoTokenizer.from_pretrained(
                MOLFORMER_ID, trust_remote_code=True, force_download=True
            )
            model = AutoModel.from_pretrained(
                MOLFORMER_ID, deterministic_eval=True, trust_remote_code=True,
                force_download=True
            )

        model.eval()
        model.to(_get_device())
        _molformer_cache['tokenizer'] = tokenizer
        _molformer_cache['model'] = model
    return _molformer_cache['tokenizer'], _molformer_cache['model']


def _stub_rdmoldraw2d():
    """Workaround: bug conhecido do RDKit 2025.09.x no Windows quebra
    rdkit.Chem.Draw (rdMolDraw2D falha ao carregar via DLL, e o proprio
    rdkit/Chem/Draw/__init__.py depende de simbolos que vem de la - mesmo
    stubando so o submodulo rdMolDraw2D, o __init__.py do pacote Draw ainda
    quebra ao tentar usar MolDrawOptions como valor padrao de argumento).
    Não precisamos de desenho de moléculas pra gerar embeddings, então
    stubamos o pacote Draw INTEIRO antes que o unimol_tools tente
    importa-lo via PandasTools (evita executar o __init__.py quebrado)."""
    import sys
    import types
    if "rdkit.Chem.Draw" not in sys.modules:
        stub_draw = types.ModuleType("rdkit.Chem.Draw")
        stub_rmd2d = types.ModuleType("rdkit.Chem.Draw.rdMolDraw2D")
        stub_draw.rdMolDraw2D = stub_rmd2d
        sys.modules["rdkit.Chem.Draw"] = stub_draw
        sys.modules["rdkit.Chem.Draw.rdMolDraw2D"] = stub_rmd2d


def _get_unimol(model_name="unimolv1", model_size="84m"):
    """Carrega o Uni-Mol sob demanda (apenas na primeira chamada)."""
    if _unimol_cache['model'] is None:
        _stub_rdmoldraw2d()
        import torch
        from unimol_tools import UniMolRepr
        clf = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name=model_name,
            model_size=model_size,
            use_cuda=torch.cuda.is_available(),
        )
        _unimol_cache['model'] = clf
    return _unimol_cache['model']


def _get_unimol2():
    """Carrega o Uni-Mol2 (versao mais recente, mais dados de pre-treino e
    mais tamanhos de modelo disponiveis) sob demanda - mesmo pacote
    unimol_tools do Uni-Mol(v1), so trocando model_name/model_size."""
    if _unimol2_cache['model'] is None:
        _stub_rdmoldraw2d()
        import torch
        from unimol_tools import UniMolRepr
        clf = UniMolRepr(
            data_type='molecule',
            remove_hs=False,
            model_name='unimolv2',
            model_size=UNIMOL2_MODEL_SIZE,
            use_cuda=torch.cuda.is_available(),
        )
        _unimol2_cache['model'] = clf
    return _unimol2_cache['model']


def _get_selformer():
    """Carrega o SELFormer sob demanda (apenas na primeira chamada)."""
    if _selformer_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(SELFORMER_ID)
        model = AutoModel.from_pretrained(SELFORMER_ID)
        model.eval()
        model.to(_get_device())
        _selformer_cache['tokenizer'] = tokenizer
        _selformer_cache['model'] = model
    return _selformer_cache['tokenizer'], _selformer_cache['model']


def _get_fg_generator():
    """Cria e cacheia o gerador de fingerprints Morgan (só importa rdkit quando chamado)."""
    if _fingerprint_cache['generator'] is None:
        from rdkit.Chem import rdFingerprintGenerator
        _fingerprint_cache['generator'] = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )
    return _fingerprint_cache['generator']


def gerar_fingerprint(smile):
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = _get_fg_generator().GetFingerprint(mol)
    return np.array(fp)


def _get_atompair_generator():
    """Cria e cacheia o gerador de fingerprint Atom-Pair (Carhart et al. 1985)."""
    if _atompair_cache['generator'] is None:
        from rdkit.Chem import rdFingerprintGenerator
        _atompair_cache['generator'] = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
    return _atompair_cache['generator']


def gerar_embedding_atompair(smile):
    """Fingerprint Atom-Pair: conta pares de atomos por (tipo, distancia
    topologica), diferente do Morgan/ECFP (que conta ambientes circulares)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = _get_atompair_generator().GetFingerprint(mol)
    return np.array(fp)


def gerar_embedding_graphfp(smile, fp_size=2048):
    """Fingerprint topologico baseado em caminhos do grafo molecular
    (RDKit RDKFingerprint, ao estilo Daylight) - referido como "Graph
    fingerprint" em benchmarks (ex.: MoleculeNet/DeepPurpose), em contraste
    com fingerprints circulares (Morgan/ECFP) ou de sub-estrutura (MACCS)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = Chem.RDKFingerprint(mol, fpSize=fp_size)
    return np.array(fp)


def _get_torsion_generator():
    """Cria e cacheia o gerador de fingerprint Topological-Torsion."""
    if _torsion_cache['generator'] is None:
        from rdkit.Chem import rdFingerprintGenerator
        _torsion_cache['generator'] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
    return _torsion_cache['generator']


def gerar_embedding_tt(smile):
    """Fingerprint Topological-Torsion (TT): caminhos de 4 atomos (torsoes)
    descritos por (tipo de atomo, num. de ligacoes-pi, num. de ramificacoes)
    de cada atomo no caminho - complementar ao Atom-Pair (que usa so pares)."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = _get_torsion_generator().GetFingerprint(mol)
    return np.array(fp)


def _get_mol2vec():
    """Carrega o Word2Vec pre-treinado do Mol2Vec sob demanda."""
    if _mol2vec_cache['model'] is None:
        from gensim.models import word2vec
        checkpoint_path = _baixar_checkpoint(MOL2VEC_CHECKPOINT_URL, "mol2vec_model_300dim.pkl")
        _mol2vec_cache['model'] = word2vec.Word2Vec.load(str(checkpoint_path))
    return _mol2vec_cache['model']


def _mol_para_sentenca_morgan(mol, radius=MOL2VEC_RADIUS):
    """Re-implementacao do `mol2alt_sentence` do Mol2Vec (helpers.py):
    para cada atomo, encadeia o identificador do seu ambiente Morgan em cada
    raio de 0 ate `radius`, alternando entre atomos - a "sentenca" que o
    Word2Vec do Mol2Vec trata como texto (cada identificador = uma "palavra"
    de sub-estrutura)."""
    from rdkit.Chem import AllChem

    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    identificador_por_atomo_raio = {atom.GetIdx(): {r: None for r in range(radius + 1)}
                                    for atom in mol.GetAtoms()}
    for identificador, ocorrencias in info.items():
        for atom_idx, r in ocorrencias:
            identificador_por_atomo_raio[atom_idx][r] = identificador

    sentenca = []
    for atom_idx in identificador_por_atomo_raio:
        for r in range(radius + 1):
            ident = identificador_por_atomo_raio[atom_idx][r]
            if ident is not None:
                sentenca.append(str(ident))
    return sentenca


def gerar_embedding_mol2vec(smile):
    """Gera o embedding Mol2Vec: soma os vetores Word2Vec (300-dim) dos
    identificadores de sub-estrutura Morgan da molecula; identificadores fora
    do vocabulario caem no token MOL2VEC_UNSEEN (convencao do proprio modelo
    pre-treinado, ver notebook oficial do Mol2Vec)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    modelo = _get_mol2vec()
    sentenca = _mol_para_sentenca_morgan(mol)
    vocab = modelo.wv.key_to_index
    vetor_unk = modelo.wv[MOL2VEC_UNSEEN]

    vetores = [modelo.wv[palavra] if palavra in vocab else vetor_unk for palavra in sentenca]
    if not vetores:
        return np.zeros(modelo.wv.vector_size, dtype=np.float32)
    return np.sum(vetores, axis=0)


def gerar_embedding_chembert(smile, max_length=256):
    import torch
    tokenizer_chembert, chembert = _get_chembert()
    device = _get_device()
    inputs = tokenizer_chembert(
        smile,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = chembert(**inputs)
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_hidden = last_hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts
    return mean_pooled.squeeze().cpu().numpy()


def gerar_embedding_chemberta3(smile, max_length=256):
    """Gera o embedding do ChemBERTa-3 (DeepChem/ChemBERTa-100M-MLM,
    pré-treinado no ZINC20). Mesmo esquema de mean pooling usado no
    gerar_embedding_chembert, só trocando o modelo/tokenizer."""
    import torch
    tokenizer_chemberta3, chemberta3 = _get_chemberta3()
    device = _get_device()
    inputs = tokenizer_chemberta3(
        smile,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = chemberta3(**inputs)
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_hidden = last_hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts
    return mean_pooled.squeeze().cpu().numpy()


def gerar_embedding_molformer(smile, max_length=256):
    import torch
    tokenizer_molformer, molformer = _get_molformer()
    device = _get_device()
    inputs = tokenizer_molformer(
        smile,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = molformer(**inputs)
    embedding = outputs.pooler_output
    return embedding.squeeze().cpu().numpy()


def gerar_embedding_unimol(smile):
    clf = _get_unimol()
    repr_out = clf.get_repr([smile], return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    return np.array(cls_repr[0])


def gerar_embeddings_unimol_batch(smiles_list):
    """Versão em lote: mais eficiente que chamar gerar_embedding_unimol em loop."""
    clf = _get_unimol()
    repr_out = clf.get_repr(smiles_list, return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    embeddings = np.array(cls_repr)
    assert embeddings.shape[0] == len(smiles_list), (
        f"esperava {len(smiles_list)} embeddings, recebi {embeddings.shape[0]} "
        "(confira as moleculas que falharam na geracao de conformeros 3D)"
    )
    return embeddings


def gerar_embedding_unimol2(smile):
    clf = _get_unimol2()
    repr_out = clf.get_repr([smile], return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    return np.array(cls_repr[0])


def gerar_embeddings_unimol2_batch(smiles_list):
    """Versao em lote do Uni-Mol2 - mesma logica de gerar_embeddings_unimol_batch."""
    clf = _get_unimol2()
    repr_out = clf.get_repr(smiles_list, return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    embeddings = np.array(cls_repr)
    assert embeddings.shape[0] == len(smiles_list), (
        f"esperava {len(smiles_list)} embeddings, recebi {embeddings.shape[0]} "
        "(confira as moleculas que falharam na geracao de conformeros 3D)"
    )
    return embeddings


def _get_chemgpt():
    """Carrega o ChemGPT sob demanda (apenas na primeira chamada)."""
    if _chemgpt_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(CHEMGPT_ID)
        model = AutoModelForCausalLM.from_pretrained(CHEMGPT_ID)
        if tokenizer.pad_token is None:
            # o tokenizer do ChemGPT nao vem com pad_token NEM eos_token
            # configurados; adiciona um pad_token novo e redimensiona a
            # embedding do modelo. A posicao extra so recebe pad (mascarada
            # pela attention_mask no pooling), entao o embedding aleatorio
            # dela nunca entra no resultado.
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        model.to(_get_device())
        _chemgpt_cache['tokenizer'] = tokenizer
        _chemgpt_cache['model'] = model
    return _chemgpt_cache['tokenizer'], _chemgpt_cache['model']


def gerar_embedding_chemgpt(smile, max_length=256):
    """Gera o embedding do ChemGPT (GPT-Neo pre-treinado em SELFIES). Mesmo
    esquema de conversao SMILES->SELFIES do SELFormer, e mesmo mean pooling
    (ponderado pela attention_mask) sobre o ultimo hidden state, aplicado
    aqui a um modelo causal em vez de bidirecional."""
    import torch
    import selfies as sf
    tokenizer_chemgpt, chemgpt = _get_chemgpt()
    device = _get_device()
    try:
        selfies_str = sf.encoder(smile)
    except sf.EncoderError:
        return None

    inputs = tokenizer_chemgpt(
        selfies_str,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = chemgpt(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_hidden = last_hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts
    return mean_pooled.squeeze().cpu().numpy()


def _get_chemfm():
    """Carrega o ChemFM sob demanda (apenas na primeira chamada). Diferente
    do ChemGPT, opera direto sobre SMILES (nao SELFIES) - arquitetura padrao
    LlamaForCausalLM, ja vem com pad_token_id definido no config.json."""
    if _chemfm_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(CHEMFM_ID)
        model = AutoModelForCausalLM.from_pretrained(CHEMFM_ID)
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        model.to(_get_device())
        _chemfm_cache['tokenizer'] = tokenizer
        _chemfm_cache['model'] = model
    return _chemfm_cache['tokenizer'], _chemfm_cache['model']


def gerar_embedding_chemfm(smile, max_length=256):
    """Gera o embedding do ChemFM (foundation model causal, 178M moleculas
    do UniChem). Mesmo mean pooling ponderado pela attention_mask dos
    demais modelos causais (ChemGPT), mas direto sobre a string SMILES."""
    import torch
    from rdkit import Chem

    if Chem.MolFromSmiles(smile) is None:
        return None

    tokenizer_chemfm, chemfm = _get_chemfm()
    device = _get_device()
    inputs = tokenizer_chemfm(
        smile,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = chemfm(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1]
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_hidden = last_hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts
    return mean_pooled.squeeze().cpu().numpy()


def _pool_como_generator(generator, encoded, mask):
    """Reproduz a etapa de pooling do `Generator` do huggingmolecules
    (MAT/R-MAT) SEM a projecao final (`.proj`) - a projecao leva ao
    `d_output` da tarefa de pre-treino (ex.: 1 escalar, mascaramento), o
    que queremos e a representacao agregada por molecula ANTES dela (768 ou
    1024-dim, dependendo do checkpoint). Suporta os dois tipos de agregacao
    usados pelos checkpoints publicos: 'mean' (MAT) e 'grover' (R-MAT,
    attention pooling com o att_net ja pre-treinado)."""
    import torch.nn.functional as F

    mask = mask.unsqueeze(-1).float()
    out_masked = encoded * mask
    if generator.aggregation_type == "mean":
        return out_masked.sum(dim=1) / mask.sum(dim=1)
    elif generator.aggregation_type == "grover":
        out_attn = generator.att_net(out_masked)
        out_attn = out_attn.masked_fill(mask == 0, -1e9)
        out_attn = F.softmax(out_attn, dim=1)
        pooled = out_attn.transpose(-1, -2) @ out_masked
        return pooled.reshape(pooled.size(0), -1)
    raise ValueError(f"aggregation_type '{generator.aggregation_type}' nao suportado")


def _get_mat():
    """Carrega o MAT (Molecule Attention Transformer) sob demanda."""
    if _mat_cache['model'] is None:
        from huggingmolecules import MatModel, MatFeaturizer
        modelo = MatModel.from_pretrained(MAT_CHECKPOINT)
        modelo.eval()
        modelo.to(_get_device())
        _mat_cache['model'] = modelo
        _mat_cache['featurizer'] = MatFeaturizer.from_pretrained(MAT_CHECKPOINT)
    return _mat_cache['model'], _mat_cache['featurizer']


def gerar_embedding_mat(smile):
    """Gera o embedding do MAT: roda o encoder ate a camada anterior ao
    generator de pre-treino e agrega por molecula com o MESMO pooling
    ('mean', pesos ja pre-treinados) que o checkpoint original usa."""
    import torch
    from rdkit import Chem

    if Chem.MolFromSmiles(smile) is None:
        return None

    modelo, featurizer = _get_mat()
    batch = featurizer([smile]).to(_get_device())
    with torch.no_grad():
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = modelo.src_embed(batch.node_features)
        encoded = modelo.encoder(embedded, batch_mask,
                                 adj_matrix=batch.adjacency_matrix,
                                 distance_matrix=batch.distance_matrix)
        pooled = _pool_como_generator(modelo.generator, encoded, batch_mask)
    return pooled.squeeze().cpu().numpy()


def _get_rmat():
    """Carrega o R-MAT (Relative Molecule Attention Transformer) sob demanda."""
    if _rmat_cache['model'] is None:
        from huggingmolecules import RMatModel, RMatFeaturizer
        modelo = RMatModel.from_pretrained(RMAT_CHECKPOINT)
        modelo.eval()
        modelo.to(_get_device())
        _rmat_cache['model'] = modelo
        _rmat_cache['featurizer'] = RMatFeaturizer.from_pretrained(RMAT_CHECKPOINT)
    return _rmat_cache['model'], _rmat_cache['featurizer']


def gerar_embedding_rmat(smile):
    """Gera o embedding do R-MAT - mesma logica do MAT, mas o encoder usa
    features relativas de estrutura (`relative_matrix`) em vez de
    adjacencia/distancia diretas, e o pooling do checkpoint publico e
    'grover' (attention), nao 'mean'."""
    import torch
    from rdkit import Chem

    if Chem.MolFromSmiles(smile) is None:
        return None

    modelo, featurizer = _get_rmat()
    batch = featurizer([smile]).to(_get_device())
    with torch.no_grad():
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = modelo.src_embed(batch.node_features)
        distances_matrix = modelo.dist_rbf(batch.distance_matrix)
        edges_att = torch.cat((batch.bond_features, batch.relative_matrix, distances_matrix), dim=1)
        encoded = modelo.encoder(embedded, batch_mask, edges_att=edges_att)
        pooled = _pool_como_generator(modelo.generator, encoded, batch_mask)
    return pooled.squeeze().cpu().numpy()


def _clamp_fingerprint(smile, fp_size=CLAMP_FP_SIZE, radius=CLAMP_FP_RADIUS):
    """Reproduz `getFingerprint(..., which='morganc+rdkc')` de
    mhnreact/molutils.py (mesma featurizacao usada pra pre-treinar o CLAMP):
    Morgan contado + RDKit contado, cada um hasheado (indice % fp_size, COM
    o "bug" do original de sobrescrever em vez de somar colisoes - faz parte
    da distribuicao que o modelo viu no treino, entao mantido de proposito),
    somados entre si e por fim log(1+x). Devolve None p/ SMILES invalido."""
    import numpy as _np
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Chem.rdmolops import FastFindRings

    mol_check = Chem.MolFromSmiles(smile)
    if mol_check is None:
        return None

    mol = Chem.MolFromSmiles(str(smile), sanitize=False)
    Chem.SanitizeMol(mol, catchErrors=True)
    FastFindRings(mol)
    mol.UpdatePropertyCache(strict=False)

    def _fold(fp_dict):
        nd = _np.zeros(fp_size)
        for k, v in fp_dict.items():
            nd[k % fp_size] = v  # last-write-wins em colisao (fiel ao original)
        return nd

    fp_morganc = AllChem.GetMorganFingerprint(
        mol, radius, useChirality=True, useBondTypes=True, useFeatures=True, useCounts=True
    ).GetNonzeroElements()
    fp_rdkc = AllChem.UnfoldedRDKFingerprintCountBased(mol, maxPath=6).GetNonzeroElements()

    np_fp = _fold(fp_morganc) + _fold(fp_rdkc)
    return _np.log1p(np_fp).astype('float32')


def _construir_clamp_compound_encoder():
    """Constroi o compound_encoder do CLAMP: MLP 8192->4096->2048->768 com
    LayerNorm SEM parametros aprendidos (elementwise_affine=False) entre as
    camadas - por isso o checkpoint so tem pesos de Linear (ver
    NetworkLayerNorm/MultilayerPerceptron em clamp/models/perceptron.py)."""
    from torch import nn

    class CompoundEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            tamanhos = [CLAMP_FP_SIZE] + CLAMP_HIDDEN_SIZES
            self.linear_input = nn.Linear(tamanhos[0], tamanhos[1])
            self.linear_hidden_l = nn.ModuleList([
                nn.Linear(tamanhos[i], tamanhos[i + 1]) for i in range(1, len(tamanhos) - 1)
            ])
            self.linear_output = nn.Linear(tamanhos[-1], CLAMP_EMBEDDING_SIZE)
            self.norm_input = nn.LayerNorm(tamanhos[1], elementwise_affine=False)
            self.norm_hidden_l = nn.ModuleList([
                nn.LayerNorm(tamanhos[i + 1], elementwise_affine=False) for i in range(1, len(tamanhos) - 1)
            ])
            self.ativacao = nn.ReLU()

        def forward(self, x):
            x = self.ativacao(self.norm_input(self.linear_input(x)))
            for linear, norm in zip(self.linear_hidden_l, self.norm_hidden_l):
                x = self.ativacao(norm(linear(x)))
            return self.linear_output(x)

    return CompoundEncoder()


def _get_clamp():
    """Carrega o compound_encoder do CLAMP sob demanda (apenas na primeira
    chamada). So os pesos com prefixo 'compound_encoder.' sao usados - o
    assay_encoder (lado texto/ensaio, que nao usamos) fica no checkpoint mas
    e ignorado."""
    if _clamp_cache['model'] is None:
        import torch
        modelo = _construir_clamp_compound_encoder()
        checkpoint_path = _baixar_checkpoint(CLAMP_CHECKPOINT_URL, "clamp_checkpoint.pt")
        cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        estado_bruto = cp['model_state_dict']
        estado = {
            'linear_input.weight': estado_bruto['compound_encoder.linear_input.weight'],
            'linear_input.bias': estado_bruto['compound_encoder.linear_input.bias'],
            'linear_output.weight': estado_bruto['compound_encoder.linear_output.weight'],
            'linear_output.bias': estado_bruto['compound_encoder.linear_output.bias'],
        }
        for i in range(len(CLAMP_HIDDEN_SIZES) - 1):
            estado[f'linear_hidden_l.{i}.weight'] = estado_bruto[f'compound_encoder.linear_hidden_l.{i}.weight']
            estado[f'linear_hidden_l.{i}.bias'] = estado_bruto[f'compound_encoder.linear_hidden_l.{i}.bias']
        modelo.load_state_dict(estado, strict=True)
        modelo.eval()
        modelo.to(_get_device())
        _clamp_cache['model'] = modelo
    return _clamp_cache['model']


def gerar_embedding_clamp(smile):
    """Gera o embedding do CLAMP (768-dim) - equivalente ao `encode_smiles`
    oficial: fingerprint morganc+rdkc (8192-dim) -> compound_encoder (MLP).
    Sem normalizacao L2 final (o pacote oficial so normaliza dentro do
    `forward`/produto interno com o assay_encoder, nao em `encode_smiles`)."""
    import torch

    fp = _clamp_fingerprint(smile)
    if fp is None:
        return None

    modelo = _get_clamp()
    device = _get_device()
    x = torch.from_numpy(fp).float().unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = modelo(x)
    return embedding.squeeze().cpu().numpy()


def gerar_embedding_selformer(smile, max_length=512):
    """Gera o embedding do SELFormer. Internamente converte o SMILES para SELFIES
    (requisito do modelo) antes de tokenizar."""
    import torch
    import selfies as sf
    tokenizer_selformer, selformer = _get_selformer()
    device = _get_device()
    try:
        selfies_str = sf.encoder(smile)
    except sf.EncoderError:
        return None

    inputs = tokenizer_selformer(
        selfies_str,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        outputs = selformer(**inputs)
    last_hidden = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1)
    masked_hidden = last_hidden * attention_mask
    summed = masked_hidden.sum(dim=1)
    counts = attention_mask.sum(dim=1)
    mean_pooled = summed / counts
    return mean_pooled.squeeze().cpu().numpy()


def _get_moldeberta():
    """Carrega o MolDeBERTa sob demanda (apenas na primeira chamada)."""
    if _moldeberta_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(MOLDEBERTA_ID)
        model = AutoModel.from_pretrained(MOLDEBERTA_ID)
        model.eval()
        model.to(_get_device())
        _moldeberta_cache['tokenizer'] = tokenizer
        _moldeberta_cache['model'] = model
    return _moldeberta_cache['tokenizer'], _moldeberta_cache['model']


def gerar_embedding_moldeberta(smile, max_length=MOLDEBERTA_MAX_LEN):
    """
    Gera o embedding de um SMILES com o MolDeBERTa: mean pooling sobre o
    last_hidden_state, ponderado pela attention_mask (ignora tokens de
    padding). Retorna None para SMILES invalido, mesmo padrao dos demais
    gerar_embedding_*.

    IMPORTANTE: max_length=MOLDEBERTA_MAX_LEN e passado explicitamente porque
    o modelo tem apenas 128 posicoes de embedding. Sem isso, truncation=True
    usa o model_max_length "cru" do tokenizer (que pode nao bater com o
    limite real do modelo) e SMILES longos geram um RuntimeError de indice
    fora do range dentro do forward - erro que gerar_vetores captura e trata
    como falha silenciosa, descartando a molecula.
    """
    import torch
    from rdkit import Chem

    if Chem.MolFromSmiles(smile) is None:
        return None

    tokenizer_moldeberta, moldeberta = _get_moldeberta()
    device = _get_device()
    inputs = tokenizer_moldeberta(
        smile, return_tensors='pt', padding=True,
        truncation=True, max_length=max_length,
    ).to(device)

    with torch.no_grad():
        outputs = moldeberta(**inputs)

    last_hidden = outputs.last_hidden_state
    attention_mask = inputs['attention_mask'].unsqueeze(-1).float()
    summed = (last_hidden * attention_mask).sum(dim=1)
    counts = attention_mask.sum(dim=1).clamp(min=1e-9)
    return (summed / counts).squeeze(0).cpu().numpy()


def _get_molclr():
    """Carrega o encoder GIN do MolCLR sob demanda (arquitetura vendorizada de
    github.com/yuyangw/MolCLR, models/ginet_finetune.py - sem porte no HF Hub).
    Checkpoint pre-treinado por contrastive learning em ~10M moleculas do
    PubChem, baixado uma unica vez do proprio repo do GitHub."""
    if _molclr_cache['model'] is None:
        import torch
        from torch import nn
        from torch_geometric.nn import MessagePassing, global_mean_pool
        from torch_geometric.utils import add_self_loops

        NUM_BOND_TYPE = 5       # 4 tipos + 1 reservado pro self-loop
        NUM_BOND_DIRECTION = 3

        class GINEConv(MessagePassing):
            def __init__(self, emb_dim):
                super().__init__()
                self.mlp = nn.Sequential(
                    nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                    nn.Linear(2 * emb_dim, emb_dim),
                )
                self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
                self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

            def forward(self, x, edge_index, edge_attr):
                edge_index = add_self_loops(edge_index, num_nodes=x.size(0))[0]
                self_loop_attr = torch.zeros(
                    x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
                self_loop_attr[:, 0] = 4
                edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
                edge_embeddings = (self.edge_embedding1(edge_attr[:, 0])
                                   + self.edge_embedding2(edge_attr[:, 1]))
                return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

            def message(self, x_j, edge_attr):
                return x_j + edge_attr

            def update(self, aggr_out):
                return self.mlp(aggr_out)

        class GINet(nn.Module):
            def __init__(self, num_layer=5, emb_dim=300, feat_dim=512, drop_ratio=0):
                super().__init__()
                self.num_layer = num_layer
                self.drop_ratio = drop_ratio
                self.x_embedding1 = nn.Embedding(119, emb_dim)
                self.x_embedding2 = nn.Embedding(3, emb_dim)
                self.gnns = nn.ModuleList([GINEConv(emb_dim) for _ in range(num_layer)])
                self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])
                self.pool = global_mean_pool
                self.feat_lin = nn.Linear(emb_dim, feat_dim)

            def forward(self, data):
                h = self.x_embedding1(data.x[:, 0]) + self.x_embedding2(data.x[:, 1])
                for layer in range(self.num_layer):
                    h = self.gnns[layer](h, data.edge_index, data.edge_attr)
                    h = self.batch_norms[layer](h)
                    if layer == self.num_layer - 1:
                        h = nn.functional.dropout(h, self.drop_ratio, training=self.training)
                    else:
                        h = nn.functional.dropout(torch.relu(h), self.drop_ratio, training=self.training)
                h = self.pool(h, data.batch)
                return self.feat_lin(h)

        modelo = GINet()
        checkpoint_path = _baixar_checkpoint(MOLCLR_CHECKPOINT_URL, "molclr_pretrained_gin.pth")
        estado_pretreinado = torch.load(checkpoint_path, map_location="cpu")
        estado_proprio = modelo.state_dict()
        # copia so as chaves que batem (o checkpoint bruto nao tem "pred_head.*",
        # que so existe apos fine-tuning supervisionado - mesma logica do
        # load_my_state_dict original do MolCLR, que ignora chaves ausentes)
        for nome, param in estado_pretreinado.items():
            if nome in estado_proprio and estado_proprio[nome].shape == param.shape:
                estado_proprio[nome].copy_(param)

        modelo.eval()
        modelo.to(_get_device())
        _molclr_cache['model'] = modelo
    return _molclr_cache['model']


def gerar_embedding_molclr(smile):
    """Gera o embedding do MolCLR (GIN pre-treinado por contrastive learning,
    512 dimensoes, pooling por media). Adiciona hidrogenios explicitos antes
    de montar o grafo (mesma convencao do dataset_test.py original)."""
    import torch
    from torch_geometric.data import Batch

    dado = _smiles_para_grafo_pyg(smile, add_hs=True)
    if dado is None:
        return None
    modelo = _get_molclr()
    lote = Batch.from_data_list([dado]).to(_get_device())
    with torch.no_grad():
        embedding = modelo(lote)
    return embedding.squeeze().cpu().numpy()


def _construir_gnn_pretrain_gnns():
    """Constroi a arquitetura de encoder GIN compartilhada pela familia
    "pretrain-gnns" (Hu et al. 2020, referencia [34] do artigo Gero-LLM):
    Mole-BERT, GraphMVP e o GIN "puro" (contextpred) sao todos treinados
    sobre esse MESMO encoder (mesmos hiperparametros num_atom_type=120,
    num_chirality_tag=3, num_bond_type=6, num_bond_direction=3 - GraphMVP e
    Mole-BERT sao ambos derivados do codigo original do pretrain-gnns), o que
    permite reusar uma unica implementacao pra carregar os tres checkpoints."""
    import torch
    from torch import nn
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops

    NUM_ATOM_TYPE = 120       # inclui tokens de mascara do pre-treino
    NUM_CHIRALITY_TAG = 3
    NUM_BOND_TYPE = 6         # inclui self-loop e token de mascara
    NUM_BOND_DIRECTION = 3

    class GINConv(MessagePassing):
        def __init__(self, emb_dim):
            super().__init__(aggr="add")
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim), nn.ReLU(),
                nn.Linear(2 * emb_dim, emb_dim),
            )
            self.edge_embedding1 = nn.Embedding(NUM_BOND_TYPE, emb_dim)
            self.edge_embedding2 = nn.Embedding(NUM_BOND_DIRECTION, emb_dim)

        def forward(self, x, edge_index, edge_attr):
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
            self_loop_attr = torch.zeros(
                x.size(0), 2, dtype=edge_attr.dtype, device=edge_attr.device)
            self_loop_attr[:, 0] = 4
            edge_attr = torch.cat((edge_attr, self_loop_attr), dim=0)
            edge_embeddings = (self.edge_embedding1(edge_attr[:, 0])
                               + self.edge_embedding2(edge_attr[:, 1]))
            return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)

        def message(self, x_j, edge_attr):
            return x_j + edge_attr

        def update(self, aggr_out):
            return self.mlp(aggr_out)

    class GNN(nn.Module):
        def __init__(self, num_layer=5, emb_dim=300, drop_ratio=0):
            super().__init__()
            self.num_layer = num_layer
            self.drop_ratio = drop_ratio
            self.x_embedding1 = nn.Embedding(NUM_ATOM_TYPE, emb_dim)
            self.x_embedding2 = nn.Embedding(NUM_CHIRALITY_TAG, emb_dim)
            self.gnns = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layer)])
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

        def forward(self, x, edge_index, edge_attr):
            h = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])
            for layer in range(self.num_layer):
                h = self.gnns[layer](h, edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                if layer == self.num_layer - 1:
                    h = nn.functional.dropout(h, self.drop_ratio, training=self.training)
                else:
                    h = nn.functional.dropout(torch.relu(h), self.drop_ratio, training=self.training)
            return h  # representacao por atomo (JK="last")

    return GNN()


def _get_molebert():
    """Carrega o encoder GIN do Mole-BERT sob demanda (arquitetura vendorizada
    de github.com/junxia97/Mole-BERT, model.py classe GNN - mesma familia
    "pretrain-gnns" de Hu et al. 2020, referencia [34] do artigo Gero-LLM;
    sem porte no HF Hub). Checkpoint pre-treinado com o tokenizer VQ-VAE do
    proprio Mole-BERT, baixado uma unica vez do repo do GitHub."""
    if _molebert_cache['model'] is None:
        import torch
        modelo = _construir_gnn_pretrain_gnns()
        checkpoint_path = _baixar_checkpoint(MOLEBERT_CHECKPOINT_URL, "molebert_gin.pth")
        # aqui o checkpoint bate exatamente com os parametros da GNN (mesmo
        # esquema do from_pretrained original: self.gnn.load_state_dict(...)),
        # entao carregamento estrito, sem copia parcial
        modelo.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        modelo.eval()
        modelo.to(_get_device())
        _molebert_cache['model'] = modelo
    return _molebert_cache['model']


def _get_gin_pretrain():
    """Carrega o GIN "puro" pre-treinado por context prediction (Hu et al.
    2020, github.com/snap-stanford/pretrain-gnns) - o encoder original do
    qual Mole-BERT e GraphMVP derivam, sem nenhuma etapa adicional de
    pre-treino contrastivo/gerativo por cima."""
    if _gin_pretrain_cache['model'] is None:
        import torch
        modelo = _construir_gnn_pretrain_gnns()
        checkpoint_path = _baixar_checkpoint(GIN_PRETRAIN_CHECKPOINT_URL, "gin_contextpred.pth")
        modelo.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        modelo.eval()
        modelo.to(_get_device())
        _gin_pretrain_cache['model'] = modelo
    return _gin_pretrain_cache['model']


def _construir_gnn_ogb():
    """Constroi o encoder GNN estilo OGB (ogb.graphproppred.mol_encoder /
    examples/graphproppred/mol/conv.py) usado pelo GraphMVP - arquitetura
    DIFERENTE da familia simples pretrain-gnns (_construir_gnn_pretrain_gnns):
    AtomEncoder com 9 embeddings (uma por feature categorica), GINConv com
    epsilon aprendido + BondEncoder (3 embeddings) por camada, e batch_norms
    externo por camada. Nomes de atributos (atom_encoder, gnns, batch_norms,
    eps, bond_encoder, mlp) e dimensoes conferidos direto no state_dict do
    checkpoint do GraphMVP."""
    import torch
    from torch import nn
    from torch_geometric.nn import MessagePassing

    class AtomEncoder(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.atom_embedding_list = nn.ModuleList(
                [nn.Embedding(d, emb_dim) for d in OGB_ATOM_FEATURE_DIMS])

        def forward(self, x):
            saida = 0
            for i, emb in enumerate(self.atom_embedding_list):
                saida = saida + emb(x[:, i])
            return saida

    class BondEncoder(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.bond_embedding_list = nn.ModuleList(
                [nn.Embedding(d, emb_dim) for d in OGB_BOND_FEATURE_DIMS])

        def forward(self, edge_attr):
            saida = 0
            for i, emb in enumerate(self.bond_embedding_list):
                saida = saida + emb(edge_attr[:, i])
            return saida

    class GINConv(MessagePassing):
        def __init__(self, emb_dim):
            super().__init__(aggr="add")
            self.mlp = nn.Sequential(
                nn.Linear(emb_dim, 2 * emb_dim), nn.BatchNorm1d(2 * emb_dim),
                nn.ReLU(), nn.Linear(2 * emb_dim, emb_dim),
            )
            self.eps = nn.Parameter(torch.zeros(1))
            self.bond_encoder = BondEncoder(emb_dim)

        def forward(self, x, edge_index, edge_attr):
            edge_embedding = self.bond_encoder(edge_attr)
            return self.mlp((1 + self.eps) * x
                            + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        def message(self, x_j, edge_attr):
            return torch.relu(x_j + edge_attr)

    class GNN(nn.Module):
        def __init__(self, num_layer=5, emb_dim=300):
            super().__init__()
            self.num_layer = num_layer
            self.atom_encoder = AtomEncoder(emb_dim)
            self.gnns = nn.ModuleList([GINConv(emb_dim) for _ in range(num_layer)])
            self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

        def forward(self, x, edge_index, edge_attr):
            h = self.atom_encoder(x)
            for layer in range(self.num_layer):
                h = self.gnns[layer](h, edge_index, edge_attr)
                h = self.batch_norms[layer](h)
                if layer != self.num_layer - 1:
                    h = torch.relu(h)
            return h  # representacao por atomo (JK="last")

    return GNN()


def _get_graphmvp():
    """Carrega o encoder GraphMVP-C (Liu et al., ICLR 2022) sob demanda."""
    if _graphmvp_cache['model'] is None:
        import torch
        modelo = _construir_gnn_ogb()
        checkpoint_path = _baixar_checkpoint(GRAPHMVP_CHECKPOINT_URL, "graphmvp_c.pth")
        modelo.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        modelo.eval()
        modelo.to(_get_device())
        _graphmvp_cache['model'] = modelo
    return _graphmvp_cache['model']


def gerar_embedding_graphmvp(smile):
    """Gera o embedding do GraphMVP-C (300 dimensoes, pooling por media dos
    atomos). Usa a featurizacao COMPLETA do OGB (_smiles_para_grafo_ogb),
    diferente da featurizacao simples usada por GIN/MolCLR/Mole-BERT."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool

    dado = _smiles_para_grafo_ogb(smile)
    if dado is None:
        return None
    modelo = _get_graphmvp()
    lote = Batch.from_data_list([dado]).to(_get_device())
    with torch.no_grad():
        representacao_por_atomo = modelo(lote.x, lote.edge_index, lote.edge_attr)
    embedding = global_mean_pool(representacao_por_atomo, lote.batch)
    return embedding.squeeze().cpu().numpy()


def gerar_embedding_gin(smile):
    """Gera o embedding do GIN pre-treinado por context prediction (300
    dimensoes, pooling por media dos atomos). Sem hidrogenios explicitos,
    mesma convencao do loader.py original do pretrain-gnns."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool

    dado = _smiles_para_grafo_pyg(smile, add_hs=False)
    if dado is None:
        return None
    modelo = _get_gin_pretrain()
    lote = Batch.from_data_list([dado]).to(_get_device())
    with torch.no_grad():
        representacao_por_atomo = modelo(lote.x, lote.edge_index, lote.edge_attr)
    embedding = global_mean_pool(representacao_por_atomo, lote.batch)
    return embedding.squeeze().cpu().numpy()


def gerar_embedding_molebert(smile):
    """Gera o embedding do Mole-BERT (GIN pre-treinado, 300 dimensoes, pooling
    por media dos atomos). Sem hidrogenios explicitos - mesma convencao
    implicita do loader.py original do Mole-BERT."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool

    dado = _smiles_para_grafo_pyg(smile, add_hs=False)
    if dado is None:
        return None
    modelo = _get_molebert()
    lote = Batch.from_data_list([dado]).to(_get_device())
    with torch.no_grad():
        representacao_por_atomo = modelo(lote.x, lote.edge_index, lote.edge_attr)
    embedding = global_mean_pool(representacao_por_atomo, lote.batch)
    return embedding.squeeze().cpu().numpy()