import numpy as np

# Caches for models/tokenizers/device (start empty, loaded on demand)
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
# ChemBERTa-3: pretrained on ZINC20, not on PubChem like CHEMBERT_ID above (that is what
# distinguishes v2 from v3).
CHEMBERTA3_ID = "DeepChem/ChemBERTa-100M-MLM"
# Mind the spelling: it is "MoL" (capital L), not "Mol"
MOLFORMER_ID = "ibm-research/MoLFormer-XL-both-10pct"
SELFORMER_ID = "HUBioDataLab/SELFormer"
MOLDEBERTA_ID = "SaeedLab/MolDeBERTa-base-123M-mtr"  # variante SOTA geral recomendada pelos autores
MOLDEBERTA_MAX_LEN = 128  # the model's positional-embedding limit (see its architecture section)

# MolCLR and Mole-BERT have no port on the HF Hub: the architecture (GIN) and the
# featurization are vendored from the original repositories below, and the pretrained
# checkpoint is downloaded straight from GitHub (committed in the repo, not a Git LFS pointer).
MOLCLR_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/yuyangw/MolCLR/master/"
    "ckpt/pretrained_gin/checkpoints/model.pth"
)  # https://github.com/yuyangw/MolCLR (models/ginet_finetune.py)
MOLEBERT_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/junxia97/Mole-BERT/main/"
    "model_gin/Mole-BERT.pth"
)  # https://github.com/junxia97/Mole-BERT (model.py, the GNN class)

# Plain GIN (Hu et al. 2020, pretrain-gnns) - the same vendored architecture as Mole-BERT
# (num_atom_type=120, num_chirality_tag=3, num_bond_type=6, num_bond_direction=3), only the
# checkpoint changes; see _build_pretrain_gnns_encoder.
# CAREFUL: GraphMVP does NOT share this architecture (see GRAPHMVP_CHECKPOINT_URL below).
# Although it also belongs to the "pretrain-gnns" family, the published checkpoint uses the
# full OGB featurization (9 atom features / 3 bond features), a different architecture from the
# simplified one used by MolCLR/Mole-BERT/GIN.
GIN_PRETRAIN_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/snap-stanford/pretrain-gnns/master/"
    "chem/model_gin/contextpred.pth"
)  # https://github.com/snap-stanford/pretrain-gnns (chem/model.py, the GNN class)

# GraphMVP (Liu et al., ICLR 2022): contrastive 3D-2D plus generative pretraining,
# the GraphMVP-C variant. The checkpoint is mirrored on the HF Hub by the authors themselves.
# Architecture = OGB-style GNN (ogb.graphproppred.mol_encoder): a 9-feature AtomEncoder
# (9 embeddings, dims [119,4,12,12,10,6,6,2,2]) + GINConv with a learned epsilon and a
# 3-feature BondEncoder (dims [5,6,2]) PER LAYER, plus an outer batch_norms per layer. The dims
# were read directly from the state_dict of the downloaded checkpoint (see _build_ogb_encoder).
GRAPHMVP_CHECKPOINT_URL = (
    "https://huggingface.co/chao1224/MoleculeSTM/resolve/main/"
    "pretrained_GraphMVP/GraphMVP_C/model.pth"
)  # https://github.com/chao1224/GraphMVP

# ChemFM (Gao et al., "Neural Scaling Law...", Nature Communications Chemistry
# 2025): foundation model causal (GPT-like) pre-treinado em ~178M moleculas
# (UniChem). The smallest variant published on the HF Hub, used the same way as ChemGPT
# (AutoModel + mean pooling of the last hidden state).
CHEMFM_ID = "ChemFM/ChemFMv2-20M"

# Uni-Mol2 (the newer Uni-Mol, trained on more data and offered in more model sizes) - the
# same unimol_tools package already used for Uni-Mol v1 in _get_unimol, only with a different
# model_name/model_size.
UNIMOL2_MODEL_SIZE = "84m"

# MAT (Maziarka et al., 2020) e R-MAT (Maziarka et al., 2024): transformers
# molecular transformers that fold the adjacency/distance matrix (MAT) or relative structural
# features (R-MAT) directly into the attention. No port on the HF Hub - they come from their own
# package, "huggingmolecules" (github.com/gmum/huggingmolecules), installed with
# `pip install -e ./src` from a clone (the setup.py under src/ only asks for
# torch/sklearn/filelock/gdown, not the old pins of the requirements.txt at the repository
# root, which exist only to reproduce the training experiments). The checkpoint is downloaded
# from Google Drive (gdown) on the first call to from_pretrained.
MAT_CHECKPOINT = "mat_masking_20M"
RMAT_CHECKPOINT = "rmat_4M"

# CLAMP (Seidl et al., 2023): a contrastive compound-by-assay model (CLIP-like).
# Only the COMPOUND ENCODER (an MLP) is used, without the assay text encoder - the equivalent
# of `encode_smiles` in the official package (github.com/ml-jku/clamp), reimplemented here
# instead of installing the `clamp` package and its `mhnreact` dependency, which is not on
# PyPI, because the two pieces needed are simple: (1) the "morganc+rdkc" fingerprint hashed
# into 8192 dims (the `getFingerprint` function of mhnreact/molutils.py, reproduced bit for bit
# below, including the "bug" of overwriting rather than adding on hash collisions, which is
# part of what the model was trained on) and (2) the compound_encoder MLP (768-dim,
# 8192->4096->2048->768, LayerNorm WITHOUT learned parameters between layers, which is why no
# LayerNorm weights appear in the checkpoint). Confirmed against the published hyperparameters
# (hp.json) and the actual weight shapes in the downloaded checkpoint.pt.
CLAMP_CHECKPOINT_URL = "https://cloud.ml.jku.at/s/7nxgpAQrTr69Rp2/download/checkpoint.pt"
CLAMP_FP_SIZE = 8192
CLAMP_FP_RADIUS = 2
CLAMP_HIDDEN_SIZES = [4096, 2048]
CLAMP_EMBEDDING_SIZE = 768

# Mol2Vec (Jaeger, Fulle & Turk 2018): Word2Vec treinado sobre "sentencas" de
# identificadores de sub-estruturas Morgan (radius 0 e 1). Checkpoint oficial
# from the method's own repository, trained on ~19M ZINC molecules.
MOL2VEC_CHECKPOINT_URL = (
    "https://raw.githubusercontent.com/samoturk/mol2vec/master/"
    "examples/models/model_300dim.pkl"
)  # https://github.com/samoturk/mol2vec
MOL2VEC_RADIUS = 1
MOL2VEC_UNSEEN = "UNK"  # token for identifiers outside the vocabulary, trained in the model

# ChemGPT (Frey et al., "Neural Scaling of Deep Chemical Models" 2023):
# GPT-Neo pretrained on SELFIES from PubChem10M. This is the large variant (1.2B parameters,
# 2048 dims); ncfrey/ChemGPT-4.7M and ncfrey/ChemGPT-19M also exist on the HF Hub with the same
# interface, if download or inference cost matters.
CHEMGPT_ID = "ncfrey/ChemGPT-1.2B"


def _download_checkpoint(url, nome_arquivo):
    """Downloads once and caches locally a checkpoint that does not live on the HF Hub."""
    from pathlib import Path
    import urllib.request

    destino = Path.home() / ".cache" / "pibic_embeddings" / nome_arquivo
    destino.parent.mkdir(parents=True, exist_ok=True)
    if not destino.exists():
        urllib.request.urlretrieve(url, destino)
    return destino


def _smiles_to_pyg_graph(smile, add_hs):
    """
    The "simple" featurization of the pretrain-gnns family (Hu et al. 2020), used by both
    MolCLR and Mole-BERT: atom -> [index of the atomic number (1-118), index of the chirality];
    bond -> [index of the bond type, index of the direction], mirrored in both directions
    (an undirected graph). Returns None for an invalid SMILES.

    add_hs=True adds explicit hydrogens (Chem.AddHs) before building the graph, which is what
    MolCLR does (dataset_test.py); Mole-BERT (loader.py) does not, so pass add_hs=False for it.
    Using the wrong convention runs the model over an input distribution different from the one
    it was pretrained on, silently degrading the embedding.
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

    rows, cols, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [BOND_LIST.index(bond.GetBondType()), BONDDIR_LIST.index(bond.GetBondDir())]
        rows += [i, j]
        cols += [j, i]
        attrs += [feat, feat]

    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _safe_index(lista, valor):
    """Index of `valor` in `lista`; when absent it falls back to the last index, the
    "misc"/catch-all category - the same convention as OGB (ogb.utils.features)."""
    try:
        return lista.index(valor)
    except ValueError:
        return len(lista) - 1


# Category lists of the full OGB featurization scheme
# (ogb.utils.features.atom_to_feature_vector / bond_to_feature_vector), used by GraphMVP -
# RICHER than the simple scheme of _smiles_to_pyg_graph (2 atom features / 2 bond features).
# The size of each list was checked against the embeddings of the downloaded checkpoint (see
# GRAPHMVP_CHECKPOINT_URL): atoms = [119, 4, 12, 12, 10, 6, 6, 2, 2], bonds = [5, 6, 2].
_OGB_ATOMIC_NUM = list(range(1, 119)) + ['misc']                          # 119
_OGB_CHIRALITY = ['CHI_UNSPECIFIED', 'CHI_TETRAHEDRAL_CW',
                  'CHI_TETRAHEDRAL_CCW', 'CHI_OTHER']                     # 4 (no 'misc')
_OGB_DEGREE = list(range(11)) + ['misc']                                 # 12
_OGB_FORMAL_CHARGE = list(range(-5, 6)) + ['misc']                       # 12
_OGB_NUM_H = list(range(9)) + ['misc']                                   # 10
_OGB_NUM_RADICAL_E = list(range(5)) + ['misc']                           # 6
_OGB_HYBRIDIZATION = ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc']       # 6
_OGB_BOND_TYPE = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'misc']      # 5
_OGB_BOND_STEREO = ['STEREONONE', 'STEREOZ', 'STEREOE',
                    'STEREOCIS', 'STEREOTRANS', 'STEREOANY']            # 6 (no 'misc')

OGB_ATOM_FEATURE_DIMS = [119, 4, 12, 12, 10, 6, 6, 2, 2]
OGB_BOND_FEATURE_DIMS = [5, 6, 2]


def _smiles_to_ogb_graph(smile):
    """
    The "full" OGB featurization (ogb.graphproppred.mol_encoder), used by GraphMVP: 9
    categorical features per atom (atomic number, chirality, degree, formal charge, number of
    hydrogens, number of radical electrons, hybridization, aromaticity, whether it is in a
    ring) and 3 per bond (type, stereochemistry, whether it is conjugated), mirrored in both
    directions (an undirected graph). Returns None for an invalid SMILES.
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

    rows, cols, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = [
            _safe_index(_OGB_BOND_TYPE, str(bond.GetBondType())),
            _safe_index(_OGB_BOND_STEREO, str(bond.GetStereo())),
            int(bond.GetIsConjugated()),
        ]
        rows += [i, j]
        cols += [j, i]
        attrs += [feat, feat]

    if rows:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(attrs, dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _get_device():
    """Detects and caches the device (cuda/cpu). Imports torch only when called."""
    if _device_cache['value'] is None:
        import torch
        _device_cache['value'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device_cache['value']


def _get_chembert():
    """Loads ChemBERTa on demand (only on the first call)."""
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
    """Loads ChemBERTa-3 on demand (only on the first call)."""
    if _chemberta3_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA3_ID)
        model = AutoModel.from_pretrained(CHEMBERTA3_ID)
        model.eval()
        model.to(_get_device())
        _chemberta3_cache['tokenizer'] = tokenizer
        _chemberta3_cache['model'] = model
    return _chemberta3_cache['tokenizer'], _chemberta3_cache['model']


def _clear_molformer_cache():
    """Clears MolFormer's dynamic-code cache, in case it is corrupted or incomplete."""
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
    """Loads MolFormer on demand (only on the first call).

    If the dynamic-code cache is corrupted or incomplete (ModuleNotFoundError under
    transformers_modules...), it clears the cache and forces a fresh download.
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
            _clear_molformer_cache()
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


def _stub_rdkit_draw():
    """Workaround: a known RDKit 2025.09.x bug on Windows breaks rdkit.Chem.Draw
    (rdMolDraw2D fails to load through its DLL, and rdkit/Chem/Draw/__init__.py itself depends
    on symbols coming from there - even stubbing only the rdMolDraw2D submodule, the Draw
    package __init__.py still breaks when it tries to use MolDrawOptions as a default argument).
    Drawing molecules is not needed to compute embeddings, so the WHOLE Draw package is stubbed
    before unimol_tools tries to import it through PandasTools, which avoids running the broken
    __init__.py."""
    import sys
    import types
    if "rdkit.Chem.Draw" not in sys.modules:
        stub_draw = types.ModuleType("rdkit.Chem.Draw")
        stub_rmd2d = types.ModuleType("rdkit.Chem.Draw.rdMolDraw2D")
        stub_draw.rdMolDraw2D = stub_rmd2d
        sys.modules["rdkit.Chem.Draw"] = stub_draw
        sys.modules["rdkit.Chem.Draw.rdMolDraw2D"] = stub_rmd2d


def _get_unimol(model_name="unimolv1", model_size="84m"):
    """Loads Uni-Mol on demand (only on the first call)."""
    if _unimol_cache['model'] is None:
        _stub_rdkit_draw()
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
    """Loads Uni-Mol2 (the newer release, with more pretraining data and more model sizes
    available) on demand - the same package
    unimol_tools package as Uni-Mol v1, only with a different model_name/model_size."""
    if _unimol2_cache['model'] is None:
        _stub_rdkit_draw()
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
    """Loads SELFormer on demand (only on the first call)."""
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
    """Creates and caches the Morgan fingerprint generator (imports rdkit only when called)."""
    if _fingerprint_cache['generator'] is None:
        from rdkit.Chem import rdFingerprintGenerator
        _fingerprint_cache['generator'] = rdFingerprintGenerator.GetMorganGenerator(
            radius=2, fpSize=2048
        )
    return _fingerprint_cache['generator']


def embed_ecfp4(smile):
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


def embed_atompair(smile):
    """Atom-Pair fingerprint: counts atom pairs by (type, topological distance), unlike
    Morgan/ECFP, which counts circular environments."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = _get_atompair_generator().GetFingerprint(mol)
    return np.array(fp)


def embed_graphfp(smile, fp_size=2048):
    """Topological fingerprint based on paths of the molecular graph (RDKit RDKFingerprint, in
    the Daylight style) - called the "Graph fingerprint" in benchmarks such as
    MoleculeNet/DeepPurpose, as opposed to circular (Morgan/ECFP) or substructure (MACCS)
    fingerprints."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = Chem.RDKFingerprint(mol, fpSize=fp_size)
    return np.array(fp)


def _get_torsion_generator():
    """Creates and caches the Topological-Torsion fingerprint generator."""
    if _torsion_cache['generator'] is None:
        from rdkit.Chem import rdFingerprintGenerator
        _torsion_cache['generator'] = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048)
    return _torsion_cache['generator']


def embed_tt(smile):
    """Topological-Torsion (TT) fingerprint: paths of 4 atoms (torsions) described by
    (atom type, number of pi bonds, number of branches) for each atom along the path -
    complementary to Atom-Pair, which uses pairs only."""
    from rdkit import Chem
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None
    fp = _get_torsion_generator().GetFingerprint(mol)
    return np.array(fp)


def _get_mol2vec():
    """Loads the pretrained Mol2Vec Word2Vec model on demand."""
    if _mol2vec_cache['model'] is None:
        from gensim.models import word2vec
        checkpoint_path = _download_checkpoint(MOL2VEC_CHECKPOINT_URL, "mol2vec_model_300dim.pkl")
        _mol2vec_cache['model'] = word2vec.Word2Vec.load(str(checkpoint_path))
    return _mol2vec_cache['model']


def _mol_to_morgan_sentence(mol, radius=MOL2VEC_RADIUS):
    """Reimplementation of Mol2Vec's `mol2alt_sentence` (helpers.py): for each atom it chains
    the identifier of its Morgan environment at every radius from 0 to `radius`, alternating
    between atoms - the "sentence" that Mol2Vec's Word2Vec treats as text, where each
    identifier is one substructure "word"."""
    from rdkit.Chem import AllChem

    info = {}
    AllChem.GetMorganFingerprint(mol, radius, bitInfo=info)
    identifier_by_atom_radius = {atom.GetIdx(): {r: None for r in range(radius + 1)}
                                    for atom in mol.GetAtoms()}
    for identifier, occurrences in info.items():
        for atom_idx, r in occurrences:
            identifier_by_atom_radius[atom_idx][r] = identifier

    sentence = []
    for atom_idx in identifier_by_atom_radius:
        for r in range(radius + 1):
            ident = identifier_by_atom_radius[atom_idx][r]
            if ident is not None:
                sentence.append(str(ident))
    return sentence


def embed_mol2vec(smile):
    """Mol2Vec embedding: sums the 300-dim Word2Vec vectors of the molecule's Morgan
    substructure identifiers; identifiers outside the vocabulary fall back to the
    MOL2VEC_UNSEEN token, the convention of the pretrained model itself (see the official
    Mol2Vec notebook)."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return None

    model = _get_mol2vec()
    sentence = _mol_to_morgan_sentence(mol)
    vocab = model.wv.key_to_index
    vetor_unk = model.wv[MOL2VEC_UNSEEN]

    vectors = [model.wv[palavra] if palavra in vocab else vetor_unk for palavra in sentence]
    if not vectors:
        return np.zeros(model.wv.vector_size, dtype=np.float32)
    return np.sum(vectors, axis=0)


def embed_chemberta2(smile, max_length=256):
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


def embed_chemberta3(smile, max_length=256):
    """ChemBERTa-3 embedding (DeepChem/ChemBERTa-100M-MLM, pretrained on ZINC20). The same
    mean-pooling scheme as embed_chemberta2, with a different model/tokenizer."""
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


def embed_molformer(smile, max_length=256):
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


def embed_unimol(smile):
    clf = _get_unimol()
    repr_out = clf.get_repr([smile], return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    return np.array(cls_repr[0])


def embed_unimol_batch(smiles_list):
    """Batch version: cheaper than calling embed_unimol in a loop."""
    clf = _get_unimol()
    repr_out = clf.get_repr(smiles_list, return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    embeddings = np.array(cls_repr)
    assert embeddings.shape[0] == len(smiles_list), (
        f"esperava {len(smiles_list)} embeddings, recebi {embeddings.shape[0]} "
        "(check which molecules failed 3D conformer generation)"
    )
    return embeddings


def embed_unimol2(smile):
    clf = _get_unimol2()
    repr_out = clf.get_repr([smile], return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    return np.array(cls_repr[0])


def embed_unimol2_batch(smiles_list):
    """Batch version for Uni-Mol2 - the same logic as embed_unimol_batch."""
    clf = _get_unimol2()
    repr_out = clf.get_repr(smiles_list, return_atomic_reprs=False)
    cls_repr = repr_out['cls_repr'] if isinstance(repr_out, dict) else repr_out
    embeddings = np.array(cls_repr)
    assert embeddings.shape[0] == len(smiles_list), (
        f"esperava {len(smiles_list)} embeddings, recebi {embeddings.shape[0]} "
        "(check which molecules failed 3D conformer generation)"
    )
    return embeddings


def _get_chemgpt():
    """Loads ChemGPT on demand (only on the first call)."""
    if _chemgpt_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tokenizer = AutoTokenizer.from_pretrained(CHEMGPT_ID)
        model = AutoModelForCausalLM.from_pretrained(CHEMGPT_ID)
        if tokenizer.pad_token is None:
            # ChemGPT's tokenizer ships with neither pad_token nor eos_token configured; a new
            # pad_token is added and the model embedding is resized. The extra position only
            # ever receives padding, which the attention_mask masks out during pooling, so its
            # random embedding never reaches the result.
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))
        model.eval()
        model.to(_get_device())
        _chemgpt_cache['tokenizer'] = tokenizer
        _chemgpt_cache['model'] = model
    return _chemgpt_cache['tokenizer'], _chemgpt_cache['model']


def embed_chemgpt(smile, max_length=256):
    """ChemGPT embedding (GPT-Neo pretrained on SELFIES). The same SMILES->SELFIES conversion
    as SELFormer and the same mean pooling (weighted by the attention_mask) over the last
    hidden state, applied here to a causal rather than a bidirectional model."""
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
    """Loads ChemFM on demand (only on the first call). Unlike ChemGPT it works directly on
    SMILES rather than SELFIES - a standard LlamaForCausalLM architecture, which already
    defines pad_token_id in its config.json."""
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


def embed_chemfm(smile, max_length=256):
    """ChemFM embedding (a causal foundation model, 178M molecules
    of UniChem). The same attention_mask-weighted mean pooling as the
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


def _pool_like_generator(generator, encoded, mask):
    """Reproduces the pooling step of the huggingmolecules `Generator` (MAT/R-MAT) WITHOUT the
    final projection (`.proj`). That projection maps to the `d_output` of the pretraining task
    (a single scalar, for masking, for instance), whereas what is wanted here is the aggregated
    per-molecule representation BEFORE it (768 or 1024 dims, depending on the checkpoint).
    Supports both aggregations used by the public checkpoints: 'mean' (MAT) and 'grover'
    (R-MAT, attention pooling with the pretrained att_net)."""
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
    raise ValueError(f"unsupported aggregation_type '{generator.aggregation_type}'")


def _get_mat():
    """Carrega o MAT (Molecule Attention Transformer) sob demanda."""
    if _mat_cache['model'] is None:
        from huggingmolecules import MatModel, MatFeaturizer
        model = MatModel.from_pretrained(MAT_CHECKPOINT)
        model.eval()
        model.to(_get_device())
        _mat_cache['model'] = model
        _mat_cache['featurizer'] = MatFeaturizer.from_pretrained(MAT_CHECKPOINT)
    return _mat_cache['model'], _mat_cache['featurizer']


def embed_mat(smile):
    """MAT embedding: runs the encoder up to the layer before the pretraining generator and
    aggregates per molecule with the SAME pooling ('mean', with pretrained weights) that the
    original checkpoint uses."""
    import torch
    from rdkit import Chem

    if Chem.MolFromSmiles(smile) is None:
        return None

    model, featurizer = _get_mat()
    batch = featurizer([smile]).to(_get_device())
    with torch.no_grad():
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = model.src_embed(batch.node_features)
        encoded = model.encoder(embedded, batch_mask,
                                 adj_matrix=batch.adjacency_matrix,
                                 distance_matrix=batch.distance_matrix)
        pooled = _pool_like_generator(model.generator, encoded, batch_mask)
    return pooled.squeeze().cpu().numpy()


def _get_rmat():
    """Carrega o R-MAT (Relative Molecule Attention Transformer) sob demanda."""
    if _rmat_cache['model'] is None:
        from huggingmolecules import RMatModel, RMatFeaturizer
        model = RMatModel.from_pretrained(RMAT_CHECKPOINT)
        model.eval()
        model.to(_get_device())
        _rmat_cache['model'] = model
        _rmat_cache['featurizer'] = RMatFeaturizer.from_pretrained(RMAT_CHECKPOINT)
    return _rmat_cache['model'], _rmat_cache['featurizer']


def embed_rmat(smile):
    """R-MAT embedding - the same logic as MAT, except that the encoder uses relative
    structural features (`relative_matrix`) instead of adjacency/distance directly, and the
    pooling of the public checkpoint is 'grover' (attention) rather than 'mean'."""
    import torch
    from rdkit import Chem

    if Chem.MolFromSmiles(smile) is None:
        return None

    model, featurizer = _get_rmat()
    batch = featurizer([smile]).to(_get_device())
    with torch.no_grad():
        batch_mask = torch.sum(torch.abs(batch.node_features), dim=-1) != 0
        embedded = model.src_embed(batch.node_features)
        distances_matrix = model.dist_rbf(batch.distance_matrix)
        edges_att = torch.cat((batch.bond_features, batch.relative_matrix, distances_matrix), dim=1)
        encoded = model.encoder(embedded, batch_mask, edges_att=edges_att)
        pooled = _pool_like_generator(model.generator, encoded, batch_mask)
    return pooled.squeeze().cpu().numpy()


def _clamp_fingerprint(smile, fp_size=CLAMP_FP_SIZE, radius=CLAMP_FP_RADIUS):
    """Reproduces `getFingerprint(..., which='morganc+rdkc')` of mhnreact/molutils.py, the
    featurization CLAMP was pretrained with: counted Morgan + counted RDKit, each hashed
    (index % fp_size, WITH the original's "bug" of overwriting rather than adding on
    collisions, which is part of the distribution the model was trained on, so it is kept on
    purpose), added together and finally log(1+x). Returns None for an invalid SMILES."""
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


def _build_clamp_compound_encoder():
    """Builds CLAMP's compound_encoder: an MLP 8192->4096->2048->768 with LayerNorm WITHOUT
    learned parameters (elementwise_affine=False) between layers, which is why the checkpoint
    only holds Linear weights (see NetworkLayerNorm/MultilayerPerceptron in
    clamp/models/perceptron.py)."""
    from torch import nn

    class CompoundEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            sizes = [CLAMP_FP_SIZE] + CLAMP_HIDDEN_SIZES
            self.linear_input = nn.Linear(sizes[0], sizes[1])
            self.linear_hidden_l = nn.ModuleList([
                nn.Linear(sizes[i], sizes[i + 1]) for i in range(1, len(sizes) - 1)
            ])
            self.linear_output = nn.Linear(sizes[-1], CLAMP_EMBEDDING_SIZE)
            self.norm_input = nn.LayerNorm(sizes[1], elementwise_affine=False)
            self.norm_hidden_l = nn.ModuleList([
                nn.LayerNorm(sizes[i + 1], elementwise_affine=False) for i in range(1, len(sizes) - 1)
            ])
            self.ativacao = nn.ReLU()

        def forward(self, x):
            x = self.ativacao(self.norm_input(self.linear_input(x)))
            for linear, norm in zip(self.linear_hidden_l, self.norm_hidden_l):
                x = self.ativacao(norm(linear(x)))
            return self.linear_output(x)

    return CompoundEncoder()


def _get_clamp():
    """Loads CLAMP's compound_encoder on demand (only on the first call). Only the weights
    prefixed with 'compound_encoder.' are used; the assay_encoder (the text side, unused here)
    stays in the checkpoint and is ignored."""
    if _clamp_cache['model'] is None:
        import torch
        model = _build_clamp_compound_encoder()
        checkpoint_path = _download_checkpoint(CLAMP_CHECKPOINT_URL, "clamp_checkpoint.pt")
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
        model.load_state_dict(estado, strict=True)
        model.eval()
        model.to(_get_device())
        _clamp_cache['model'] = model
    return _clamp_cache['model']


def embed_clamp(smile):
    """CLAMP embedding (768-dim) - the equivalent of the official `encode_smiles`: the
    morganc+rdkc fingerprint (8192-dim) fed to the compound_encoder MLP. No final L2
    normalization (the official package normalizes inside `forward`, for the inner product with
    the assay_encoder, not in `encode_smiles`)."""
    import torch

    fp = _clamp_fingerprint(smile)
    if fp is None:
        return None

    model = _get_clamp()
    device = _get_device()
    x = torch.from_numpy(fp).float().unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(x)
    return embedding.squeeze().cpu().numpy()


def embed_selformer(smile, max_length=512):
    """SELFormer embedding. Converts the SMILES to SELFIES internally, as the model requires,
    before tokenizing."""
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
    """Loads MolDeBERTa on demand (only on the first call)."""
    if _moldeberta_cache['model'] is None:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(MOLDEBERTA_ID)
        model = AutoModel.from_pretrained(MOLDEBERTA_ID)
        model.eval()
        model.to(_get_device())
        _moldeberta_cache['tokenizer'] = tokenizer
        _moldeberta_cache['model'] = model
    return _moldeberta_cache['tokenizer'], _moldeberta_cache['model']


def embed_moldeberta(smile, max_length=MOLDEBERTA_MAX_LEN):
    """
    Embedding of one SMILES with MolDeBERTa: mean pooling over the last_hidden_state,
    weighted by the attention_mask so that padding tokens are ignored. Returns None for an
    invalid SMILES, like every other embedding function here.

    IMPORTANT: max_length=MOLDEBERTA_MAX_LEN is passed explicitly because the model only has
    128 embedding positions. Without it, truncation=True uses the tokenizer's raw
    model_max_length, which may not match the model's real limit, and long SMILES raise an
    index-out-of-range RuntimeError inside the forward pass - an error the caller catches and
    treats as a silent failure, dropping the molecule.
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
    """Loads MolCLR's GIN encoder on demand (architecture vendored from
    github.com/yuyangw/MolCLR, models/ginet_finetune.py - no port on the HF Hub). The
    checkpoint was pretrained by contrastive learning on ~10M PubChem molecules and is
    downloaded once from that GitHub repository."""
    if _molclr_cache['model'] is None:
        import torch
        from torch import nn
        from torch_geometric.nn import MessagePassing, global_mean_pool
        from torch_geometric.utils import add_self_loops

        NUM_BOND_TYPE = 5       # 4 types + 1 reserved for the self-loop
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

        model = GINet()
        checkpoint_path = _download_checkpoint(MOLCLR_CHECKPOINT_URL, "molclr_pretrained_gin.pth")
        estado_pretreinado = torch.load(checkpoint_path, map_location="cpu")
        estado_proprio = model.state_dict()
        # copies only the matching keys (the raw checkpoint has no "pred_head.*", which only
        # exists after supervised fine-tuning - the same logic as MolCLR's own
        # load_my_state_dict, which ignores missing keys)
        for name, param in estado_pretreinado.items():
            if name in estado_proprio and estado_proprio[name].shape == param.shape:
                estado_proprio[name].copy_(param)

        model.eval()
        model.to(_get_device())
        _molclr_cache['model'] = model
    return _molclr_cache['model']


def embed_molclr(smile):
    """MolCLR embedding (GIN pretrained by contrastive learning, 512 dims, mean pooling). Adds
    explicit hydrogens before building the graph, the same convention as the original
    dataset_test.py."""
    import torch
    from torch_geometric.data import Batch

    datum = _smiles_to_pyg_graph(smile, add_hs=True)
    if datum is None:
        return None
    model = _get_molclr()
    lote = Batch.from_data_list([datum]).to(_get_device())
    with torch.no_grad():
        embedding = model(lote)
    return embedding.squeeze().cpu().numpy()


def _build_pretrain_gnns_encoder():
    """Builds the GIN encoder architecture shared by the "pretrain-gnns" family (Hu et al.
    2020): Mole-BERT, GraphMVP and the plain GIN (contextpred) are all trained over the SAME
    encoder (the same hyperparameters num_atom_type=120, num_chirality_tag=3, num_bond_type=6,
    num_bond_direction=3, since GraphMVP and Mole-BERT both derive from the original
    pretrain-gnns code), which allows a single implementation to load all three
    checkpoints."""
    import torch
    from torch import nn
    from torch_geometric.nn import MessagePassing
    from torch_geometric.utils import add_self_loops

    NUM_ATOM_TYPE = 120       # includes the pretraining mask tokens
    NUM_CHIRALITY_TAG = 3
    NUM_BOND_TYPE = 6         # includes self-loop and the mask token
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
            return h  # per-atom representation (JK="last")

    return GNN()


def _get_molebert():
    """Loads Mole-BERT's GIN encoder on demand (architecture vendored from
    github.com/junxia97/Mole-BERT, the GNN class in model.py - the same "pretrain-gnns" family
    as Hu et al. 2020; no port on the HF Hub). The checkpoint was pretrained with Mole-BERT's
    own VQ-VAE tokenizer and is downloaded once from that GitHub repository."""
    if _molebert_cache['model'] is None:
        import torch
        model = _build_pretrain_gnns_encoder()
        checkpoint_path = _download_checkpoint(MOLEBERT_CHECKPOINT_URL, "molebert_gin.pth")
        # here the checkpoint matches the GNN parameters exactly (the same scheme as the
        # original from_pretrained: self.gnn.load_state_dict(...)), so the load is strict, with
        # no partial copy
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        model.to(_get_device())
        _molebert_cache['model'] = model
    return _molebert_cache['model']


def _get_gin_pretrain():
    """Loads the plain GIN pretrained by context prediction (Hu et al. 2020,
    github.com/snap-stanford/pretrain-gnns) - the original encoder from which Mole-BERT and
    GraphMVP derive, with no further contrastive or generative pretraining on top."""
    if _gin_pretrain_cache['model'] is None:
        import torch
        model = _build_pretrain_gnns_encoder()
        checkpoint_path = _download_checkpoint(GIN_PRETRAIN_CHECKPOINT_URL, "gin_contextpred.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        model.to(_get_device())
        _gin_pretrain_cache['model'] = model
    return _gin_pretrain_cache['model']


def _build_ogb_encoder():
    """Builds the OGB-style GNN encoder (ogb.graphproppred.mol_encoder /
    examples/graphproppred/mol/conv.py) used by GraphMVP - an architecture DIFFERENT from the
    simple pretrain-gnns family (_build_pretrain_gnns_encoder): an AtomEncoder with 9
    embeddings (one per categorical feature), GINConv with a learned epsilon plus a BondEncoder
    (3 embeddings) per layer, and an outer batch_norms per layer. The attribute names
    (atom_encoder, gnns, batch_norms, eps, bond_encoder, mlp) and the dimensions were read
    directly from the state_dict of the GraphMVP checkpoint."""
    import torch
    from torch import nn
    from torch_geometric.nn import MessagePassing

    class AtomEncoder(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.atom_embedding_list = nn.ModuleList(
                [nn.Embedding(d, emb_dim) for d in OGB_ATOM_FEATURE_DIMS])

        def forward(self, x):
            output = 0
            for i, emb in enumerate(self.atom_embedding_list):
                output = output + emb(x[:, i])
            return output

    class BondEncoder(nn.Module):
        def __init__(self, emb_dim):
            super().__init__()
            self.bond_embedding_list = nn.ModuleList(
                [nn.Embedding(d, emb_dim) for d in OGB_BOND_FEATURE_DIMS])

        def forward(self, edge_attr):
            output = 0
            for i, emb in enumerate(self.bond_embedding_list):
                output = output + emb(edge_attr[:, i])
            return output

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
            return h  # per-atom representation (JK="last")

    return GNN()


def _get_graphmvp():
    """Carrega o encoder GraphMVP-C (Liu et al., ICLR 2022) sob demanda."""
    if _graphmvp_cache['model'] is None:
        import torch
        model = _build_ogb_encoder()
        checkpoint_path = _download_checkpoint(GRAPHMVP_CHECKPOINT_URL, "graphmvp_c.pth")
        model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        model.eval()
        model.to(_get_device())
        _graphmvp_cache['model'] = model
    return _graphmvp_cache['model']


def embed_graphmvp(smile):
    """GraphMVP-C embedding (300 dims, mean pooling over atoms). Uses the FULL OGB
    featurization (_smiles_to_ogb_graph), unlike the simple featurization used by
    GIN/MolCLR/Mole-BERT."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool

    datum = _smiles_to_ogb_graph(smile)
    if datum is None:
        return None
    model = _get_graphmvp()
    lote = Batch.from_data_list([datum]).to(_get_device())
    with torch.no_grad():
        representacao_por_atomo = model(lote.x, lote.edge_index, lote.edge_attr)
    embedding = global_mean_pool(representacao_por_atomo, lote.batch)
    return embedding.squeeze().cpu().numpy()


def embed_gin(smile):
    """Embedding of the GIN pretrained by context prediction (300 dims, mean pooling over
    atoms). No explicit hydrogens, the same convention as the original pretrain-gnns
    loader.py."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool

    datum = _smiles_to_pyg_graph(smile, add_hs=False)
    if datum is None:
        return None
    model = _get_gin_pretrain()
    lote = Batch.from_data_list([datum]).to(_get_device())
    with torch.no_grad():
        representacao_por_atomo = model(lote.x, lote.edge_index, lote.edge_attr)
    embedding = global_mean_pool(representacao_por_atomo, lote.batch)
    return embedding.squeeze().cpu().numpy()


def embed_molebert(smile):
    """Mole-BERT embedding (pretrained GIN, 300 dims, mean pooling over atoms). No explicit
    hydrogens - the same implicit convention as Mole-BERT's original loader.py."""
    import torch
    from torch_geometric.data import Batch
    from torch_geometric.nn import global_mean_pool

    datum = _smiles_to_pyg_graph(smile, add_hs=False)
    if datum is None:
        return None
    model = _get_molebert()
    lote = Batch.from_data_list([datum]).to(_get_device())
    with torch.no_grad():
        representacao_por_atomo = model(lote.x, lote.edge_index, lote.edge_attr)
    embedding = global_mean_pool(representacao_por_atomo, lote.batch)
    return embedding.squeeze().cpu().numpy()