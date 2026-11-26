"""
Gera e salva embeddings Uni-Mol a partir de uma base de dados CSV com SMILES e rótulos.

Saídas:
    embeddings_unimol.npy  -> matriz X (n_moleculas, dim_embedding)
    rotulos_unimol.npy     -> vetor y (n_moleculas,)
"""

import sys
sys.path.append(r"C:\Users\Luiggy\PIBIC-Bioinformatica\PIBIC-Luiggy\Treinamento do modelo\Scripts auxiliares")

import Dados
import Embeddings

import pandas as pd
import numpy as np
#from rdkit import Chem
import unimol_tools


# ---------------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------------
PATH_BD = r"C:\Users\Luiggy\PIBIC-Bioinformatica\PIBIC-Luiggy\Bases de dados\Treinamento - limpas\Base_Celegans\concatenada.csv"
COLUNA_DADOS = "canonical_smiles"
COLUNA_ROTULOS = "lifespan_extender"

OUT_EMBEDDINGS = r"C:\Users\Luiggy\PIBIC-Bioinformatica\PIBIC-Luiggy\Treinamento do modelo\Embeddings_salvos\Embeddings_Celegans\embeddings_unimol.npy"
OUT_ROTULOS = r"C:\Users\Luiggy\PIBIC-Bioinformatica\PIBIC-Luiggy\Treinamento do modelo\Embeddings_salvos\Embeddings_Celegans\rotulos_unimol.npy"


#def _smiles_valido(smile):
 #   """Retorna True se o RDKit conseguir parsear o SMILES."""
 #   return Chem.MolFromSmiles(smile) is not None


def gerar_vetores_unimol(path_bd, coluna_dados, coluna_rotulos):
    """Lê o CSV, filtra SMILES inválidos e gera embeddings Uni-Mol em lote.

    Retorna
    -------
    X : np.ndarray, shape (n_moleculas_validas, dim_embedding)
    y : np.ndarray, shape (n_moleculas_validas,)
    """
    df_bd = pd.read_csv(rf"{path_bd}")

    total = len(df_bd)
    mascara_valida = df_bd[coluna_dados].apply(_smiles_valido)
    n_invalidos = total - mascara_valida.sum()

    if n_invalidos > 0:
        print(f"[aviso] {n_invalidos}/{total} SMILES inválidos foram descartados.")

    df_valido = df_bd[mascara_valida].reset_index(drop=True)
    smiles = df_valido[coluna_dados].tolist()
    rotulos = df_valido[coluna_rotulos].values

    print(f"Gerando embeddings Uni-Mol para {len(smiles)} moléculas...")
    X = Embeddings.gerar_embeddings_unimol_batch(smiles)
    y = rotulos

    return X, y


if __name__ == "__main__":
    X_unimol, y_unimol = gerar_vetores_unimol(PATH_BD, COLUNA_DADOS, COLUNA_ROTULOS)

    np.save(OUT_EMBEDDINGS, X_unimol)
    np.save(OUT_ROTULOS, y_unimol)

    print(f"Embeddings salvos em: {OUT_EMBEDDINGS}  (shape={X_unimol.shape})")
    print(f"Rótulos salvos em:    {OUT_ROTULOS}  (shape={y_unimol.shape})")