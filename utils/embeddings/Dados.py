import pandas as pd
import numpy as np
import Embeddings

# recebe o caminho para a base de dados em CSV, as colunas dos dados e dos rótulos e o modelo de geração
def gerar_vetores(path_bd, coluna_dados, coluna_rotulos, modelo):
    df_bd = pd.read_csv(rf"{path_bd}")
    smiles = df_bd[coluna_dados].tolist()
    rotulos = df_bd[coluna_rotulos].values
    vetorizador = ''
    if modelo == 'molformer':
        vetorizador = Embeddings.gerar_embedding_molformer
    elif modelo == 'chembert':
        vetorizador = Embeddings.gerar_embedding_chembert
    elif modelo == 'unimol':
        vetorizador = Embeddings.gerar_embedding_unimol
    elif modelo == 'selformer':
        vetorizador = Embeddings.gerar_embedding_selformer
    else:
        vetorizador = Embeddings.gerar_fingerprint
    for i in range(len(smiles)):
        smiles[i] = vetorizador(smiles[i])
    X = np.vstack(smiles)
    y = rotulos
    return X, y
#def predict_smile #ADICIONAR
path = r"C:\Users\Luiggy\PIBIC-Bioinformatica\PIBIC-Luiggy\Bases de dados\Treinamento - limpas\Base_Celegans\concatenada.csv"
#X_molformer,y_molformer = gerar_vetores(path, "canonical_smiles", "lifespan_extender", "molformer")
#np.save("embeddings_molformer.npy", X_molformer)
#np.save("rotulos_molformer.npy", y_molformer)
#X_chembert,y_chembert = gerar_vetores(path, "canonical_smiles", "lifespan_extender", "chembert")
#np.save("embeddings_chembert.npy", X_chembert)
#np.save("rotulos_chembert.npy", y_chembert)
#X_fingerprints,y_fingerprints = gerar_vetores(path, "canonical_smiles", "lifespan_extender", "fingerprint")
#np.save("fingerprints.npy", X_fingerprints)
#np.save("rotulos_fingerprints.npy", y_fingerprints)

X_unimol,y_unimol = gerar_vetores(path, "canonical_smiles", "lifespan_extender", "unimol")
np.save("embeddings_unimol.npy", X_unimol)
np.save("rotulos_unimol.npy", y_unimol)

#X_selformer,y_selformer = gerar_vetores(path, "canonical_smiles", "lifespan_extender", "selformer")
#np.save("embeddings_selformer.npy", X_selformer)
#np.save("rotulos_selformer.npy", y_selformer)