import dill
import numpy as np
from sklearn.decomposition import PCA

def read_pkl(path):
    with open(path, 'rb') as f:
        data = dill.load(f)
    return data

def reduce_embeddings(medicine, embedding_dim):
    codes = list(medicine.keys())
    embeddings = np.array([medicine[code] for code in codes])

    # PCA降维
    pca = PCA(n_components=embedding_dim)
    reduced_embeddings = pca.fit_transform(embeddings)
    # 重新构造成字典
    medicine_embs = {code: reduced_embeddings[i] for i, code in enumerate(codes)}
    return medicine_embs

def load_text_embeddings():
    # read diagnosis code embeddings
    embs = read_pkl('F:\Project\TRANS-main\chatgpt_desc\diag_embs.pkl')#使用 read_pkl 函数读取诊断-处方嵌入文件。
    diag_embs = {}
    for code, emb in embs.items():
        #print(code[2:],emb)
        diag_embs[code[2:].replace('.', '')] = emb#创建一个字典 diag_embs，将每个代码（去掉前缀并替换点）与其对应的嵌入关联。

    # read procedure code embeddings
    embs = read_pkl('F:\Project\TRANS-main\chatgpt_desc\pro_embs.pkl')
    proce_embs = {}
    for code, emb in embs.items():
        proce_embs[code[2:].replace('.', '')] = emb
#同样地，读取手术-处方嵌入文件并创建字典 proce_embs，将程序代码与嵌入关联。

    # read atc code embeddings
    embs = read_pkl('F:\Project\TRANS-main\chatgpt_desc\\atc_embs.pkl')
    atc_embs = {}#读取 ATC-处方嵌入文件，创建字典 atc_embs。实质是RXNORM_embeding
    for code, emb in embs.items():
        atc_embs[code[2:].replace('.', '')] = emb
            #将所有匹配的 NDC 代码与相应的嵌入关联。
    return diag_embs, proce_embs, atc_embs
