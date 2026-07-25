
import umap
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn.metrics.pairwise import euclidean_distances
import heapq

def save_embeddings_to_csv(names, features, filename):
    df = pd.DataFrame(features.numpy())  # 转为 DataFrame
    df.insert(0, 'name', names)          # 插入名称列
    df.to_csv(filename, index=False)
    print(f"Saved to {filename}")

def load_embeddings_from_csv(filename):
    df = pd.read_csv(filename)
    names = df['name'].tolist()
    features = torch.tensor(df.drop(columns='name').values, dtype=torch.float)
    return names, features

def find_similar_items(target_name, target_list, target_features, compare_list, compare_features, topk=10):
    idx = target_list.index(target_name)
    target_vec = target_features[idx]
    sims = F.cosine_similarity(target_vec.unsqueeze(0), compare_features, dim=1)
    sorted_idx = sims.argsort(descending=True)
    if target_list == compare_list:
        sorted_idx = sorted_idx[1:topk+1]  # 排除自己
    else:
        sorted_idx = sorted_idx[:topk]
    print(f"\n与 '{target_name}' 最相似的实体：")
    for i in sorted_idx:
        print(f"{compare_list[i]} \t 相似度: {sims[i].item():.4f}")

def get_top_k_similar_drug_pairs_euclidean(drugs_list, drugs_features, top_k=10):
    valid_drugs_list = drugs_list[1:]
    valid_drugs_features = drugs_features[1:]

    # 计算欧氏距离
    distance_matrix = euclidean_distances(valid_drugs_features)
    similar_pairs = []

    num_drugs = len(valid_drugs_list)
    for i in range(num_drugs):
        for j in range(i + 1, num_drugs):
            dist = distance_matrix[i][j]
            heapq.heappush(similar_pairs, (dist, (valid_drugs_list[i], valid_drugs_list[j], dist)))

    top_k_pairs = heapq.nsmallest(top_k, similar_pairs)
    return [(drug1, drug2, dist) for dist, (drug1, drug2, dist) in top_k_pairs]

import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


def get_top_related_pairs(drugs_list, drugs_features, top_k=10):
    """
    从所有药物中选出欧氏距离最小的 top_k 对。
    返回：(药物对列表，涉及的药物名集合)
    """
    dist_matrix = euclidean_distances(drugs_features)
    drug_pairs = []

    num = len(drugs_list)
    for i in range(num):
        for j in range(i + 1, num):
            dist = dist_matrix[i][j]
            heapq.heappush(drug_pairs, (dist, (i, j)))

    # 获取 top_k 对
    top_pairs = heapq.nsmallest(top_k, drug_pairs)

    # 提取涉及的药物索引
    selected_indices = set()
    for _, (i, j) in top_pairs:
        selected_indices.add(i)
        selected_indices.add(j)

    selected_indices = sorted(list(selected_indices))
    selected_names = [drugs_list[i] for i in selected_indices]
    selected_features = drugs_features[selected_indices]
    selected_dist = euclidean_distances(selected_features)

    return selected_names, selected_dist


def plot_similarity_heatmap(drug_names, dist_matrix):
    """
    画热力图，颜色反映“距离”，距离越小颜色越深（使用 1 / (1 + dist) 转相似度）
    """
    print(dist_matrix)
    similarity = 1 / (1 + dist_matrix)  # 距离越小，相似度越大
    mask = ~np.eye(similarity.shape[0], dtype=bool)
    sim_values = similarity[mask]
    sim_min = sim_values.min()
    sim_max = sim_values.max()

    # Step 3: 对非对角线元素归一化
    similarity_normalized = similarity.copy()
    similarity_normalized[mask] = (sim_values - sim_min) / (sim_max - sim_min + 1e-8)
    similarity_normalized[~mask] = 1.0  # 自身相似度设为1
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 20
    plt.figure(figsize=(12, 10))
    sns.heatmap(similarity_normalized, xticklabels=drug_names, yticklabels=drug_names,
                cmap='viridis_r', square=True)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('fig4.png')
    plt.show()


seed = 42

# 分别读取三个文件
conds_list, conds_features = load_embeddings_from_csv('conds_embeddings.csv')
proc_list, proc_features = load_embeddings_from_csv('proc_embeddings.csv')
drugs_list, drugs_features = load_embeddings_from_csv('drugs_embeddings.csv')

top_drug_pairs = get_top_k_similar_drug_pairs_euclidean(conds_list, conds_features, top_k=20)
for i, (drug1, drug2, dist) in enumerate(top_drug_pairs, 1):
    print(f"{i}. {drug1} - {drug2}: Euclidean Distance = {dist:.4f}")


#find_similar_items("S01F", drugs_list, drugs_features, drugs_list, drugs_features)

selected_names, selected_dist = get_top_related_pairs(conds_list[1:], conds_features[1:], top_k=20)
plot_similarity_heatmap(selected_names, selected_dist)


print(conds_list)
print(proc_list)
print(drugs_list)

print(len(proc_list))


#test_features = np.concatenate((test_features1, test_features2, test_features3), axis=0)

conds_pca = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=seed).fit_transform(conds_features)
proc_pca = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=seed).fit_transform(proc_features)
drugs_pca = umap.UMAP(n_neighbors=5, min_dist=0.3, random_state=seed).fit_transform(drugs_features)

colors = ['#F94144', '#277DA1', '#90BE6D', '#F3722C', '#7400B8', '#70D6FF', '#FF70A6', '#0D3B66', '#4D908E',
          '#577590', '#43AA8B', '#C38E70', '#F9844A', '#FAF0CA', '#E9FF70']

# 创建一个散点图
plt.figure(figsize=(8, 6))
plt.scatter(conds_pca[:, 0], conds_pca[:, 1], c=colors[0], s=5)
plt.scatter(proc_pca[:, 0], proc_pca[:, 1], c=colors[1], s=5)
plt.scatter(drugs_pca[:, 0], drugs_pca[:, 1], c=colors[2], s=5)

# 添加标签和标题
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA Visualization with Labels')
plt.show()