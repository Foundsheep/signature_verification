import torch

def calculate_similarity(embed_1, embed_2):
    if embed_1.dim() == 1:
        embed_1 = embed_1.unsqueeze(0)
    if embed_2.dim() == 1:
        embed_2 = embed_2.unsqueeze(0)
    embed_1 = embed_1 / torch.linalg.vector_norm(embed_1, dim=1).unsqueeze(1)
    embed_2 = embed_2 / torch.linalg.vector_norm(embed_2, dim=1).unsqueeze(1)

    score_mtx = torch.matmul(embed_1, embed_2.T)
    score_vec = score_mtx.diag()
    return score_vec