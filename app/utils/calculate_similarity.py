from torch import nn

cos = nn.CosineSimilarity(dim=1)

def cos_simil(input_1, input_2):
    result = (cos(input_1, input_2) + 1.) / 2.
    return result