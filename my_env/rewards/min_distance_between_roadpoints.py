import torch


def min_distance_between_roadpoints(roadpoints:torch.tensor,roadpoint1:torch.tensor, roadpoint2:torch.tensor):

def Build_Matrix(roads:torch.tensor):
    distances = torch.zeros(roads.shape[0], roads.shape[0])
    for i in roads.shape[0]:
        for j in roads.shape[0]:
            distances[i][j] =




