import torch


def min_distance_between_roadpoints(
    roadpoints: torch.Tensor,
    roadpoint1: torch.Tensor,
    roadpoint2: torch.Tensor,
) -> torch.Tensor:
    """Placeholder: compute a simple Euclidean distance between two roadpoints.
    This is a stub to keep the module valid; replace with graph distance if needed.

    Args:
        roadpoints: All road points (unused here).
        roadpoint1: Coordinate tensor [x, y].
        roadpoint2: Coordinate tensor [x, y].
    Returns:
        Tensor scalar distance.
    """
    dx = roadpoint1[0] - roadpoint2[0]
    dy = roadpoint1[1] - roadpoint2[1]
    return torch.sqrt(dx * dx + dy * dy)


def Build_Matrix(roads: torch.Tensor) -> torch.Tensor:
    """Placeholder: pairwise Euclidean distances between road points.
    Replace with road network distances if required.
    """
    n = int(roads.shape[0])
    distances = torch.zeros(n, n, dtype=roads.dtype, device=roads.device if roads.is_cuda else None)
    for i in range(n):
        for j in range(n):
            dx = roads[i, 0] - roads[j, 0]
            dy = roads[i, 1] - roads[j, 1]
            distances[i, j] = torch.sqrt(dx * dx + dy * dy)
    return distances
