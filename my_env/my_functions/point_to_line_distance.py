import torch
import utils


@utils.count_runtime
def distance_point_to_line_segment(px, py, ax, ay, bx, by):
    ## px, py (n,) ax,bx,ay,by (k,)

    dx1, dy1 = (bx - ax).view(-1,1), (by - ay).view(-1,1)
    # dx1 (k,1) dy1 (k,1)
    # AP
    dx2, dy2 = px.view(1, -1) - ax.view(-1,1), py.view(1,-1) - ay.view(-1,1)
    # dx2 (k,n) dy2 (k,n)


    length_squared = dx1 ** 2 + dy1 ** 2
    # length_squared (k)
    # # 线段长度为零
    # if length_squared == 0:
    #     return math.sqrt(dx2 ** 2 + dy2 ** 2)

    # 投影标量 t ∈ [0, 1]
    t = ((dx2 * dx1 + dy2 * dy1) / length_squared).clamp(0, 1)

    # t (k,n)

    # 投影坐标
    projx = ax.reshape(-1,1) + t * dx1
    projy = ay.reshape(-1,1) + t * dy1

    # P到投影点距离
    return torch.sqrt((px.reshape(1,-1) - projx) ** 2 + (py.reshape(1,-1) - projy) ** 2) # (k,n)

if __name__ == '__main__':
    px, py = torch.tensor([0]), torch.tensor([0])
    x1, y1 = torch.tensor([1, 0, 3]), torch.tensor([1, 3, 2])
    x2, y2 = torch.tensor([1, 1, 2]), torch.tensor([3, 3, 6])

    print(distance_point_to_line_segment(px, py, x1, y1, x2, y2))
