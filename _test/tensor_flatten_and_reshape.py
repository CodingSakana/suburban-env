
import torch
import utils

@utils.count_runtime(show_args=False)
def flatten_and_reshape(tensor):
    shape = tensor.shape
    print(shape)
    flattened = tensor.flatten()
    return flattened.reshape(shape)


if __name__ == '__main__':
    flatten_and_reshape(torch.randn(3, 128, 128))
    flatten_and_reshape(torch.randn(3, 3, 3))
    flatten_and_reshape(torch.randn(3, 128, 128))
    flatten_and_reshape(torch.randn(3, 3, 3))
    flatten_and_reshape(torch.randn(3, 128, 128))