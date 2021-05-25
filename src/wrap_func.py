import torch


def dilig(
    func
):
    '''dilig == dilient_in_types'''
    def wrapped_func(img):
        if torch.is_tensor(img):
            out = func(img.cpu())
            return torch.from_numpy(out)
        return func(img)
    return wrapped_func