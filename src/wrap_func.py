import torch
import numpy as np

def dilig(
    func
):
    '''
    dilig == dilient_in_types
    入力と出力の型を一致させる。
    '''
    def wrapped_func(img, *args, **kwargs):
        if torch.is_tensor(img):
            # actionにはnumpy配列で入力する
            out = func(img.cpu().numpy())
            out = torch.from_numpy(out)
            return out.type(img.dtype)
        out = func(img, *args, **kwargs)
        out = out.astype(img.dtype)
        return out
    return wrapped_func