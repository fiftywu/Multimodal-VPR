from .SegNet import SegNet
from .SegTransNet import SegTransNet


def create_model(opt):
    if opt.mode == "Seg":
        model = SegNet(opt)
    elif opt.mode == "SegTrans":
        model = SegTransNet(opt)
    else:
        raise NotImplementedError
    return model