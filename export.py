import torch
from model.deeplab import Res_Deeplab

if __name__ == "__main__":
    x = torch.randn(1, 3, 224, 224)
    model = Res_Deeplab(num_classes=19)
    try:
        torch.export.export(model, (x,))
        print("[JIT] torch.export successed.")
        exit(0)
    except Exception as e:
        print("[JIT] torch.export failed.")
        raise e
