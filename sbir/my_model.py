from options import Option
from model.model import Model
from utils.util import load_checkpoint
from data_utils.utils import preprocess
import torch.nn.functional as F


class SBIRModel:
    def __init__(self):
        args = Option().parse()
        model = Model(args)
        checkpoint = load_checkpoint(args.load)
        cur = model.state_dict()
        new = {k: v for k, v in checkpoint['model'].items() if k in cur.keys()}
        cur.update(new)
        model.load_state_dict(cur)
        self.model = model.cpu()

    def extract_embedding(self, image_path):
        im = preprocess(image_path=image_path, img_type="im")
        im = im.unsqueeze(0)
        im, im_idxs = self.model(im, None, 'test', only_sa=True)
        im = im.squeeze()
        im = im[:1]
        im = F.normalize(im)
        im = im[0]
        print(im.shape)

    def extract_sketch_embedding(self, image_path):
        sk = preprocess(image_path=image_path, img_type="sk")
        sk = sk.unsqueeze(0)
        sk, sk_idxs = self.model(sk, None, 'test', only_sa=True)
        sk = sk.squeeze()
        sk = sk[:1]
        sk = F.normalize(sk)
        sk = sk[0]
        print(sk.shape)


if __name__ == "__main__":
    model = SBIRModel()
    model.extract_embedding("../../data/keyframes/Keyframes_L01/keyframes/L01_V001/001.jpg")
    model.extract_sketch_embedding("../../data/keyframes/Keyframes_L01/keyframes/L01_V001/001.jpg")
