import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from vietocr.tool.translate import build_model, translate, translate_beam_search, predict
from vietocr.tool.utils import download_weights
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg



class NewPredictor(Predictor):
    def predict(self, img, return_prob=False):
        # process input
        img = img[np.newaxis, ...]
        img = torch.FloatTensor(img)
        
        img = img.to(self.config['device'])

        if self.config['predictor']['beamsearch']:
            sent = translate_beam_search(img, self.model)
            s = sent
            prob = None
        else:
            s, prob = translate(img, self.model)
            s = s[0].tolist()
            prob = prob[0]

        s = self.vocab.decode(s)
        
        if return_prob:
            return s, prob
        else:
            return s

def ocr(img):
    config = Cfg.load_config_from_name('vgg_transformer')
    # config['weights'] = './weights/transformerocr.pth'
    config['weights'] = 'https://drive.google.com/uc?id=13327Y1tz1ohsm5YZMyXVMPIOjoOA0OaA' # thay bằng link của Tú
    config['cnn']['pretrained']=False
    config['device'] = 'cuda:0'
    config['predictor']['beamsearch']=False

    detector = NewPredictor(config)
    s = detector.predict(img)
    return s