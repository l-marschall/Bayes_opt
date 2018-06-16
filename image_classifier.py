from fastai.fastai.imports import *
from fastai.fastai.transforms import *
from fastai.fastai.conv_learner import *
from fastai.fastai.model import *
from fastai.fastai.dataset import *
from fastai.fastai.sgdr import *
#from fastai.fastai.plots import *

torch.manualSeed(40)

def fit_eval_imageclassifier(max_zoom, lr, ps, epochs, cycle_len, cycle_mult): 
    # load data and model 
    sz = 224 
    PATH = "data/dogscats/"
    arch = resnet34
    tfms = tfms_from_model(resnet34, sz, aug_tfms=transforms_side_on, max_zoom=max_zoom)
    data = ImageClassifierData.from_paths(PATH, tfms=tfms)
    learn = ConvLearner.pretrained(arch, data, precompute=True, ps=ps)
    # train and evaluate model 
    learn.fit(lr, epochs, cycle_len=cycle_len, cycle_mult=cycle_mult)
    log_preds, y = learn.TTA()
    probs = np.mean(np.exp(log_preds), 0)
    return accuracy_np(probs, y)