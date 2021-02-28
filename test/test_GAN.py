import nibabel as nib
import numpy as np
import setting as st
import torch
from torch.autograd import Variable
import torch.nn as nn
import utils as ut
import os
from scipy import stats
from sklearn.metrics import confusion_matrix
from plot import *

def test(config, fold, model_1, model_2,loader, hyperParam, dir_to_load, dir_heatmap, dir_confusion):
    """ free all GPU memory """
    torch.cuda.empty_cache()
    test_loader = loader

    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    if model_dir !=None:
        model_1.load_state_dict(torch.load(model_dir))
    model_1.eval()
    dict_result = ut.eval_classification_model_VAGAN(config, fold, 1000, test_loader, model_1, hyperParam, dir_heatmap, confusion_save_dir=dir_confusion)
    ut.MRI_plot_fake_img(config, fold, 1001, test_loader, model_1, hyperParam, dir_heatmap)
    return dict_result
