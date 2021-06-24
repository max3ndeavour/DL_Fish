import torch
import numpy as np
import argparse
import pandas as pd
import sys
import os
from torch import nn
from torch.nn import functional as F
import tqdm
import pprint
# from DeepFish.src import utils as ut
import torchvision
from haven import haven_utils as hu
from haven import haven_chk as hc

from DeepFish.src import  models
from torch.utils.data import DataLoader
from torch.utils.data.sampler import RandomSampler
from DeepFish.src import wrappers
from haven import haven_wizard as hw
import pickle
import DeepFish.exp_configs as exp_configs

def trainval(checkUploadCanceled, onProgressUpload, onFishCounted, item_id, original_video_path, predicted_video_path, model_dir="DeepFish/model"):
    """
    exp_dict: dictionary defining the hyperparameters of the experiment
    savedir: the directory where the experiment will be saved
    args: arguments passed through the command line
    """
    exp_dict = exp_configs.EXP_GROUPS
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = 'cuda'
    torch.cuda.manual_seed_all(seed)
    assert torch.cuda.is_available(), 'cuda is not, available please run with "-c 0"'
    print('Running on device: %s' % device)
    
    model_original = models.get_model(exp_dict['loc'][0]['model'], exp_dict=exp_dict['loc'][0]).cuda()
    opt = torch.optim.Adam(model_original.parameters(), 
                        lr=1e-5, weight_decay=0.0005)

    model = wrappers.get_wrapper(exp_dict['loc'][0]["wrapper"], model=model_original, opt=opt).cuda()


    model_path = os.path.join(model_dir, "model_state_dict.pth")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.predict_video(checkUploadCanceled, onProgressUpload, onFishCounted, item_id, original_video_path, predicted_video_path)
    print("FINISH!!!!!")
