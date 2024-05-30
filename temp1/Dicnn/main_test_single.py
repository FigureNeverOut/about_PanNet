import torch.nn.modules as nn
import torch
import cv2
import numpy as np
from model import Dicnn
import h5py
import scipy.io as sio
import os

def calculate_metrics(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    mae = np.mean(np.abs(y_true - y_pred))
    return mse, mae

def get_edge(data):  # get high-frequency
    rs = np.zeros_like(data)
    if len(rs.shape) == 3:
        for i in range(data.shape[2]):
            rs[:, :, i] = data[:, :, i]
    else:
        rs = data
    return rs
def load_set(file_path):
    data = sio.loadmat(file_path)  # HxWxC=256x256x8

    # tensor type:
    gt = torch.from_numpy(data['gt'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms_hp = torch.from_numpy(get_edge(data['ms'] / 2047.0)).permute(2, 0, 1)  # CxHxW= 8x64x64

    # 使用transpose而不是permute处理'pan'
    pan_hp = torch.from_numpy(get_edge(data['pan'] / 2047.0)).transpose(1, 0)  # HxW = 256x256

    return gt, lms, ms_hp, pan_hp



ckpt = "Weights/1000.pth"   # chose model

def test():
    file_path = "test_data/new_data6.mat"
    y_true, lms, ms_hp, pan_hp = load_set(file_path)

    model = Dicnn().eval()   # fixed, important!
    weight = torch.load(ckpt)  # load Weights!
    model.load_state_dict(weight) # fixed

    with torch.no_grad():

        x1, x2, x3 = lms, ms_hp, pan_hp   # read data: CxHxW (numpy type)
        print(x1.shape)
        x1 = x1.cpu().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x2 = x2.cpu().unsqueeze(dim=0).float()  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        x3 = x3.cpu().unsqueeze(dim=0).unsqueeze(dim=1).float()  # convert to tensor type: 1x1xHxW

        hp_sr = model(x2, x3)  # tensor type: CxHxW
        sr = x1 + hp_sr        # tensor type: CxHxW

        # calculate and print metrics
        mse, mae = calculate_metrics(y_true.cpu().numpy(), sr.cpu().numpy())
        print(f"MSE: {mse}, MAE: {mae}")

        # convert to numpy type with permute and squeeze: HxWxC (go to cpu for easy saving)
        sr = torch.squeeze(sr).permute(1, 2, 0).cpu().detach().numpy()  # HxWxC

        print(sr.shape)
        save_name = os.path.join("test_results", "new_data6_pannet.mat")  # fixed! save as .mat format that will used in Matlab!
        sio.savemat(save_name, {'new_data6_pannet': sr})  # fixed!

if __name__ == '__main__':
    file_path = "test_data/new_data6.mat"
    test()   # recall test function