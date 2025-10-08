import os
import cv2
import datetime

import sys
import lpips
import numpy
import torch
import time
import math
import warnings
import random
import pandas as pd
import numpy as np
import sigpy.mri as mr
import matplotlib.pyplot as plt
import torch.nn.functional as F

from scipy.ndimage import rotate
from scipy.io import savemat
from scipy.io import loadmat
from scipy.ndimage import label
from scipy.ndimage import convolve
from scipy.ndimage import binary_dilation
from scipy.ndimage import gaussian_laplace
from PIL import Image
from skimage.util import img_as_ubyte
from skimage import io
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

from utils.HaarPSI import HaarPSI
from utils.DISTS.DISTS import compute_DISTS
from utils.ESPIRiT.espirit import espirit
from utils.shearUtils import *

warnings.filterwarnings("ignore")
torch.autograd.set_detect_anomaly(True)

def load_data_and_mask(filename, maskfilename, Ker_Size=(5,5), normalizedCoeByEnergy=True):
    import main
    now = datetime.datetime.now()
    year_month = now.strftime('%Y_%m')
    year_month_day = now.strftime('%Y_%m_%d')
    dt = now.strftime('%Y_%m_%d_%H_%M_%S')
    dt = f"{dt}_{filename}"
    save_path = os.path.join('./result', year_month, year_month_day, dt)
    os.makedirs(save_path, exist_ok=True)

    data_file = f'./data/{filename}.mat'
    data = loadmat(data_file)
    ksfull = data['ksfull'].astype(np.complex64)

    row, col, coils = ksfull.shape

    tokens = maskfilename.split('_')
    random_or_uniform = tokens[1]

    size_str = f"{row}x{col}"
    maskfilepath = os.path.join('mask', size_str, random_or_uniform, f"{maskfilename}.png")

    mask = Image.open(maskfilepath)
    mask = np.array(mask).astype(np.int)
    mask = rotate(mask, angle=90, reshape=True)
    io.imsave(os.path.join(save_path, f"{maskfilename}.png"), img_as_ubyte(mask / np.max(mask)))
    mask = np.tile(mask[:, :, np.newaxis], (1, 1, coils))

    if normalizedCoeByEnergy:
        ksfull = NormalizedCoeByEnergy(ksfull)

    ref = sos(IFFT2_3D_N(ksfull))
    io.imsave(os.path.join(save_path, '1-reference.png'), img_as_ubyte(ref / np.max(ref)))

    ksdata = mask * ksfull
    un_image = sos(IFFT2_3D_N(ksdata))
    io.imsave(os.path.join(save_path, 'aliasing.png'), img_as_ubyte(un_image / np.max(un_image)))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mask = torch.from_numpy(mask).to(device)
    ksdata = torch.from_numpy(ksdata).to(device=device, dtype=torch.complex64)
    ksfull = torch.from_numpy(ksfull).to(device=device, dtype=torch.complex64)

    main.ksfull, main.ksdata, main.mask, main.save_path = ksfull, ksdata, mask, save_path

    start_time = time.time()
    # sensitivity, sensitivityLi, ACS = get_sensitivity(ksdata, mask)
    sensitivity, sensitivityLi, ACS = get_ESPIRiT_sensitivity(ksdata, mask)
    main.t_sense = time.time() - start_time

    start_time = time.time()
    Ker, Ker_Tra = Kernel_Estimation(ksdata, ACS, Ker_Size)
    main.t_kernel = time.time() - start_time

    start_time = time.time()
    Lip_C = Est_Lip_Ker_Mat_C_GPU(ksdata, Ker, Ker_Size)
    main.t_lip = time.time() - start_time

    main.sensitivity, main.sensitivityLi, main.Ker, main.Ker_Tra, main.Lip_C = sensitivity, sensitivityLi, Ker, Ker_Tra, Lip_C

    return save_path

def Est_Lip_Ker_Mat_C_GPU(ful_kspace, ker, ker_size=(7, 7)):
    def compute_Lip(ful_kspace, ker, ker_size):
        row, col, coi = ful_kspace.shape
        ker_matrix = torch.zeros((row, col), dtype=torch.float32, device=ful_kspace.device)
        ker_matrix_circ_fft = torch.zeros((row, col, coi, coi), dtype=torch.complex64, device=ful_kspace.device)
        for coi_num in range(coi):
            for i in range(coi):
                ker_matrix[:ker_size[0], :ker_size[1]] = ker[:, :, i, coi_num]
                ker_matrix_circ = torch.roll(ker_matrix, shifts=(-(ker_size[0] // 2), -(ker_size[1] // 2)), dims=(0, 1))
                ker_matrix_circ_fft[:, :, i, coi_num] = torch.fft.fft2(ker_matrix_circ)
        eig_vals = np.linalg.norm(ker_matrix_circ_fft.cpu().numpy().reshape(row * col, coi, coi), ord=2, axis=(1, 2))
        lip = eig_vals.max()
        return lip
    if ker.dim() == 4:
        return compute_Lip(ful_kspace, ker, ker_size)
    elif ker.dim() == 5:
        Lip = -1
        for i in range(ker.shape[-1]):
            Lip = max(Lip, compute_Lip(ful_kspace, ker[..., i], ker_size))
        return Lip
    else:
        return None

def Kernel_Estimation_Real_Imag(ACS_Data, Ker_Size=(7, 7), Itr=800, ConjugateTran=False):
    Mat_Data, Tag_Vec = Spirit_Kernel(Ker_Size, ACS_Data)
    if ConjugateTran:
        ACS_Data = ACS_Data[::-1, ::-1, ...]
        MTmp, TTmp = Spirit_Kernel(Ker_Size, ACS_Data)
        Mat_Data, Tag_Vec = np.concatenate((Mat_Data, MTmp), axis=0), np.concatenate((Tag_Vec, TTmp), axis=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Mat_Data = torch.tensor(Mat_Data, dtype=torch.complex64, device=device)
    Tag_Vec = torch.tensor(Tag_Vec, dtype=torch.complex64, device=device)

    Mat_Data_r = Mat_Data.real
    Mat_Data_i = Mat_Data.imag
    Tag_Vec_r = Tag_Vec.real
    Tag_Vec_i = Tag_Vec.imag
    w_r = np.zeros((Mat_Data.shape[1], Mat_Data.shape[2]), dtype=np.float32)
    w_i = np.zeros((Mat_Data.shape[1], Mat_Data.shape[2]), dtype=np.float32)

    for i in range(Mat_Data.shape[2]):
        w_r[:, i] = solve_linear_constrain(Mat_Data_r[:, :, i], Tag_Vec_r[:, i], Itr)
        w_i[:, i] = solve_linear_constrain(Mat_Data_i[:, :, i], Tag_Vec_i[:, i], Itr)

    Ker_out = None
    Ker_Tra_out = None

    for w in [w_r, w_i]:
        Con_w = np.sum(w ** 2, axis=0)
        Con_w_Max = np.max(Con_w)
        IndDel = int(np.floor((Ker_Size[0] - 1) / 2) * Ker_Size[1] + np.ceil(Ker_Size[1] / 2)) - 1

        # 初始化 Ker_w 和 Ker
        Ker_w = np.zeros((w.shape[0] + 1, w.shape[1]), dtype=np.float32)
        Ker = np.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=np.float32)
        Ker_Tra = np.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=np.float32)

        # 计算 Ker_Size 的积
        Ker_Size_prod = np.prod(Ker_Size)

        # 循环处理每一列
        for i in range(Ker_w.shape[1]):
            Target_loc = i * Ker_Size_prod + IndDel
            Ind_List = list(range(0, Target_loc)) + list(range(Target_loc + 1, Ker_Size_prod * Ker_w.shape[1]))

            # 更新 Ker_w
            Ker_w[Ind_List, i] = w[:, i]

            # 更新 Ker
            reshaped_Ker_w = Ker_w[:, i].reshape(Ker_w.shape[1], Ker_Size[0], Ker_Size[1])
            Ker[:, :, :, i] = np.transpose(reshaped_Ker_w, (2, 1, 0))
            RecFraOpe = transpose_filter(Ker[:, :, :, i])
            Ker_Tra[:, :, :, i] = RecFraOpe

        Trans_Ker = np.zeros_like(Ker_Tra)

        for coil in range(Ker_w.shape[1]):
            for i in range(Ker_w.shape[1]):
                Trans_Ker[:, :, i, coil] = Ker_Tra[:, :, coil, i]

        Ker_Tra = Trans_Ker
        KerW = np.sum(Ker ** 2, axis=(0, 1, 2))
        Ker_TraW = np.sum(Ker_Tra ** 2, axis=(0, 1, 2))

        Ker_W_Max = np.max(KerW * Ker_TraW)

        if Ker_W_Max > 1:
            Ker_Con = np.sqrt(Ker_W_Max + 1e-8)  # 使用 1e-8 作为 eps
            Ker = Ker / Ker_Con
            Ker_Tra = Ker_Tra / Ker_Con


        Ker_out = Ker if Ker_out is None else Ker_out + 1j * Ker
        Ker_Tra_out = Ker_Tra if Ker_Tra_out is None else Ker_Tra_out + 1j * Ker_Tra

    return Ker_out, Ker_Tra_out

def CCN_Kernel_Estimation(Full_KSpace, mask, Ker_Size=(7, 7), Low_Frequency_Lines=10):
    h, w, c = Full_KSpace.shape

    _, ACS = getCalibSize_1D_Edt(mask)
    ACS = np.abs(ACS[1] - ACS[0]) + 1

    if ACS < Low_Frequency_Lines:
        raise ValueError("ACS area is too small!")

    HalfACSL = ACS // 2
    HalfLFLs = Low_Frequency_Lines // 2

    HalfH, HalfW = h // 2, w // 2
    startR, endR = HalfH - HalfACSL, HalfH + HalfACSL
    startRL, endRL = HalfH - HalfLFLs, HalfH + HalfLFLs
    startCL, endCL = HalfW - HalfLFLs, HalfW + HalfLFLs
    Low_Frequency_Data = Full_KSpace[startRL: endRL, startCL: endCL, :]
    Low_Mat_Data, Low_Tag_Vec = Spirit_Kernel(Ker_Size, Low_Frequency_Data)

    Ker = torch.zeros((Ker_Size[0], Ker_Size[1], c, c, 2), device=Full_KSpace.device)
    Ker_Tra = torch.zeros((Ker_Size[0], Ker_Size[1], c, c, 2), device=Full_KSpace.device)

    High_Frequency_Data = Full_KSpace[startR: HalfH, : HalfW, :]
    Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel(Ker_Size, High_Frequency_Data)
    Mat_Data, Tag_Vec = torch.cat((Low_Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Low_Tag_Vec, Tag_Vec_Tmp), dim=0)
    High_Frequency_Data = Full_KSpace[HalfH: endR, HalfW:, :]
    Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel(Ker_Size, High_Frequency_Data)
    Mat_Data, Tag_Vec = torch.cat((Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Tag_Vec, Tag_Vec_Tmp), dim=0)
    Ker[..., 0], Ker_Tra[..., 0] = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

    High_Frequency_Data = Full_KSpace[startR: HalfH, HalfW:, :]
    Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel(Ker_Size, High_Frequency_Data)
    Mat_Data, Tag_Vec = torch.cat((Low_Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Low_Tag_Vec, Tag_Vec_Tmp), dim=0)
    High_Frequency_Data = Full_KSpace[HalfH: endR, : HalfW, :]
    Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel(Ker_Size, High_Frequency_Data)
    Mat_Data, Tag_Vec = torch.cat((Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Tag_Vec, Tag_Vec_Tmp), dim=0)
    Ker[..., 1], Ker_Tra[..., 1] = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

    return Ker, Ker_Tra

def LCL_Kernel_Estimation(Full_kspace, mask, Ker_Lines=5, Low_Frequency_Lines=10):
    h, w, c = Full_kspace.shape

    _, ACS = getCalibSize_1D_Edt(mask)
    ACS = np.abs(ACS[1] - ACS[0]) + 1

    if ACS < Low_Frequency_Lines:
        raise ValueError("ACS area is too small!")

    HalfLFLs = Low_Frequency_Lines // 2
    HalfH, HalfW = h // 2, w // 2
    startR, endR = HalfH - HalfLFLs, HalfH + HalfLFLs
    startC, endC = HalfW - HalfLFLs, HalfW + HalfLFLs
    Low_Frequency_Data = Full_kspace[startR: endR, startC: endC, :]
    Low_Mat_Data, Low_Tag_Vec = Spirit_Kernel((Ker_Lines, Ker_Lines), Low_Frequency_Data)

    HalfACSL = ACS // 2
    HalfKLLs = Ker_Lines // 2
    maxLines = max(HalfH, HalfW)
    KernelList = torch.zeros((maxLines - HalfACSL - HalfKLLs, Ker_Lines, Ker_Lines, c, c), dtype=torch.float32, device=mask.device)
    KernelTraList = torch.zeros((maxLines - HalfACSL - HalfKLLs, Ker_Lines, Ker_Lines, c, c), dtype=torch.float32, device=mask.device)

    for i in range(HalfACSL + 1, maxLines - HalfKLLs + 1):
        top = max(HalfH - i - HalfKLLs, 0)
        bottom = min(HalfH + i + HalfKLLs, h)
        left = max(HalfW - i - HalfKLLs, 0)
        right = min(HalfW + i + HalfKLLs, w)

        Mat_Data, Tag_Vec = Low_Mat_Data, Low_Tag_Vec

        if HalfH - i - HalfKLLs >= 0:
            High_Frequency_Data = Full_kspace[top: top + Ker_Lines, left: right, :]
            Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel((Ker_Lines, Ker_Lines), High_Frequency_Data)
            Mat_Data, Tag_Vec = torch.cat((Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Tag_Vec, Tag_Vec_Tmp), dim=0)

            High_Frequency_Data = Full_kspace[bottom - Ker_Lines: bottom, left: right, :]
            Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel((Ker_Lines, Ker_Lines), High_Frequency_Data)
            Mat_Data, Tag_Vec = torch.cat((Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Tag_Vec, Tag_Vec_Tmp), dim=0)

        if HalfW - i - HalfKLLs >= 0:
            High_Frequency_Data = Full_kspace[top: bottom, left: left + Ker_Lines, :]
            Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel((Ker_Lines, Ker_Lines), High_Frequency_Data)
            Mat_Data, Tag_Vec = torch.cat((Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Tag_Vec, Tag_Vec_Tmp), dim=0)

            High_Frequency_Data = Full_kspace[top: bottom, right - Ker_Lines: right, :]
            Mat_Data_Tmp, Tag_Vec_Tmp = Spirit_Kernel((Ker_Lines, Ker_Lines), High_Frequency_Data)
            Mat_Data, Tag_Vec = torch.cat((Mat_Data, Mat_Data_Tmp), dim=0), torch.cat((Tag_Vec, Tag_Vec_Tmp), dim=0)


        start = time.time()
        Ker, Ker_Tra = SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, (Ker_Lines, Ker_Lines))
        print(f"use-{time.time() - start}")
        KernelList[i - HalfACSL - 1, ...] = Ker
        KernelTraList[i - HalfACSL - 1, ...] = Ker_Tra


    return KernelList, KernelTraList

def Kernel_Estimation(Full_kspace, ACS_Line, Ker_Size=(7, 7)):
    center = Full_kspace.shape[0] // 2
    start = center - ACS_Line // 2
    end = center + ACS_Line // 2
    ACS_Data = Full_kspace[start:end, ...]
    Mat_Data, Tag_Vec = Spirit_Kernel(Ker_Size, ACS_Data)
    return SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size)

def Spirit_Kernel(Block, Acs):
    y, x, z = Acs.shape
    if Block[0] > y or Block[1] > x:
        raise ValueError("ACS area is too small!")
    Cal_mat_row_length = (y - Block[0] + 1) * (x - Block[1] + 1)
    Cal_mat_col_length = Block[0] * Block[1]

    Cal_mat = torch.zeros((Cal_mat_row_length, Cal_mat_col_length, z), dtype=Acs.dtype, device=Acs.device)

    count = 0
    for ii in range(Block[1]):  # x direction
        Block_x_head = ii
        Block_x_end = x - Block[1] + ii
        for jj in range(Block[0]):  # y direction
            Block_y_head = jj
            Block_y_end = y - Block[0] + jj
            # 从 Acs 中提取块并进行转置
            acs_block = Acs[Block_y_head:Block_y_end + 1, Block_x_head:Block_x_end + 1, :].permute(1, 0, 2)
            Cal_mat[:, count, :] = acs_block.reshape(Cal_mat_row_length, z)
            count += 1

    acs_block = Acs[Block[0] // 2:y - (Block[0] - 1) // 2, Block[1] // 2:x - (Block[1] - 1) // 2, :].permute(1, 0, 2)
    Target_vec = acs_block.reshape(Cal_mat_row_length, z)
    Temp_mat = Cal_mat.permute(0, 2, 1).reshape(Cal_mat_row_length, Cal_mat_col_length * z)
    IndDel = (Block[0] - 1) // 2 * Block[1] + Block[1] // 2
    Cal_mat = torch.zeros((Temp_mat.shape[0], Temp_mat.shape[1]-1, z), dtype=Temp_mat.dtype, device=Temp_mat.device)

    for i in range(z):
        Target_loc = i * Block[0] * Block[1] + IndDel
        list_ = list(range(Target_loc)) + list(range(Target_loc + 1, Block[0] * Block[1] * z))
        Cal_mat[:, :, i] = Temp_mat[:, list_]

    return Cal_mat, Target_vec

def SPIRiT_Kernel_Estimation(Mat_Data, Tag_Vec, Ker_Size, Itr=350):
    Mat_Data_r = Mat_Data.real
    Mat_Data_i = Mat_Data.imag
    Tag_Vec_r = Tag_Vec.real
    Tag_Vec_i = Tag_Vec.imag
    w = torch.zeros((Mat_Data.shape[1], Mat_Data.shape[2]), dtype=torch.float32, device=Mat_Data.device)

    for i in range(Mat_Data.shape[2]):
        # 构造 M 和 v
        M = torch.cat((Mat_Data_r[:, :, i], Mat_Data_i[:, :, i]), dim=0)
        v = torch.cat((Tag_Vec_r[:, i], Tag_Vec_i[:, i]), dim=0)
        w[:, i] = solve_linear_constrain(M, v, Itr)

    IndDel = int(np.floor((Ker_Size[0] - 1) / 2) * Ker_Size[1] + np.ceil(Ker_Size[1] / 2)) - 1

    # 初始化 Ker_w 和 Ker
    Ker_w = torch.zeros((w.shape[0] + 1, w.shape[1]), dtype=torch.float32, device=Mat_Data.device)
    Ker = torch.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=torch.float32, device=Mat_Data.device)
    Ker_Tra = torch.zeros((Ker_Size[0], Ker_Size[1], w.shape[1], w.shape[1]), dtype=torch.float32, device=Mat_Data.device)

    # 计算 Ker_Size 的积
    Ker_Size_prod = np.prod(Ker_Size)

    # 循环处理每一列
    for i in range(Ker_w.shape[1]):
        Target_loc = i * Ker_Size_prod + IndDel
        Ind_List = list(range(0, Target_loc)) + list(range(Target_loc + 1, Ker_Size_prod * Ker_w.shape[1]))

        # 更新 Ker_w
        Ker_w[Ind_List, i] = w[:, i]

        # 更新 Ker
        reshaped_Ker_w = Ker_w[:, i].reshape(Ker_w.shape[1], Ker_Size[0], Ker_Size[1])
        Ker[:, :, :, i] = reshaped_Ker_w.permute(2, 1, 0)
        RecFraOpe = transpose_filter(Ker[:, :, :, i])
        Ker_Tra[:, :, :, i] = RecFraOpe

    Ker_Tra = Ker_Tra.permute(0, 1, 3, 2).contiguous()
    KerW = torch.sum(Ker ** 2, dim=(0, 1, 2))
    Ker_TraW = torch.sum(Ker_Tra ** 2, dim=(0, 1, 2))

    Ker_W_Max = torch.max(KerW * Ker_TraW)

    if Ker_W_Max > 1:
        Ker_Con = (Ker_W_Max + 1e-8) ** 0.5
        Ker = Ker / Ker_Con
        Ker_Tra = Ker_Tra / Ker_Con

    return Ker, Ker_Tra

def transpose_filter(FilOpe):
    RecFraOpe = torch.zeros_like(FilOpe)
    _, _, DimFilOpe = FilOpe.shape
    for i in range(DimFilOpe):
        RecFraOpe[:, :, i] = torch.flip(FilOpe[:, :, i], dims=[0,1])

    return RecFraOpe

def solve_linear_constrain(A, b, Itr=500):
    # 初始化
    xk = torch.zeros((A.size(1), 1), dtype=A.dtype, device=A.device)
    y = torch.zeros((A.size(1), 1), dtype=A.dtype, device=A.device)
    AtA = torch.mm(A.T, A)
    Atb = torch.mm(A.T, b.reshape(-1, 1))
    tk = 1

    # 计算 Lipschitz 常数 L
    L = torch.max(torch.sum(torch.abs(AtA), dim=1))

    for _ in range(Itr):
        xk1 = y - (torch.mm(AtA, y) - Atb) / L
        x_Square = torch.norm(xk1, p=2)
        if x_Square > 0.99:
            xk1 = xk1 / (x_Square * 1.0102)

        tk1 = (1 + np.sqrt(1 + 4 * tk * tk)) / 2
        Acc_Wei = (tk - 1) / tk1
        tk = tk1

        y = xk1 + Acc_Wei * (xk1 - xk)
        xk = xk1

    return xk.flatten()

def get_ACSMask(mask):
    calib, AC_region = getCalibSize_1D_Edt(mask)
    ACS_line = np.abs(AC_region[1] - AC_region[0]) + 1
    Sense_Est_Mask = torch.zeros_like(mask)
    Sense_Est_Mask[AC_region[0]:AC_region[1] + 1, ...] = 1
    return Sense_Est_Mask, ACS_line

def get_sensitivity(ksfull, mask):
    Sense_Est_Mask, ACS_line = get_ACSMask(mask)
    kscenter = ksfull * Sense_Est_Mask
    sense = Sensitivity_Compute(kscenter)
    # ========================
    # import main
    # sense = Sensitivity_Compute(main.ksfull)

    return sense, sense, ACS_line

def get_ESPIRiT_sensitivity(ksfull, mask):
    # sense, sense, ACS_line = get_sensitivity(ksfull, mask)
    # ksdata = torch.zeros_like(ksfull)
    # for i in range(ksfull.shape[-1]):
    #     ksdata[:, :, i] = torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(ksfull[:, :, i])))
    # ksdata = ksdata.unsqueeze(0)
    # esp = np.squeeze(espirit(ksdata.cpu().numpy(), 4, min(ACS_line, 24), 0.01, 0.9925))
    # sensitivity = esp[:, :, :, 0]
    # sensitivity = torch.from_numpy(sensitivity).to(device=ksfull.device, dtype=ksfull.dtype)
    # return sensitivity, sense, ACS_line

    sense, sense, ACS_line = get_sensitivity(ksfull, mask)
    ksdata = torch.zeros_like(ksfull)
    for i in range(ksfull.shape[-1]):
        ksdata[:, :, i] = torch.fft.fft2(torch.fft.ifftshift(torch.fft.ifft2(ksfull[:, :, i])))

    mps = mr.app.EspiritCalib(
        ksdata.permute(2, 0, 1).cpu().numpy(),
        calib_width=min(ACS_line, 24),
        thresh=0.02,
        kernel_width=4,
        crop=0.9925,
        max_iter=100
    ).run()
    sensitivity = torch.from_numpy(mps).permute(1, 2, 0).to(device=ksfull.device, dtype=ksfull.dtype)
    return sensitivity, sense, ACS_line

def Sensitivity_Compute(kscenter):
    Img_zero_fill = IFFT2_3D_N(kscenter)
    Img_clip_sos = sos(Img_zero_fill) + torch.finfo(torch.float32).eps
    Sense = Img_zero_fill / Img_clip_sos.unsqueeze(-1)
    return Sense

def getCalibSize_1D_Edt(mask):
    mask = mask[:, :, 0]
    mask_row, mask_col = mask.shape
    mask_row_center = mask_row // 2
    mask_col_center = mask_col // 2
    sx = 1
    sy = 1
    row_pos = mask_row_center - 1
    while row_pos >= 0:
        if mask[row_pos, mask_col_center] == 0:
            break
        else:
            sx += 1
        row_pos -= 1
    ACS_row_up_edge = row_pos + 1
    row_pos = mask_row_center + 1
    while row_pos < mask_row:
        if mask[row_pos, mask_col_center] == 0:
            break
        else:
            sx += 1
        row_pos += 1
    ACS_row_down_edge = row_pos - 1
    col_pos = mask_col_center - 1
    while col_pos >= 0:
        if mask[mask_row_center, col_pos] == 0:
            break
        else:
            sy += 1
        col_pos -= 1
    col_pos = mask_col_center + 1
    while col_pos < mask_col:
        if mask[mask_row_center, col_pos] == 0:
            break
        else:
            sy += 1
        col_pos += 1
    calib_size = [sx, sy]
    ACS_edge = [ACS_row_up_edge, ACS_row_down_edge]
    return calib_size, ACS_edge

def NormalizedCoeByEnergy(Coe):
    EngCoe = np.abs(Coe) ** 2
    SumEng = np.sum(np.sum(EngCoe, axis=1), axis=0)
    MaxEng = Coe.shape[0] * Coe.shape[1]
    WeiEng = MaxEng / SumEng
    NorCoe = np.empty_like(Coe)
    NorCoe = Coe * np.sqrt(WeiEng)[np.newaxis, np.newaxis, :]
    return NorCoe

def sos(x, pnorm=2):
    if isinstance(x, np.ndarray):
        sum_norm = np.sum(np.abs(x) ** pnorm, axis=-1)
        res = np.power(sum_norm, 1 / pnorm)
    if isinstance(x, torch.Tensor):
        sum_norm = torch.sum(torch.abs(x) ** pnorm, dim=-1)
        res = torch.pow(sum_norm, 1 / pnorm)
    return res

def sos_T(x, sense_map):
    return sense_map * x.unsqueeze(-1)

def sos_complex(x, sense_map):
    sense_map_conj = sense_map.real - 1j * sense_map.imag
    return (x * sense_map_conj).sum(dim=-1)

def sos_complex_T(x, sense_map):
    x_cpy = x.unsqueeze(2).repeat(1, 1, sense_map.size(-1))
    re = x_cpy.real * sense_map.real - x_cpy.imag * sense_map.imag
    im = x_cpy.real * sense_map.imag + x_cpy.imag * sense_map.real
    return re + 1j * im

def FFT2_3D_N(Img, fcoe=0):
    if isinstance(Img, np.ndarray):
        fcoe = np.zeros_like(Img)
        for i in range(Img.shape[2]):
            fcoe[:, :, i] = np.fft.fftshift(np.fft.fft2(Img[:, :, i]))
    if isinstance(Img, torch.Tensor):
        fcoe = torch.zeros_like(Img)
        for i in range(Img.shape[2]):
            fcoe[:, :, i] = torch.fft.fftshift(torch.fft.fft2(Img[:, :, i]))

    fcoe /= np.sqrt(Img.shape[0] * Img.shape[1])

    return fcoe

def IFFT2_3D_N(Fcoe, Img=0):
    if isinstance(Fcoe, np.ndarray):
        Img = np.zeros_like(Fcoe, dtype=np.complex)
        for i in range(Fcoe.shape[2]):
            Img[:, :, i] = np.fft.ifft2(np.fft.ifftshift(Fcoe[:, :, i]))
    if isinstance(Fcoe, torch.Tensor):
        Img = torch.zeros_like(Fcoe, dtype=torch.complex64)
        for i in range(Fcoe.shape[2]):
            Img[:, :, i] = torch.fft.ifft2(torch.fft.ifftshift(Fcoe[:, :, i]))

    Img *= np.sqrt(Fcoe.shape[0] * Fcoe.shape[1])
    return Img

def double2uint8(im, save_path):
    ref_uint8 = np.abs(im)
    ref_uint8 = ref_uint8 / np.max(ref_uint8)
    ref_uint8 = img_as_ubyte(ref_uint8)
    return ref_uint8

def shearHaarDec(img, filterXY):
    ConstHaar = 5
    ConstShear = 13
    StoreLev = ConstHaar + ConstShear - 1
    H, W = img.shape
    TFcoef = torch.zeros(H, W, StoreLev, dtype=img.dtype, device=img.device)

    FOut = astf_dec2(img, filterXY)

    TFcoef[..., ConstHaar:ConstHaar + 12] = torch.cat([
        torch.stack(FOut['coefs'][0][0], dim=-1),
        torch.stack(FOut['coefs'][0][1], dim=-1)
    ], dim=-1)

    TFcoef[..., : ConstHaar] = DCT7Dec2(FOut['coefs'][1])

    return TFcoef

def multiShearHaarDec(multiCoilImg, filterXY):
    ConstHaar = 5
    ConstShear = 13
    StoreLev = ConstHaar + ConstShear - 1
    H, W, C = multiCoilImg.shape
    multi_coil_TFcoef = torch.zeros(H, W, C, StoreLev, dtype=multiCoilImg.dtype, device=multiCoilImg.device)
    for i in range(C):
        multi_coil_TFcoef[:, :, i, :] = shearHaarDec(multiCoilImg[:, :, i], filterXY)

    return multi_coil_TFcoef

def shearHaarRec(TFcoef, filterXY, filterHaar):
    ConstHaar = 5

    LowPass = DCT7Rec2(TFcoef[..., :ConstHaar], filterHaar)
    FOut = {'coefs': [None, None]}
    FOut['coefs'][1] = LowPass
    HighPass1 = [TFcoef[:, :, ConstHaar + i] for i in range(6)]
    HighPass2 = [TFcoef[:, :, ConstHaar + i + 6] for i in range(6)]

    FOut['coefs'][0] = [HighPass1, HighPass2]

    return astf_rec2(FOut, filterXY)

def multiShearHaarRec(multiCoilImg, filterXY, filterHaar):
    H, W, C, _ = multiCoilImg.shape
    multi_coil = torch.zeros(H, W, C, dtype=multiCoilImg.dtype, device=multiCoilImg.device)
    for i in range(C):
        multi_coil[:, :, i] = shearHaarRec(multiCoilImg[:, :, i, :], filterXY, filterHaar)
    return multi_coil

def DCT7Dec2(img, lev=1):
    img = img / 4
    img1_1 = torch.roll(img, shifts=lev, dims=0)
    img1_1 = torch.roll(img1_1, shifts=lev, dims=1)
    img0_1 = torch.roll(img, shifts=lev, dims=1)
    img1_0 = torch.roll(img, shifts=lev, dims=0)

    h, w = img.shape
    DecImg = torch.zeros((h, w, 5), dtype=img.dtype, device=img.device)
    DecImg[:, :, 0] = img + img1_1 + img0_1 + img1_0
    DecImg[:, :, 1] = img1_1 - img
    DecImg[:, :, 2] = img0_1 - img1_0
    DecImg[:, :, 3] = img1_1 - img1_0
    DecImg[:, :, 4] = img1_1 - img0_1
    return DecImg

def DCT7Rec2(DecImg, Filter):
    h, w, C = DecImg.shape
    kh, kw, C2 = Filter.shape
    scale = torch.tensor([1, 1, 1, 2, 2], dtype=Filter.dtype, device=Filter.device)
    Filter = Filter * scale.view(1, 1, -1)

    pad_h, pad_w = kh // 2, kw // 2
    Data_pad = torch.cat((DecImg[-pad_h:, :, :], DecImg, DecImg[:pad_h, :, :]), dim=0)
    temp = Data_pad[:, :pad_w, :]
    Data_pad = torch.cat((Data_pad[:, -pad_w:, :], Data_pad, temp), dim=1)

    Data_pad = Data_pad.permute(2,0,1).unsqueeze(0)
    Filter = Filter.permute(2,0,1).unsqueeze(1)
    if torch.is_complex(DecImg):
        real = F.conv2d(Data_pad.real.to(torch.float32), Filter.to(torch.float32), groups=C)
        imag = F.conv2d(Data_pad.imag.to(torch.float32), Filter.to(torch.float32), groups=C)
        out = torch.complex(real, imag)
    else:
        out = F.conv2d(Data_pad.to(torch.float32), Filter.to(torch.float32), groups=C)
    out = out.squeeze(0).permute(1,2,0)
    return out.sum(dim=-1)

def ImgDecFram(img, Frame):
    kh, kw, N = Frame.shape
    pad_h, pad_w = kh // 2, kw // 2
    img_pad = torch.cat((img[-pad_h:, :], img, img[:pad_h, :]), dim=0)
    temp = img_pad[:, :pad_w]
    img_pad = torch.cat((img_pad[:, -pad_w:], img_pad, temp), dim=1)

    img_pad = img_pad.unsqueeze(0).unsqueeze(0)
    Frame = Frame.permute(2,0,1).unsqueeze(1)
    if torch.is_complex(img):
        real = F.conv2d(img_pad.real.to(torch.float32), Frame.to(torch.float32))
        imag = F.conv2d(img_pad.imag.to(torch.float32), Frame.to(torch.float32))
        out = torch.complex(real, imag)
    else:
        out = F.conv2d(img_pad.to(torch.float32), Frame.to(torch.float32))
    return out.squeeze(0).permute(1,2,0)

def ImgRecFram(Data, RecFrame):
    h, w, N = Data.shape
    kh, kw, N2 = RecFrame.shape

    pad_h, pad_w = kh // 2, kw // 2
    Data_pad = torch.cat((Data[-pad_h:, :, :], Data, Data[:pad_h, :, :]), dim=0)
    temp = Data_pad[:, :pad_w, :]
    Data_pad = torch.cat((Data_pad[:, -pad_w:, :], Data_pad, temp), dim=1)
    Data_pad = Data_pad.permute(2,0,1).unsqueeze(0)
    RecFrame = RecFrame.permute(2,0,1).unsqueeze(1)
    if torch.is_complex(Data):
        real = F.conv2d(Data_pad.real.to(torch.float32), RecFrame.to(torch.float32), groups=N)
        imag = F.conv2d(Data_pad.imag.to(torch.float32), RecFrame.to(torch.float32), groups=N)
        out = torch.complex(real, imag)
    else:
        out = F.conv2d(Data_pad.to(torch.float32), RecFrame.to(torch.float32), groups=N)
    out = out.squeeze(0).permute(1,2,0)
    return out.sum(dim=-1)

def imfilter_symmetric(img, frame):
    # 确定 padding 以实现循环边界效果
    pad_h = frame.size(0) // 2
    pad_w = frame.size(1) // 2

    # 创建一个新的张量 img_pad，用于存放扩展后的结果
    temp = img[:pad_h, :]
    img_pad = torch.cat((torch.flip(temp, dims=[0]), img), dim=0)
    temp = img[-pad_h:, :]
    img_pad = torch.cat((img_pad, torch.flip(temp, dims=[0])), dim=0)
    temp = img_pad[:, :pad_w]
    img_pad = torch.cat((torch.flip(temp, dims=[1]), img_pad), dim=1)
    temp = img_pad[:, -pad_w:]
    img_pad = torch.cat((img_pad, torch.flip(temp, dims=[1])), dim=1)

    img = img_pad

    # 处理输入，确保 img 和 frame 是 PyTorch 张量
    img = img.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度
    frame = frame.unsqueeze(0).unsqueeze(0)  # 添加 batch 和 channel 维度


    # 使用 PyTorch 的卷积操作
    if torch.is_complex(img):
        conv_real = F.conv2d(img.real.to(dtype=torch.float32), frame.to(dtype=torch.float32), padding=0)
        conv_imag = F.conv2d(img.imag.to(dtype=torch.float32), frame.to(dtype=torch.float32), padding=0)
        filtered = torch.complex(conv_real, conv_imag)
    else:
        filtered = F.conv2d(img.to(dtype=torch.float32), frame.to(dtype=torch.float32), padding=0)

    return filtered.squeeze(0).squeeze(0)  # 移除 batch 和 channel 维度

def imfilter_symmetric_3D(img, frame):
    _, _, subs = img.shape
    Sigma = torch.zeros_like(img)
    for sub in range(subs):
        Sigma[:, :, sub] = imfilter_symmetric(img[:, :, sub], frame)
    return Sigma

def imfilter_symmetric_4D(img):
    frame = torch.ones((3, 3), dtype=torch.float64, device=img.device) / 9

    H, W, coils, subs = img.shape
    batch_size = coils * subs

    pad_h = frame.size(0) // 2
    pad_w = frame.size(1) // 2
    img_pad = torch.cat([torch.flip(img[:pad_h, :, :, :], [0]), img, torch.flip(img[-pad_h:, :, :, :], [0])], dim=0)
    img_pad = torch.cat([torch.flip(img_pad[:, :pad_w, :, :], [1]), img_pad, torch.flip(img_pad[:, -pad_w:, :, :], [1])], dim=1)

    img_pad = img_pad.permute(2, 3, 0, 1).reshape(batch_size, 1, H + 2*pad_h, W + 2*pad_w)
    frame = frame.unsqueeze(0).unsqueeze(0).to(img_pad.dtype).to(img_pad.device)
    if torch.is_complex(img_pad):
        conv_real = F.conv2d(img_pad.real.float(), frame.float(), padding=0)
        conv_imag = F.conv2d(img_pad.imag.float(), frame.float(), padding=0)
        filtered = torch.complex(conv_real, conv_imag)
    else:
        filtered = F.conv2d(img_pad.float(), frame, padding=0)
    filtered = filtered.reshape(coils, subs, H, W).permute(2, 3, 0, 1)

    return filtered

def wthresh(coef, thresh):
    if torch.is_complex(coef):
        res = coef * torch.maximum((torch.abs(coef) - thresh), torch.tensor(0.0)) / torch.abs(coef)
    else:
        res = torch.sgn(coef) * torch.maximum(torch.abs(coef) - thresh, torch.tensor(0.0))
    return res

def VidHaarDec3S(Vid, Lev=2):
    if Lev == 1:
        DecVid  = HaarDec3S(Vid, 1)
    elif Lev == 2:
        DecVid2 = HaarDec3S(Vid, 1)
        DecVid  = HaarDec3S(DecVid2[:,:,:, 0], 2)
        DecVid  = torch.cat((DecVid, DecVid2[:, :, :, 1:]), dim=3)
    else:
        raise ValueError('Lev is too big')

    return DecVid

def HaarDec3S(Vid, Lev=2):
    Vid0 = Vid.clone()  # Copy of Vid

    # Circular shifts using torch.roll
    shift_amount = 2 ** (Lev - 1)
    Vid1 = torch.roll(Vid0, shifts=[0, 0, shift_amount], dims=[0, 1, 2])  # 0 0 1
    Vid1m = torch.roll(Vid0, shifts=[0, 0, -shift_amount], dims=[0, 1, 2])  # 0 0 -1

    Vid2 = torch.roll(Vid0, shifts=[0, shift_amount, 0], dims=[0, 1, 2])  # 0 1 0

    Vid3 = torch.roll(Vid0, shifts=[0, shift_amount, shift_amount], dims=[0, 1, 2])  # 0 1 1
    Vid3m = torch.roll(Vid0, shifts=[0, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 0 -1 -1

    Vid4 = torch.roll(Vid0, shifts=[shift_amount, 0, 0], dims=[0, 1, 2])  # 1 0 0

    Vid5 = torch.roll(Vid0, shifts=[shift_amount, 0, shift_amount], dims=[0, 1, 2])  # 1 0 1
    Vid5m = torch.roll(Vid0, shifts=[-shift_amount, 0, -shift_amount], dims=[0, 1, 2])  # -1 0 -1

    Vid6 = torch.roll(Vid0, shifts=[shift_amount, shift_amount, 0], dims=[0, 1, 2])  # 1 1 0

    Vid7 = torch.roll(Vid0, shifts=[shift_amount, shift_amount, shift_amount], dims=[0, 1, 2])  # 1 1 1
    Vid7m = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, -shift_amount], dims=[0, 1, 2])  # -1 -1 -1

    Vidp1p1m1 = torch.roll(Vid0, shifts=[shift_amount, shift_amount, -shift_amount], dims=[0, 1, 2])  # 1 1 -1
    Vidm1m1p1 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, shift_amount], dims=[0, 1, 2])  # -1 -1 1

    Vidp1m1p1 = torch.roll(Vid0, shifts=[shift_amount, -shift_amount, shift_amount], dims=[0, 1, 2])  # 1 -1 1
    Vidm1p1m1 = torch.roll(Vid0, shifts=[-shift_amount, shift_amount, -shift_amount], dims=[0, 1, 2])  # -1 1 -1

    Vidm1p1p1 = torch.roll(Vid0, shifts=[-shift_amount, shift_amount, shift_amount], dims=[0, 1, 2])  # -1 1 1
    Vidp1m1m1 = torch.roll(Vid0, shifts=[shift_amount, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 1 -1 -1

    Vidp10m1 = torch.roll(Vid0, shifts=[shift_amount, 0, -shift_amount], dims=[0, 1, 2])  # 1 0 -1
    Vidm10p1 = torch.roll(Vid0, shifts=[-shift_amount, 0, shift_amount], dims=[0, 1, 2])  # -1 0 1

    Vid0p1m1 = torch.roll(Vid0, shifts=[0, shift_amount, -shift_amount], dims=[0, 1, 2])  # 0 1 -1
    Vid0m1p1 = torch.roll(Vid0, shifts=[0, -shift_amount, shift_amount], dims=[0, 1, 2])  # 0 -1 1


    DecVid = torch.zeros([Vid.size(0), Vid.size(1), Vid.size(2), 6], dtype=Vid.dtype, device=Vid.device)


    # Compute decomposition
    DecVid[:, :, :, 0] = (Vid0 + Vid1 + Vid2 + Vid3 + Vid4 + Vid5 + Vid6 + Vid7) / 8  # low-pass filter

    DecVid[:, :, :, 1] = (Vid4 - Vid0) / 4  # (1,0,0) - (0,0,0) x-axis
    DecVid[:, :, :, 2] = (Vid2 - Vid0) / 4  # (0,1,0) - (0,0,0) y-axis

    DecVid[:, :, :, 3] = np.sqrt(2) / 8 * (Vid6 - Vid0)  # (1,1,0) - (0,0,0) xy-plane 1
    DecVid[:, :, :, 4] = np.sqrt(2) / 8 * (Vid4 - Vid2)  # (1,0,0) - (0,1,0) xy-plane 2

    DecVid[:, :, :, 5] = (1 / 2 * Vid0
                          - 1 / 16 * (Vid1 + Vid1m)
                          - 1 / 32 * (Vid3 + Vid3m)
                          - 1 / 32 * (Vid5 + Vid5m)
                          - 1 / 64 * (Vid7 + Vid7m)
                          - 1 / 64 * (Vidp1p1m1 + Vidm1m1p1)
                          - 1 / 64 * (Vidp1m1p1 + Vidm1p1m1)
                          - 1 / 64 * (Vidm1p1p1 + Vidp1m1m1)
                          - 1 / 32 * (Vidp10m1 + Vidm10p1)
                          - 1 / 32 * (Vid0p1m1 + Vid0m1p1))  # auxiliary filter

    return DecVid

def VidHaarRec3S(input, Lev=2):
    DecVid = input.clone()
    if Lev == 1:
        Vid                 = HaarRec3S(DecVid, 1)
    elif Lev == 2:
        DecVid[:, :, :, 5]  = HaarRec3S(DecVid[:, :, :, :6], 2)
        Vid                 = HaarRec3S(DecVid[:, :, :, 5:], 1)
    else:
        raise ValueError('Lev is too big')

    return Vid

def HaarRec3S(DecVid, Lev):
    Vid0 = DecVid[:, :, :, 0]

    # Perform circular shifts using torch.roll
    shift_amount = 2 ** (Lev - 1)
    Vid1 = torch.roll(Vid0, shifts=[0, 0, -shift_amount], dims=[0, 1, 2])  # 0 0 1
    Vid2 = torch.roll(Vid0, shifts=[0, -shift_amount, 0], dims=[0, 1, 2])  # 0 1 0
    Vid3 = torch.roll(Vid0, shifts=[0, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 0 1 1
    Vid4 = torch.roll(Vid0, shifts=[-shift_amount, 0, 0], dims=[0, 1, 2])  # 1 0 0
    Vid5 = torch.roll(Vid0, shifts=[-shift_amount, 0, -shift_amount], dims=[0, 1, 2])  # 1 0 1
    Vid6 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, 0], dims=[0, 1, 2])  # 1 1 0
    Vid7 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, -shift_amount], dims=[0, 1, 2])  # 1 1 1

    # Initialize Vid
    Vid = (Vid0 + Vid1 + Vid2 + Vid3 + Vid4 + Vid5 + Vid6 + Vid7) / 8

    # Perform the reconstruction based on different filters

    # (1,0,0)-(0,0,0) x-axis
    Vid0 = DecVid[:, :, :, 1]
    Vid0 = torch.roll(Vid0, shifts=[-shift_amount, 0, 0], dims=[0, 1, 2]) - Vid0
    Vid += 1 / 4 * Vid0

    # (0,1,0)-(0,0,0) y-axis
    Vid0 = DecVid[:, :, :, 2]
    Vid0 = torch.roll(Vid0, shifts=[0, -shift_amount, 0], dims=[0, 1, 2]) - Vid0
    Vid += 1 / 4 * Vid0

    # (1,1,0)-(0,0,0) xy-plane 1
    Vid0 = DecVid[:, :, :, 3]
    Vid0 = torch.roll(Vid0, shifts=[-shift_amount, -shift_amount, 0], dims=[0, 1, 2]) - Vid0
    Vid += np.sqrt(2) / 8 * Vid0

    # (1,0,0)-(0,1,0) xy-plane 2
    Vid0 = DecVid[:, :, :, 4]
    Vid0 = torch.roll(Vid0, shifts=[-shift_amount, 0, 0], dims=[0, 1, 2]) - torch.roll(Vid0, shifts=[0, -shift_amount, 0], dims=[0, 1, 2])
    Vid += np.sqrt(2) / 8 * Vid0

    # Auxiliary filter
    Vid += DecVid[:, :, :, 5]

    return Vid

def Kernel_Rec_ks_C_I_Pro(ks_data, Ker, Pro=1):
    ks_Rec = torch.zeros_like(ks_data)
    for coi in range(ks_data.size(2)):
        ks_Rec[:, :, coi] = ImgRecFram(ks_data, Ker[:, :, :, coi])

    return Pro * (ks_Rec - ks_data)

def Kernel_Rec_ks_C_Pro(ks_data, Ker, Pro=1):
    ks_Rec = torch.zeros_like(ks_data)
    for coi in range(ks_data.size(2)):
        ks_Rec[:, :, coi] = Pro * ImgRecFram(ks_data, Ker[:, :, :, coi])

    return ks_Rec

def EnergyScaling(Thr, cof):
    Thr_min = abs(Thr).min()
    Thr_max = abs(Thr).max()
    Coe_min = abs(cof).min()
    Coe_max = abs(cof).max()
    normalized_Thr = Coe_min + (Coe_max - Coe_min) * 1.0 * (Thr - Thr_min) / (Thr_max - Thr_min)
    return normalized_Thr

def EnergyScaling_3D(Thr, cof):
    n1, n2, n3 = Thr.shape
    min_Thr = Thr.view(-1, n3).max(dim=0, keepdim=True).values.view(1, 1, n3)
    maxCC = cof.abs().view(-1, n3).max(dim=0, keepdim=True).values.view(1, 1, n3)
    normalized_Thr = Thr / min_Thr * maxCC

    return normalized_Thr

def EnergyScaling_4D(Thr, cof):
    _, _, _, n4 = Thr.shape
    min_Thr = Thr.contiguous().view(-1, n4).max(dim=0, keepdim=True).values.view(1, 1, 1, n4)
    maxCC = cof.abs().view(-1, n4).max(dim=0, keepdim=True).values.view(1, 1, 1, n4)
    normalized_Thr = Thr / min_Thr * maxCC
    return normalized_Thr

def normalize_image(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # 加上小量防止除以0
    return img

def PSNR_SSIM_HaarPSI(ref, deg, model_name, save_path, utime=-1):
    image = Image.fromarray(img_as_ubyte(((abs(deg) / torch.max(abs(deg))).to(torch.float32)).cpu().numpy()))
    image.save(f'{save_path}/1-{model_name}.png')

    MAINBODY = FINDMAINBODY(ref)
    ref_pt = (ref / torch.max(ref)).to(torch.float32)
    deg_pt = (deg / torch.max(deg)).to(torch.float32)
    ref_np = normalize_image(crop_center((ref_pt * MAINBODY).cpu().numpy(), MAINBODY))
    deg_np = normalize_image(crop_center((deg_pt * MAINBODY).cpu().numpy(), MAINBODY))
    ref_pt_c = torch.from_numpy(ref_np).to(ref.device).to(torch.float32)
    deg_pt_c = torch.from_numpy(deg_np).to(deg.device).to(torch.float32)

    psnr = compare_psnr(ref_np, deg_np, data_range=1.)
    ssim = compare_ssim(ref_np, deg_np, multichannel=False, data_range=1.)
    haarpsi = HaarPSI(ref_pt_c, deg_pt_c)
    dists = compute_DISTS(ref_pt_c, deg_pt_c)
    lpips_ = compute_LPIPS(ref_pt_c, deg_pt_c)
    hfen = compute_HFEN(ref_np, deg_np)
    print(f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, DISTS: {dists.item():.4f}, HaarPSI: {haarpsi[0].item():.4f}, LPIPS: {lpips_:.4f}, HFEN: {hfen:.4f}\n")


    ref_crop = normalize_image(crop_center(ref_pt.cpu().numpy(), MAINBODY))
    Image.fromarray(img_as_ubyte(ref_crop)).save(f'{save_path}/2-reference-crop.png')
    deg_crop = normalize_image(crop_center(deg_pt.cpu().numpy(), MAINBODY))
    Image.fromarray(img_as_ubyte(deg_crop)).save(f'{save_path}/2-{model_name}-crop.png')
    # error_color_picture(np.abs(ref_crop - deg_crop), f'{save_path}/3-{model_name}-error.png')

    # image = image.convert("RGB")
    # draw = ImageDraw.Draw(image)
    # draw.text((2, 2), f"PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, HaarPSI: {haarpsi[0].item():.4f}, DISTS: {dists.item():.4f}", fill=(255, 0, 0))
    # image.save(f'{save_path}/Index-{model_name}.png')
    return [model_name, psnr, ssim, dists.item(), haarpsi[0].item(), lpips_, hfen, utime]

def compute_HaarPSI(ref, deg):
    ref_ = ref / torch.max(ref)
    deg_ = deg / torch.max(deg)
    haarpsi = HaarPSI(ref_.to(torch.float32), deg_.to(torch.float32))
    return haarpsi[0].item()

def compute_LPIPS(ref, deg):
    sys_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    func = lpips.LPIPS(net='alex').cuda()
    sys.stdout = sys_stdout
    ref = (ref * 2 - 1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    deg = (deg * 2 - 1).unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)

    device = ref.device
    func = func.to(device)
    deg = deg.to(device)
    ref = ref.to(device)
    return func(ref, deg).item()

def compute_HFEN(ref_img: np.ndarray, test_img: np.ndarray, sigma: float = 1.5) -> float:
    # 确保输入为浮点数
    ref_img = ref_img.astype(np.float32)
    test_img = test_img.astype(np.float32)

    # LoG 滤波（高斯-拉普拉斯）
    log_ref = gaussian_laplace(ref_img, sigma=sigma)
    log_test = gaussian_laplace(test_img, sigma=sigma)

    # 计算均方根误差 (RMSE)
    diff = log_ref - log_test
    hfen_value = np.sqrt(np.mean(diff ** 2))

    return hfen_value

def crop_center(array, MAINBODY):
    non_zero_indices = np.nonzero(MAINBODY)
    min_row, max_row = non_zero_indices[:,0].min(), non_zero_indices[:,0].max()
    min_col, max_col = non_zero_indices[:,1].min(), non_zero_indices[:,1].max()
    res = array[min_row:max_row + 1, min_col:max_col + 1]
    res = res / np.max(res)

    return res

def error_color_picture(error, save_file, level=-1):
    if level == -1:
        plt.imshow(error, cmap='jet', interpolation='nearest')
    else:
        plt.imshow(error, cmap='jet', interpolation='nearest', vmax=level)
    plt.axis('off')  # 关闭坐标轴
    plt.savefig(save_file, bbox_inches='tight', pad_inches=0)
    plt.close()  # 关闭绘图窗口

def tensor_split(tensor, dim=1, each_channel=2):
    tensor_splitT = torch.split(tensor.unsqueeze(-1), each_channel, dim=dim)
    return torch.cat(tensor_splitT, dim=-1).mean(-1)

def i2r(tensor):
    tensor = torch.stack([tensor.real, tensor.imag], dim=-1)
    return tensor

def r2i(tensor):
    tensor = tensor[..., 0] + 1j * tensor[..., 1]
    return tensor

def rss_complex(tensor):
    return torch.sqrt((abs(tensor) ** 2).sum(dim=-1))

def VarGuaEstimation(Data):
    Row, Col = Data.shape
    Template = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    Template = torch.tensor(Template, dtype=torch.float32, device=Data.device)
    DataNoi = imfilter_symmetric(Data, Template)
    ModRate = 1
    Var_Abs = (ModRate * np.sqrt(np.pi / 2) * torch.sum(torch.abs(DataNoi))) / (6 * (Row - 2) * (Col - 2))
    Var_Squ = torch.sum(torch.abs(DataNoi ** 2)) / (36 * (Row - 2) * (Col - 2))
    return torch.sqrt(Var_Squ * ModRate)

def VarGuaEstimation_Margin(Data):
    Row, Col = Data.shape
    row_ = int(np.floor(Row * 0.3))
    col_ = int(np.floor(Col * 0.3))
    Template = torch.tensor([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=torch.float32, device=Data.device).unsqueeze(0).unsqueeze(0)

    # 定义四个角落的索引范围
    corners = [(slice(None, row_), slice(None, col_)),
               (slice(None, row_), slice(-col_, None)),
               (slice(-row_, None), slice(None, col_)),
               (slice(-row_, None), slice(-col_, None))]

    # 计算四个角落的卷积并求平方和
    total_noise = sum((F.conv2d(abs(Data[c[0], c[1]]).unsqueeze(0).unsqueeze(0), Template, padding=0).squeeze() ** 2) for c in corners)

    Var_Squ = torch.sum(total_noise) / (36 * 4 * (row_ - 2) * (col_ - 2))
    return torch.sqrt(Var_Squ)

def SubbandSTD_NoiVar_DoubleFrame(Coef, STD_NoiVar, T_FilOpe1, T_FilOpe2):
    STD_NoiVar = STD_NoiVar * STD_NoiVar

    Wei1 = torch.sum(T_FilOpe1 ** 2, dim=(0, 1)).view(T_FilOpe1.size(2), 1)
    Wei2 = torch.sum(T_FilOpe2 ** 2, dim=(0, 1)).view(T_FilOpe2.size(2), 1)

    Const1 = T_FilOpe1.size(-1)  # The number of framelet subband
    Const2 = T_FilOpe2.size(-1)  # The number of framelet subband
    StoreLev = Const1 + Const2 - 1  # store in DecImg(:, :, 0 ~ StoreLev)

    ThreSigma = torch.zeros(StoreLev, dtype=torch.float32, device=Coef.device)
    ThreSigma[StoreLev - Const1 : StoreLev] = Wei1.squeeze() * STD_NoiVar
    StoreLev -= Const1
    STD_NoiVar *= Wei1[0, 0]  # 更新 STD_NoiVar
    ThreSigma[StoreLev - Const2 + 1 : StoreLev + 1] = Wei2.squeeze() * STD_NoiVar

    # 对 ThreSigma 进行平方根变换
    ThreSigma = torch.sqrt(ThreSigma)

    return ThreSigma

def PixelLocalVariance_Lev_DoubleFrame(Coe, T_FilOpe1, T_FilOpe2):
    KerSize1 = T_FilOpe1.size(0)
    KerSize2 = T_FilOpe2.size(0)

    Const1 = T_FilOpe1.size(-1) - 1
    Const2 = T_FilOpe2.size(-1) - 1
    StoreLev = Const1 + Const2

    VarLoc = torch.zeros_like(Coe)

    AveKer1 = torch.ones((KerSize1, KerSize1), dtype=torch.float32, device=Coe.device)
    AveKerNum = torch.sum(AveKer1)
    Const_Ave = np.sqrt(2) / AveKerNum
    AveKer1 *= Const_Ave

    for i in range(StoreLev - Const1, StoreLev):
        VarLoc[:,:,i] = imfilter_symmetric(torch.abs(Coe[:,:, i]), AveKer1)
    StoreLev = StoreLev - Const1

    AveKer2 = torch.ones((KerSize2, KerSize2), dtype=torch.float32, device=Coe.device)
    AveKerNum = torch.sum(AveKer2)
    Const_Ave = np.sqrt(2) / AveKerNum
    AveKer2 *= Const_Ave

    for i in range(StoreLev - Const2, StoreLev):
        VarLoc[:,:,i] = imfilter_symmetric(torch.abs(Coe[:,:, i]), AveKer2)

    return abs(VarLoc)

def ImgDoubleFrameDec2_DCT7(Img, TNTF):
    Const1 = TNTF.T_FilOpe1.size(-1)
    Const2 = TNTF.T_FilOpe2.size(-1)
    StoreLev = Const1 + Const2 - 1

    DecImg = torch.zeros(size=(Img.size(0), Img.size(1), StoreLev), dtype=Img.dtype, device=Img.device)
    DecImg[:, :, StoreLev - Const1:StoreLev] = DCT7Dec2(Img, 1)
    StoreLev = StoreLev - Const1

    DecImg[:, :, StoreLev - Const2 + 1 : StoreLev + 1] = ImgDecFram(DecImg[:, :, StoreLev], TNTF.T_FilOpe2)
    return DecImg

def ImgDoubleFrameRec2_DCT7(input, TNTF):
    DecImg = input.clone()
    Const2 = TNTF.T_FilOpe2.size(-1)
    StoreLev = 0

    DecImg[:, :, StoreLev + Const2 - 1] = ImgRecFram(DecImg[:, :, StoreLev : StoreLev + Const2], TNTF.T_TraFilOpe2)
    StoreLev = StoreLev + Const2
    RecImg = DCT7Rec2(DecImg[:, :, StoreLev - 1 :], TNTF.T_TraFilOpe1)

    return RecImg

def DenoiseByDiffusion(DualDomain, x, step=1):
    FLAG = x.ndim == 2
    x = i2r(x).unsqueeze(0).permute(0, 3, 1, 2) if FLAG else i2r(x).permute(2, 3, 0, 1)
    x = x.contiguous()
    with torch.no_grad():
        for _ in range(step):
            sigma = torch.tensor(DualDomain.lam).to(x.device).view(1, 1, 1, 1)
            noise = torch.randn_like(x[0, ...]).unsqueeze(0) * sigma
            x_max = x.pow(2).sum(1).sqrt().max()
            inputs = (x / x_max + noise).to(torch.float32)
            logP = torch.zeros_like(inputs)
            for coil in range(inputs.shape[0]):
                logP[coil, ...] = DualDomain.scoreNet(inputs[coil, ...].unsqueeze(0), sigma)
            res = x + noise + sigma * logP
            # res = x + noise + sigma * (logP * sigma ** 2)
            DualDomain.lam = DualDomain.lam / DualDomain.gamma
    return r2i(res.squeeze().permute(1, 2, 0)) if FLAG else r2i(res.permute(2, 3, 0, 1))

def DenoiseByDiffusion_HighPass(DualDomain, input, step=1):
    x = input.permute(2, 3, 0, 1)
    x = x.contiguous()
    with torch.no_grad():
        for _ in range(step):
            sigma = math.sqrt(DualDomain.lam / DualDomain.rho)
            step_lr = DualDomain.delta
            sigma = torch.tensor(sigma).to(x.device).view(1, 1, 1, 1)
            noise = torch.randn_like(x) * sigma
            x_max = x.pow(2).sum(1).sqrt().max()
            inputs = (x / x_max + noise).to(torch.float32)
            logP = torch.zeros_like(inputs)
            for coil in range(inputs.shape[0]):
                logP[coil, :, :, :] = DualDomain.scoreNet_high(inputs[coil, :, :, :].unsqueeze(0), sigma)
            res = x + (step_lr * sigma ** 2) * logP
            DualDomain.rho = DualDomain.gamma * DualDomain.rho
    return res.permute(2, 3, 0, 1)

def SolverForXProblem(AFunction, target, x):
    target += x
    r = target - AFunction(x)
    p = r.clone()
    rTr = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
    for iter in range(5):
        Ap = AFunction(p)
        a = rTr / torch.abs(torch.dot(p.view(-1).conj(), Ap.view(-1)))
        x += a * p
        r -= a * Ap
        rTrk = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
        b = rTrk / rTr
        p = r + b * p
        rTr = rTrk
    return x

def SolverForSubProblem(AFunction, target, x_in, iter=5):
    x = x_in.clone()
    r = target - AFunction(x)
    p = r.clone()
    rTr = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
    for iter in range(iter):
        Ap = AFunction(p)
        a = rTr / torch.abs(torch.dot(p.view(-1).conj(), Ap.view(-1)))
        x += a * p
        r -= a * Ap
        rTrk = torch.abs(torch.dot(r.view(-1).conj(), r.view(-1)))
        b = rTrk / rTr
        p = r + b * p
        rTr = rTrk
        # print(f"At iteration {iter + 1}, err = {torch.norm(torch.abs(rTr)):.4f}")
    return x

def ESPIRiT_sensemap_estimate(ksfull, mask):
    X = np.zeros_like(ksfull, dtype=np.complex)
    for i in range(ksfull.shape[-1]):
        X[:, :, i] = np.fft.fft2(np.fft.fftshift(np.fft.ifft2(ksfull[:, :, i])))
    X = X[np.newaxis, ...]
    calib, AC_region = getCalibSize_1D_Edt(mask)
    ACS_line = np.abs(AC_region[1] - AC_region[0]) + 1
    esp = np.squeeze(espirit(X, 6, ACS_line, 0.02, 0.99))
    esp = esp[..., 0]
    for idx in range(ksfull.shape[-1]):
        plt.subplot(1, ksfull.shape[-1], idx+1)
        plt.imshow(np.abs(esp[:, :, idx]), cmap='gray')
        plt.axis('off')
    plt.show()
    return esp

def imageShow(image):
    temp = abs(image) / abs(image).max()
    plt.imshow(temp, cmap='gray')
    plt.axis('off')
    plt.show()

def FINDMAINBODY(image, alpha=1.5, BlackPoint=15):
    AveKer = np.ones((3, 3))
    if torch.is_tensor(image):
        res = abs(image).cpu().numpy()
    else:
        res = image
    res = convolve(res, AveKer, mode='nearest')
    res = np.power(res / res.max(), alpha)
    res[res <= BlackPoint / 255] = BlackPoint / 255
    res = np.sqrt((res - res.min()) / (res.max() - res.min()))
    res[res > 1e-6] = 1
    res[res != 1] = 0
    res = filling(res)
    M_inverted = 1 - res
    labeled_array, num_features = label(M_inverted)
    area_threshold = 90000
    BW_no_small_objects = np.copy(M_inverted)
    for region_label in range(1, num_features + 1):
        region_area = np.sum(labeled_array == region_label)
        if region_area < area_threshold:
            BW_no_small_objects[labeled_array == region_label] = 0
    res = 1 - BW_no_small_objects
    if torch.is_tensor(image):
        return torch.from_numpy(res).to(image.device)
    else:
        return res

def filling(I, windowSize=5, thre=0.95):
    h, w = I.shape
    Hmin = h
    Hmax = 0
    halfWindow = windowSize // 2
    for i in range(w):
        index = np.where(I[:, i] == 1)[0]  # 获取当前列中为1的索引
        if index.size == 0:
            continue
        for j in index:
            submatrix = I[max(0, j - halfWindow):min(h, j + halfWindow + 1),
                          max(0, i - halfWindow):min(w, i + halfWindow + 1)]
            if np.mean(submatrix) >= thre:
                Hmin = min(Hmin, j)
                break
        for j in reversed(index):
            submatrix = I[max(0, j - halfWindow):min(h, j + halfWindow + 1),
                          max(0, i - halfWindow):min(w, i + halfWindow + 1)]
            if np.mean(submatrix) >= thre:
                Hmax = max(Hmax, j)
                break
    se = np.ones((10, 10), dtype=int)
    I[Hmin:Hmax + 1, :] = binary_dilation(I[Hmin:Hmax + 1, :], structure=se)
    return I

def Multi_DCT(COM, input):
    h, w, c = input.shape
    res = torch.zeros(h, w, c, COM.T_FilOpe2.size(-1), dtype=input.dtype, device=input.device)
    for i in range(c):
        res[:, :, i, :] = ImgDecFram(input[:, :, i], COM.T_FilOpe2)
    return res

def Multi_DCT_T(COM, input):
    h, w, c, n = input.shape
    res = torch.zeros(h, w, c, dtype=input.dtype, device=input.device)
    for i in range(c):
        res[:, :, i] = ImgRecFram(input[:, :, i, :], COM.T_TraFilOpe2)
    return res

def Multi_TNTF(COM, input):
    Const1 = COM.T_FilOpe1.size(-1)
    Const2 = COM.T_FilOpe2.size(-1)
    StoreLev = Const1 + Const2 - 1
    H, W, C = input.shape
    multi_coil_TFcoef = torch.zeros(H, W, C, StoreLev, dtype=input.dtype, device=input.device)
    for i in range(C):
        multi_coil_TFcoef[:, :, i, :] = ImgDoubleFrameDec2_DCT7(input[:, :, i], COM)
    return multi_coil_TFcoef

def Multi_TNTF_T(COM, input):
    H, W, C, N = input.shape
    multi_coil = torch.zeros(H, W, C, dtype=input.dtype, device=input.device)
    for i in range(C):
        multi_coil[:, :, i] = ImgDoubleFrameRec2_DCT7(input[:, :, i, :], COM)
    return multi_coil

def getH(x):
    h, w, c = x.shape
    res = torch.zeros(h, w, c, c, dtype=x.dtype, device=x.device)
    for i in range(c):
        tmp = torch.zeros_like(x)
        for j in range(c):
            if j != i:
                tmp[..., i] += x[..., j]
                tmp[..., j] = -x[..., i]
        res[..., i] = tmp
    return res

def getHT(x):
    c = x.shape[-1]
    res = torch.zeros_like(x)
    for i in range(c):
        for j in range(c):
            res[..., i, j] = torch.conj(x[..., j, i])
    return res

def SSK(Kernel, coilImage):
    ks_Rec = torch.zeros_like(coilImage)
    for coi in range(coilImage.size(2)):
        ks_Rec[..., coi] = torch.sum(coilImage * Kernel[..., coi], dim=-1)
    return ks_Rec

def getHTH(H):
    h, w, c,_ = H.shape
    res = torch.zeros(h, w, c, c, dtype=H.dtype, device=H.device)
    for i in range(c):
        for j in range(c):
            tmp = torch.zeros(h, w, dtype=H.dtype, device=H.device)
            for k in range(c):
                tmp += torch.conj(H[..., i, k]) * H[..., j, k]
            res[..., j, i] = tmp
    return res

def HTHx(x, HTH):
    res = torch.zeros_like(x)
    for i in range(x.shape[-1]):
        res[..., i] = torch.sum(HTH[..., i] * x, dim=-1)
    return res

def calculate_and_plot_errors(save_path):
    # 获取所有以"2-"开头的PNG文件
    all_files = [f for f in os.listdir(save_path)
                 if f.startswith('2-') and f.endswith('.png')]

    if not all_files:
        print(f"在路径 {save_path} 下未找到以'2-'开头的PNG文件")
        return

    # 查找参考图像
    ref_image_name = '2-reference-crop.png'
    if ref_image_name not in all_files:
        print(f"在路径 {save_path} 下未找到参考图像 {ref_image_name}")
        return

    # 读取参考图像
    ref_image_path = os.path.join(save_path, ref_image_name)
    ref_image = normalize_image(np.array(Image.open(ref_image_path).convert('L')))  # 转换为灰度

    # 处理其他图像
    other_images = [f for f in all_files if f != ref_image_name]
    if not other_images:
        print("没有其他图像可与参考图像进行比较")
        return

    # 计算所有误差图的最大值A
    max_A = 0
    error_images = []

    for img_name in other_images:
        img_path = os.path.join(save_path, img_name)
        img = normalize_image(np.array(Image.open(img_path).convert('L')))
        error = np.abs(img - ref_image)
        error_images.append(error)
        current_max = np.max(error)
        if current_max > max_A:
            max_A = current_max

    # 绘制并保存误差图
    for img_name, error in zip(other_images, error_images):
        output_name = img_name.replace('2-', '3-').replace('-crop.png', '-error.png')
        output_path = os.path.join(save_path, output_name)
        error_color_picture(error, output_path, max_A)

    print("所有误差图计算完成")

