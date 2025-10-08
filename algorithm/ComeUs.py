import numpy as np
import torch

from utils.utils import *

class ComeUs:
    def __init__(self, max_iter=50):

        print('=====================================')
        print('============== ComeUs ===============')
        print('=====================================')
        import main
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.max_iter, self.save_path = max_iter, main.save_path

        self.T_mask, self.T_ksdata, self.T_ksfull = main.mask, main.ksdata, main.ksfull
        self.T_Ker, self.T_Ker_Tra, self.T_sensitivity = main.Ker, main.Ker_Tra, main.sensitivityLi

        tightFrame = loadmat('./data/TNTF.mat')
        self.T_FilOpe1 = torch.from_numpy(tightFrame['FilOpe1']).to(device)
        self.T_FilOpe2 = torch.from_numpy(tightFrame['FilOpe2']).to(device)
        self.T_TraFilOpe1 = torch.from_numpy(tightFrame['TraFilOpe1']).to(device)
        self.T_TraFilOpe2 = torch.from_numpy(tightFrame['TraFilOpe2']).to(device)

        self.FS = lambda x: FFT2_3D_N(self.T_sensitivity * x.unsqueeze(-1))
        self.FST = lambda x: torch.sum(torch.conj(self.T_sensitivity) * IFFT2_3D_N(x), dim=-1)

        self.B = lambda x: Multi_TNTF(self, IFFT2_3D_N((1 - self.T_mask) * self.FS(x)))
        self.BT = lambda x: self.FST((1 - self.T_mask) * FFT2_3D_N(Multi_TNTF_T(self, x)))

        self.Lip = (1 + (1 + main.Lip_C) ** 2)
        self.rho = 1.999 / self.Lip
        self.delta = 0.999 / self.rho

        self.start_time, self.target_S = time.time() - main.t_kernel - main.t_lip - main.t_sense, None
        self.ref, self.uk = sos(IFFT2_3D_N(main.ksfull)), sos(IFFT2_3D_N(self.T_ksdata))
        self.filterXY = astf_filtZ2(self.ref, astf_loadParams())

    def process(self, Thr=0, ukA=0, USn=0):
        ck = Multi_TNTF(self, IFFT2_3D_N(self.T_ksdata))
        qk = torch.zeros_like(ck)

        for iter in range(self.max_iter):
            if (iter + 1) % int(self.max_iter / 5) == 0 or iter == 0:
                print(f"At iteration {iter + 1}, err = {torch.norm(torch.abs(self.ref - self.uk)):.4f}")

            # iterative
            gradfF = self.getGradientF(self.uk)
            qk = qk - self.delta * self.B(self.rho * self.BT(qk) - (self.uk - self.rho * gradfF)) + self.delta * ck

            # set regularization parameter
            if iter in range(0, self.max_iter, 5):
                Thr = abs(Multi_TNTF(self, sos_T(self.uk, self.T_sensitivity)))
                Thr = imfilter_symmetric_4D(Thr[..., 1:])
                Thr = EnergyScaling_4D(1 / Thr, qk[..., 1:])

            qk[..., 0], qk[..., 1:] = 0, qk[..., 1:] - wthresh(qk[..., 1:], Thr)

            self.uk = abs(self.uk - self.rho * gradfF - self.rho * self.BT(qk))

            if USn < 3 and torch.mean(torch.abs(self.uk - ukA)) < 1e-3:
                print(f"At iteration {iter + 1}, the mean absolute error is {torch.mean(torch.abs(self.uk - ukA)):.6f}, update sensitivity information.")
                # self.uk = self.UpdateSensitivityByCG()
                self.uk = self.UpdateSensitivityInMultiMethod()
                USn += 1

            if iter == self.max_iter - 1 and USn == 0:
                # PSNR_SSIM_HaarPSI(self.ref, abs(self.uk), 'ComeUs', self.save_path)
                self.max_iter += 20
                print(f"At iteration {iter + 1}, the mean absolute error is {torch.mean(torch.abs(self.uk - ukA)):.6f}, update sensitivity information.")
                # self.uk = self.UpdateSensitivityByCG()
                self.uk = self.UpdateSensitivityInMultiMethod()
                USn += 1

            ukA = self.uk

        useTime = (time.time() - self.start_time)
        print(f"ComeUs Elapsed Time: {useTime:.2f} seconds")
        return PSNR_SSIM_HaarPSI(self.ref, abs(self.uk), 'ComeUs', self.save_path, useTime)

    def getGradientF(self, uk):
        gradf1 = self.T_mask * self.FS(uk) - self.T_ksdata
        gradf1 = ((self.FST(gradf1.real)).real - (self.FST(gradf1.imag)).imag)
        gradfF = Kernel_Rec_ks_C_I_Pro((1 - self.T_mask) * self.FS(uk) + self.T_ksdata, self.T_Ker, 1)
        gradf2 = self.FST(Kernel_Rec_ks_C_I_Pro(gradfF.real, self.T_Ker_Tra, 1 - self.T_mask)).real
        gradf2 = gradf2 - self.FST(Kernel_Rec_ks_C_I_Pro(gradfF.imag, self.T_Ker_Tra, 1 - self.T_mask)).imag
        return gradf1 + gradf2

    def UpdateSensitivityByCG(self):
        uk = self.T_sensitivity * self.uk.unsqueeze(-1)
        if self.target_S is None:
            target1 = IFFT2_3D_N(self.T_ksdata)
            target2 = Kernel_Rec_ks_C_I_Pro(self.T_ksdata, self.T_Ker, 1)
            target2 = IFFT2_3D_N(Kernel_Rec_ks_C_I_Pro(target2, self.T_Ker_Tra, 1 - self.T_mask))
            self.target_S = target1 - target2
        mu = SolverForSubProblem(lambda x: self.SensitivityFunction(x), self.target_S, uk, iter=5)
        self.T_sensitivity = mu / (sos(mu).unsqueeze(-1) + torch.finfo(torch.float32).eps)
        return sos(mu)

    def SensitivityFunction(self, mu):
        s1 = IFFT2_3D_N(self.T_mask * FFT2_3D_N(mu))
        s2 = Kernel_Rec_ks_C_I_Pro((1 - self.T_mask) * FFT2_3D_N(mu), self.T_Ker, 1)
        s2 = IFFT2_3D_N(Kernel_Rec_ks_C_I_Pro(s2, self.T_Ker_Tra, 1 - self.T_mask))
        return s1 + s2

    def UpdateSensitivityInMultiMethod(self):
        sk = self.T_sensitivity * self.uk.unsqueeze(-1)

        for iter in range(20):
            # iterative
            gradf1 = IFFT2_3D_N(self.T_mask * FFT2_3D_N(sk) - self.T_ksdata)
            gradf2 = Kernel_Rec_ks_C_I_Pro((1 - self.T_mask) * FFT2_3D_N(sk) + self.T_ksdata, self.T_Ker, 1)
            gradf2 = IFFT2_3D_N(Kernel_Rec_ks_C_I_Pro(gradf2, self.T_Ker_Tra, 1 - self.T_mask))
            gradfF = gradf1 + gradf2
            sk = sk - self.rho * gradfF

        self.T_sensitivity = sk / (sos(sk).unsqueeze(-1) + torch.finfo(torch.float32).eps)
        return sos(sk)
