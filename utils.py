from keras.models import Model
from keras.layers import add, Input, Conv1D, Activation, Flatten, Dense
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam ,Adamax
from keras.layers import Input, Dense
from keras import layers
import warnings
from keras import optimizers
from keras import initializers
from keras.utils import plot_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller as adf
from statsmodels.stats.diagnostic import acorr_ljungbox
from keras.callbacks import EarlyStopping
from keras.initializers import he_normal
from keras.backend import reverse
from sklearn.model_selection import train_test_split
from scipy.signal import hilbert
from vmdpy import VMD
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau


#封装数据（分解uVMD)
def uVMD(data):
    N = len(data)
    Fs = 20000
    Ts = 1/Fs
    t = np.arange(N)
    k = np.arange(N)
    T = N/Fs
    frq = k/T
    frq1 = frq[range(int(N/2))]
#计算信号的中心频率：
# 参数： data：输入信号； Fs：采样频率
# 返回： center_freqs：中心频率数组
    def calculate_center_frequencies(data, Fs):
        freqs = np.fft.fftfreq(N, Ts)
        center_freqs = np.cumsum(freqs) / N
        return center_freqs
#使用中心频率法确定分解层数 K
# 参数： data：输入信号； Fs：采样频率 ;K_threshold：中心频率阈值
# 返回： K：分解层数
    def determine_K(data, Fs, K_threshold):
        center_freqs = calculate_center_frequencies(data, Fs)
        K = np.sum(center_freqs <= K_threshold)
        return K
#设置参数，但分解层数K和二次惩罚系数α需要仔细考虑设定，特别是K值，对分解效果影响巨大。
    alpha = 2000      # moderate bandwidth constraint
    tau = 0.01            # noise-tolerance (no strict fidelity enforcement)
    K=determine_K(data, Fs,1e-7)
    DC = 0             # no DC part imposed
    init = 1           # initialize omegas uniformly
    tol = 1e-7
    u, u_hat, omega = VMD(data, alpha, tau, K, DC, init, tol)
    return u[K-1, :]

def ceemdan(x, num_sifts, num_ensembles):
    ensembles = []
    for k in range(num_ensembles):
        h = x.copy()
        ensemble = []
        for _ in range(num_sifts):
            sd = np.std(h)
            h = sift(h, sd)
            ensemble.append(h)
        ensembles.append(ensemble)
    return ensembles

def sift(x, sd):
    imf = []
    while np.abs(sd) > 0.3:
        h = x.copy()
        sd_prev = sd + 1
        while sd < sd_prev:
            sd_prev = sd
            h = sift_iteration(h)
            sd = np.std(h)
        imf.append(h)
        x -= h
    imf.append(x)
    return imf

def sift_iteration(x):
    h = hilbert(x)
    envelope = np.abs(h)
    mean = np.mean(h)
    return np.where(h > mean, envelope, -envelope)

# 示例的方差评价函数
def evaluate_variance(x_train, x_val, num_sifts, num_ensembles):
    # 进行 CEEMDAN 分解
    imf_ensembles = ceemdan(x_train, num_sifts, num_ensembles)

    # 在验证集上计算评价指标（这里简单用方差作为评价指标）
    val_scores = []
    for ensemble in imf_ensembles:
        reconstructed_signal = np.sum(ensemble, axis=0)
        val_scores.append(np.var(x_val - reconstructed_signal))

    # 计算平均方差
    avg_variance = np.mean(val_scores)
    return avg_variance