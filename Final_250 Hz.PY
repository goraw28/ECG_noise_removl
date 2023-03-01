import numpy as np
from numpy import sqrt,mean,log,diff
import pandas as pd
import pywt
import scipy.signal as signal
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import glob
# import pybeads as be
from statistics import pvariance
from scipy.signal import savgol_filter
from BaselineRemoval import BaselineRemoval

def get_median_filter_width(sampling_rate, duration):
    res = int( sampling_rate*duration )
    res += ((res%2) - 1) # needs to be an odd number
    return res

def filter_signal(X):
    global mfa
    X0 = X  #read orignal signal
    for mi in range(0,len(mfa)):
        X0 = medfilt (X0,mfa[mi]) # apply median filter one by one on top of each other
    X0 = np.subtract(X,X0)  # finally subtract from orignal signal
    return X0

header_list = ['MLII', 'Value', 'VALUE', 'ECG', '0', 'val','V1', 'II', 'noise1', 'noise2','ii']

for i in glob.glob('C:\\Users\\x\\Desktop\\PVC\\PVC\\Digital\\360 Hz\\*.csv'):
    name = i.split("\\")[-1].split(".")[0] 
    print(name)
    # print(i)
    count = 0
    k = 0
    l= 3000
    # d = 'MLII'
    df= pd.read_csv(i)
    for header in header_list:
        try:
            df1 = pd.read_csv(i)[header]
        except:
            pass
    while True:

        initial_file = df1.iloc[k:l]
        if initial_file[k]<0.0001:
            x = df1.iloc[k:l]
        else:
            x = df1.iloc[k:l]
        
        ne1 = pd.DataFrame(x)
        for header in header_list:
            try:
                ne11 = ne1[header].tolist()
            except:
                pass
        q = pd.read_csv("250_normal.csv")["ECG"]
        ne2 = pd.DataFrame(q)
        ne22 = ne2['ECG'].tolist()
        a = np.linalg.norm(np.array(ne11))
        b = np.linalg.norm(np.array(ne22))
        outcome = (pvariance(ne22) / pvariance(ne11))
        print(outcome)
        if outcome < 0.1:
            print("NOISY")

            BASIC_SRATE= 100
            # baseline fitting by filtering
            # === Define Filtering Params for Baseline fitting Leads===
            ms_flt_array = [0.6,0.6]    #<-- length of baseline fitting filters (in seconds)
            mfa = np.zeros(len(ms_flt_array), dtype='int')
            for i in range(0, len(ms_flt_array)):
                mfa[i] = get_median_filter_width(BASIC_SRATE, ms_flt_array[i])

            polynomial_degree=3
            baseObj=BaselineRemoval(ne11)
            Modpoly_output=baseObj.IModPoly(polynomial_degree)
            base_line_shift = savgol_filter(Modpoly_output, 21, 12)
            
            signal_flt = filter_signal(base_line_shift)
            b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
            low_passed = signal.filtfilt(b, a, signal_flt)
            # wsignal = wavelet_denoising(signal_flt, wavelet='sym4', level=1)

            ax0 = plt.subplot(411)
            ax0.set_title("Raw Signal")
            ax0.plot(ne11)
            ax1 = plt.subplot(412, sharex = ax0, sharey =ax0)
            ax1.set_title("Base Line Shift")
            ax1.plot(base_line_shift)
            ax2 = plt.subplot(413, sharex = ax0, sharey =ax0)
            ax2.set_title("Baseline")
            ax2.plot(signal_flt)
            # ax3 = plt.subplot(514, sharex = ax0, sharey =ax0)
            # ax3.plot(wsignal)
            ax3 = plt.subplot(414, sharex = ax0, sharey =ax0)
            ax3.set_title("Low Pass")
            ax3.plot(low_passed)
            plt.show()
            # noisy = pd.DataFrame(low_passed)
            # noisy.to_csv('noisy.csv', index = False)
        else:
            print("NORMAL")

            BASIC_SRATE = 500
            # baseline fitting by filtering
            # === Define Filtering Params for Baseline fitting Leads===
            ms_flt_array = [0.6,0.6]    #<-- length of baseline fitting filters (in seconds)
            mfa = np.zeros(len(ms_flt_array), dtype='int')
            for i in range(0, len(ms_flt_array)):
                mfa[i] = get_median_filter_width(BASIC_SRATE, ms_flt_array[i])

            polynomial_degree=3
            baseObj=BaselineRemoval(ne11)
            Modpoly_output=baseObj.IModPoly(polynomial_degree)
            base_line_shift = savgol_filter(Modpoly_output, 21, 12)
            
            signal_flt = filter_signal(base_line_shift)
            # wsignal = wavelet_denoising(signal_flt, wavelet='sym4', level=1)
            b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
            low_passed = signal.filtfilt(b, a, signal_flt)

            ax0 = plt.subplot(411)
            ax0.set_title("Raw Signal")
            ax0.plot(ne11)
            ax1 = plt.subplot(412, sharex = ax0, sharey =ax0)
            ax1.set_title("Base Line Shift")
            ax1.plot(base_line_shift)
            ax2 = plt.subplot(413, sharex = ax0, sharey =ax0)
            ax2.set_title("Baseline")
            ax2.plot(signal_flt)
            # ax3 = plt.subplot(514, sharex = ax0, sharey =ax0)
            # ax3.plot(wsignal)
            ax3 = plt.subplot(414, sharex = ax0, sharey =ax0)
            ax3.set_title("Low Pass")
            ax3.plot(low_passed)
            plt.show()
        
        k+=3000
        l+=3000
        count+=1
        if l > len(df):
            break
    # qaz = pd.DataFrame(low_passed)
    # qaz.to_csv("2.csv")