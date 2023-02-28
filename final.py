import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.signal import medfilt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy import signal
from statistics import pvariance


# Baseline shifting and wandering

def lowpass(baseline_data):
    b, a = signal.butter(3, 0.3, btype='lowpass', analog=False)
    low_passed = signal.filtfilt(b, a, baseline_data)
    return low_passed

def baseline_reconstruction(data, kernel_Size):
    s_corrected = signal.detrend(data)
    baseline_corrected = s_corrected - medfilt(s_corrected,kernel_Size)
    return baseline_corrected

def ecg_labeling(signal):
    
    gradient = np.gradient(signal.values)
    gradient_pVAR = (pvariance(gradient)/pvariance(signal.values) )
    gradient_std = np.std(gradient)
    std = np.std(signal.values)
    corelation_val, p_val = pearsonr(gradient, signal.values)
    corelation_val = abs(corelation_val)
    
    if (gradient_pVAR<0.068 and std>1.09 )or std <0.001 or corelation_val < 2.41237972862723E-17:
        initial_label = "Non_ECG"
    elif (gradient_std<0.007 or gradient_std>0.28) and (std <0.063 or std>0.69):
            initial_label = "Non_ECG"
    elif gradient_pVAR> 0.95:
        initial_label = "Non_ECG"
    elif gradient_pVAR<0.09 and (std>0.69 and std<=1.2) :
        initial_label = "high_noise"
    else:
        initial_label = "ECG"
    
    return initial_label

def final_process(signal, label):
    if  label =="high_noise":
        baseline = baseline_reconstruction(data =signal,kernel_Size=51)
        low_pass_output = lowpass(baseline_data = baseline)

    else:
        baseline = baseline_reconstruction(data =signal,kernel_Size=101)
        low_pass_output = lowpass(baseline_data = baseline)

        
    return signal, label, baseline, low_pass_output

if __name__=="__main__":
    import glob
    
    path = input("Path : ")
    
    for i in glob.glob(f'{path}\\*.csv'):
        name = i.split("\\")[-1].split(".")[0]
        
        header_list = ['MLII', 'Value', 'VALUE', 'ECG', '0', 'val','V1', 'II', 'noise1', 'noise2','ii']
        for header in header_list:
            try:
                ecg_signal = pd.read_csv(i)[header]
            except:
                pass
        
        label = ecg_labeling(ecg_signal)
        if label != "Non_ECG":
            initial_output,label_, base, final = final_process(ecg_signal, label)
            ax0 = plt.subplot(411)
            ax0.plot(initial_output, label = label_)
            plt.legend()
            ax1 = plt.subplot(412, sharex = ax0, sharey =ax0)
            if label_ == "ECG":
                ax1.plot(initial_output, 'g', label = label_)
                plt.legend()
            elif label_ == "high_noise":
                ax1.plot(initial_output, 'r', label = label_)
                plt.legend()
            ax2 = plt.subplot(413, sharex = ax0, sharey =ax0)
            ax2.plot(base, label = "Baseline")
            plt.legend()
            ax3 = plt.subplot(414, sharex = ax0, sharey =ax0)
            ax3.plot(final, label = 'Low Pass')
            plt.legend()
            plt.show()
        
        else:
            ax0 = plt.subplot(411)
            ax0.plot(ecg_signal, label = "Non_ECG")
            plt.legend()
            ax1 = plt.subplot(412, sharex = ax0, sharey =ax0)
            ax1.plot(ecg_signal, 'black', label = "Non_ECG")
            plt.legend()
            ax2 = plt.subplot(413, sharex = ax0, sharey =ax0)
            ax2.plot(np.array(np.zeros(len(ecg_signal))), label = "Baseline")
            plt.legend()
            ax3 = plt.subplot(414, sharex = ax0, sharey =ax0)
            ax3.plot(np.array(np.zeros(len(ecg_signal))), label = 'Low Pass')
            plt.legend()
            plt.show()