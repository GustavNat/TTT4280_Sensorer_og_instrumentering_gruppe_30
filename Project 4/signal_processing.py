import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import


f_0 = 24.13*10**9 #LO of radar
c = 3*10**8

# Slicing data (cutting unecessary data)


time_cut_each_data = [
                    [4.5,   7],     # 0.36m/s
                    [1.2,   3.2],   # 1.18m/s
                    [1,     4.5]]   #-0.87m/s


def the_function(file, time_cut_index):

    sample_period, data = raspi_import("Data/"+file)
    t = np.arange(data.shape[0]) * sample_period
    max_freq = int(2.8 * 10**3)

    sample_rate = int(1/sample_period)
    start_time =    time_cut_each_data[time_cut_index][0]
    end_time =      time_cut_each_data[time_cut_index][1]
    start_time_samples = int(start_time*sample_rate)
    end_time_sapmles = int(end_time*sample_rate)
    t = t[start_time_samples:end_time_sapmles]


    #Defining IF_I, IF_Q data (and scale them)

    IF_I = data[start_time_samples:end_time_sapmles, 0]
    IF_Q = data[start_time_samples:end_time_sapmles, 1] 

    IF_I = IF_I - np.mean(IF_I)
    IF_Q = IF_Q - np.mean(IF_Q)

    AI = np.sqrt(np.mean(IF_I**2))
    AQ = np.sqrt(np.mean(IF_Q**2))
    IF_Q = IF_Q * AI/AQ

    window = np.hanning(len(t))
    IF_I = IF_I*window
    IF_Q = IF_Q*window


    # Performing FFT

    x = IF_I + 1j*IF_Q
    X = np.fft.fft(x)
    freq = np.fft.fftfreq(len(x), sample_period)

    #chaning [0,+f, -f] -> [-f,0,f]
    X = np.fft.fftshift(X)
    freq = np.fft.fftshift(freq)


    #Cutting off f>max_freq
    mask = np.abs(freq) <= max_freq
    freq = freq[mask]
    amp = np.abs(X[mask])


    #Time domain:
    # plt.plot(t, IF_I)
    # plt.plot(t,IF_Q)
    # plt.show()

    #Finding speed
    k_max = np.argmax(amp)
    f_peak = freq[k_max]
    v_r = (f_peak*c)/(2*f_0)
    print("estimated speed: ",v_r,"m/s")

    #Calculating SNR
    eps = 1e-12
    amp_dB = 20*np.log10((amp + eps) / (np.max(amp) + eps))

    SNR_threshold = 50

    i1 = max(0, k_max - SNR_threshold)
    i2 = min(len(amp), k_max + SNR_threshold + 1)

    E_signal = np.sum(amp[i1:i2]**2)
    E_noise  = np.sum(amp[:i1]**2) + np.sum(amp[i2:]**2)
    SNR_dB = 10*np.log10(E_signal / E_noise)


    k_left = k_max
    k_right = k_max

    while amp_dB[k_left] > -3:
        k_left = k_left-1

    while amp_dB[k_left] > -3:
        k_right = k_right+1
    doppler_shift_resolution = freq[k_right]-freq[k_left]

    print("doppleroppløsing: ", doppler_shift_resolution, "Hz")

    #Frequency domain:
    # plt.plot(freq, amp_dB)
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Magnitude")
    # plt.show()

    return v_r


file_118_1 = "118ms_1.bin"
file_118_2 = "118ms_2.bin"
file_118_3 = "118ms_3.bin"
file_118_4 = "118ms_4.bin"

v_r_118 = []
v_r_118.append(the_function(file_118_1, 1))
v_r_118.append(the_function(file_118_2, 1))
v_r_118.append(the_function(file_118_3, 1))
v_r_118.append(the_function(file_118_4, 1))
v_r_118 = np.array(v_r_118)

sigma_118 = np.sqrt(1/len(v_r_118)*sum(v_r_118-1.18)**2)



file_036_1 = "036ms_1.bin"
file_036_2 = "036ms_2.bin"
file_036_3 = "036ms_3.bin"
file_036_4 = "036ms_4.bin"

v_r_036 = []
v_r_036.append(the_function(file_036_1, 1))
v_r_036.append(the_function(file_036_2, 1))
v_r_036.append(the_function(file_036_3, 1))
v_r_036.append(the_function(file_036_4, 1))
v_r_036 = np.array(v_r_036)

sigma_036 = np.sqrt(1/len(v_r_036)*sum(v_r_036-0.36)**2)


file_neg087_1 = "neg087ms_1.bin"
file_neg087_2 = "neg087ms_2.bin"
file_neg087_3 = "neg087ms_3.bin"
file_neg087_4 = "neg087ms_4.bin"

v_r_neg087 = []
v_r_neg087.append(the_function(file_neg087_1, 1))
v_r_neg087.append(the_function(file_neg087_2, 1))
v_r_neg087.append(the_function(file_neg087_3, 1))
v_r_neg087.append(the_function(file_neg087_4, 1))
v_r_neg087 = np.array(v_r_neg087)

sigma_neg087 = np.sqrt(1/len(v_r_neg087)*sum(v_r_neg087-(-0.87))**2)

print(sigma_036)
print(sigma_118)
print(sigma_neg087)