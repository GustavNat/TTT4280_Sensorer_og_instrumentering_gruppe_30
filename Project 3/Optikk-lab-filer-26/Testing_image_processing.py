import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

os.chdir(os.path.dirname(__file__)) #sets filepath to this folder



cap = cv2.VideoCapture("Measurements/56.mp4")

ret, frame = cap.read() #frame.shape = [w, h, channel], channels: BGR


for n in range(100):
    ret, frame = cap.read()

means = []
for n in range(2):
    ROI = cv2.selectROI("Select ROI", frame) #select area of the frame, get (x, y, w, h)
    x,y,w,h = ROI
    cropped = frame[y:y+h, x:x+w, :]
    mean_bgr = np.mean(cropped, axis=(0, 1))
    means.append(mean_bgr)
means = np.array(means)
print(means)

B = means[:, 0]
G = means[:, 1]
R = means[:, 2]

fps = cap.get(cv2.CAP_PROP_FPS) #gets the framerate
N = len(G)
t = np.arange(N) / fps



# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

R = R - np.mean(R)
G = G - np.mean(G)
B = B - np.mean(B)

plt.plot(t, R, label="Red")
plt.plot(t, G, label="Green")
plt.plot(t, B, label="Blue")
plt.xlabel("Time [s]")
plt.ylabel("Mean intensity")
plt.legend()
plt.show()

cap.release()


X = np.fft.rfft(G)
freq = np.fft.rfftfreq(len(G), d=1/fps)
amp = np.abs(X)

plt.plot(freq, amp)
plt.xlabel("Frequency [Hz]")
plt.ylabel("Amplitude")
plt.title("FFT spectrum")
plt.show()

peak_freq = freq[np.argmax(amp)]
print("Peak frequency:", peak_freq, "Hz")
print("BPM:", peak_freq * 60)


mask = (freq >= 0.7) & (freq <= 3.0)
peak_freq = freq[mask][np.argmax(amp[mask])] #find the maximum frequency within a spesific range of frequencies
pulse_bpm = peak_freq * 60
print(pulse_bpm)

# np.savetxt("rgb_data.txt", np.column_stack((R, G, B)))
# data = np.loadtxt("rgb_data.txt")
# R = data[:, 0]
# G = data[:, 1]
# B = data[:, 2]

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.show()