import numpy as np


muabo = np.genfromtxt("./muabo.txt", delimiter=",")
muabd = np.genfromtxt("./muabd.txt", delimiter=",")

red_wavelength = 600 # Replace with wavelength in nanometres
green_wavelength = 510 # Replace with wavelength in nanometres
blue_wavelength = 460 # Replace with wavelength in nanometres

wavelength = np.array([red_wavelength, green_wavelength, blue_wavelength])

def mua_blood_oxy(x): return np.interp(x, muabo[:, 0], muabo[:, 1])
def mua_blood_deoxy(x): return np.interp(x, muabd[:, 0], muabd[:, 1])

bvf = 0.01 # Blood volume fraction, average blood amount in tissue
oxy = 0.8 # Blood oxygenation

# Absorption coefficient ($\mu_a$ in lab text)
# Units: 1/m
mua_other = 25 # Background absorption due to collagen, et cetera
mua_blood = (mua_blood_oxy(wavelength)*oxy # Absorption due to
            + mua_blood_deoxy(wavelength)*(1-oxy)) # pure blood
mua = mua_blood*bvf + mua_other

# reduced scattering coefficient ($\mu_s^\prime$ in lab text)
# the numerical constants are thanks to N. Bashkatov, E. A. Genina and
# V. V. Tuchin. Optical properties of skin, subcutaneous and muscle
# tissues: A review. In: J. Innov. Opt. Health Sci., 4(1):9-38, 2011.
# Units: 1/m
musr = 100 * (17.6*(wavelength/500)**-4 + 18.78*(wavelength/500)**-0.22)

# mua and musr are now available as shape (3,) arrays
# Red, green and blue correspond to indexes 0, 1 and 2, respectively

# TODO calculate penetration depth


# 2.1 a)
delta = np.sqrt(1/(3*(mua+musr)*mua))

print("Penetration depth [m]:", delta)
print("Penetration depth[mm]: ", delta*10**3)

# 2.1 b)
C = np.sqrt(3*mua*(mua + musr))
d = 1.5*10**-2
T = np.e**(-C*d)
print("Transmittance:", T)


# 2.1 c)

probe_dept = np.e**(-2*C*d)
print("Probe depth:", probe_dept)

# 2.1 d)


# 2.1 d)

d = 300e-6  # 300 µm in meters

def calc_mua(bvf):
    mua_blood = (mua_blood_oxy(wavelength) * oxy
               + mua_blood_deoxy(wavelength) * (1 - oxy))
    return mua_blood * bvf + mua_other

# Tissue with 1% blood volume fraction
mua_tissue = calc_mua(0.01)
C_tissue = np.sqrt(3 * mua_tissue * (mua_tissue + musr))
T_tissue = np.exp(-C_tissue * d)

# Blood vessel with 100% blood volume fraction
mua_blood_only = calc_mua(1.0)
C_blood = np.sqrt(3 * mua_blood_only * (mua_blood_only + musr))
T_blood = np.exp(-C_blood * d)

# Contrast
K = np.abs(T_blood - T_tissue) / T_tissue

print("T through normal tissue (bvf = 1%):", T_tissue)
print("T through blood vessel (bvf = 100%):", T_blood)
print("Contrast:", K)
print("Contrast [%]:", K * 100)