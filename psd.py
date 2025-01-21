import numpy as np
from scipy.ndimage import gaussian_filter1d

#compute power spectra distances and average across all dimensions
def power_spectrum_error(x_gen, x_true, sigma=1):
    pse_errors_per_dim, spectrum_true_per_dim, spectrum_gen_per_dim = power_spectrum_error_per_dim(
        x_gen, x_true, sigma
    )
    return (
        np.array(pse_errors_per_dim).mean(axis=0),
        np.array(spectrum_true_per_dim).mean(axis=0),
        np.array(spectrum_gen_per_dim).mean(axis=0)
    )

def compute_power_spectrum(x, sigma):
    fft_real = np.fft.rfft(x)
    ps = np.abs(fft_real)**2
    ps_smoothed = gaussian_filter1d(ps, sigma)
    return ps_smoothed

def get_average_spectrum(x, sigma):
    x_ = (x - x.mean()) / x.std()  # normalize individual trajectories
    spectrum = compute_power_spectrum(x_, sigma)
    return spectrum / spectrum.sum()

def power_spectrum_error_per_dim(x_gen, x_true, sigma):
    assert x_true.shape[1] == x_gen.shape[1]
    assert x_true.shape[2] == x_gen.shape[2]
    dim_x = x_gen.shape[2]
    pse_per_dim = []
    spectrum_true_per_dim = []
    spectrum_gen_per_dim = []
    for dim in range(dim_x):
        spectrum_true = get_average_spectrum(x_true[:, :, dim], sigma)
        spectrum_true_per_dim.append(spectrum_true)
        spectrum_gen = get_average_spectrum(x_gen[:, :, dim], sigma)
        spectrum_gen_per_dim.append(spectrum_gen)
        hd = hellinger_distance(spectrum_true, spectrum_gen)
        pse_per_dim.append(hd)
    return pse_per_dim, spectrum_true_per_dim, spectrum_gen_per_dim

def hellinger_distance(p, q):
    return 1 / np.sqrt(2) * np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q))**2))

#functions for smoothing power spectra with a Gaussian kernel
def kernel_smoothen(data, kernel_sigma=1):
    """
    Smoothen data with Gaussian kernel
    @param kernel_sigma: standard deviation of gaussian, kernel_size is adapted to that
    @return: internal data is modified but nothing returned
    """
    kernel = get_kernel(kernel_sigma)
    data_final = data.copy()
    data_conv = np.convolve(data[:], kernel)
    pad = int(len(kernel) / 2)
    data_final[:] = data_conv[pad:-pad]
    data = data_final
    return data

def gauss(x, sigma=1):
    return 1 / np.sqrt(2 * np.pi * sigma ** 2) * np.exp(-1 / 2 * (x / sigma) ** 2)

def get_kernel(sigma):
    size = sigma * 10 + 1
    kernel = list(range(size))
    kernel = [float(k) - int(size / 2) for k in kernel]
    kernel = [gauss(k, sigma) for k in kernel]
    kernel = [k / np.sum(kernel) for k in kernel]
    return kernel