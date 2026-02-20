import numpy as np

def estimate_delay(x, y, fs):
    """
    Estimate effective delay between x and y using cross-correlation.

    Uses the lab definition r_xy(m) = sum_n x(n)*y(n+m).
    Returns:
        m_hat : delay in samples (integer)
        dt_hat: delay in seconds
        rxy   : cross-correlation sequence (for optional plotting)
        m     : lag axis (samples), same length as rxy
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Optional but usually helpful: remove DC offset
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)

    # numpy.correlate is defined with the shift in the FIRST argument.
    # Using (y0, x0) gives: sum_n y(n+m)*x(n) = sum_n x(n)*y(n+m) = r_xy(m)
    rxy = np.correlate(y0, x0, mode="full")

    N = len(x0)  # assume same length as y0
    m = np.arange(-(N - 1), N)  # lags corresponding to rxy

    # Effective delay: maximize |r_xy(m)|
    idx = np.argmax(np.abs(rxy))
    m_hat = m[idx]
    dt_hat = m_hat / fs

    return m_hat, dt_hat, rxy, m