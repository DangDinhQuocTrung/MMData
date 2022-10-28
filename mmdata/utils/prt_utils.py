import math
import numpy as np


def fact_ratio(n, d):
    if n >= d:
        prod = 1.0
        for i in range(d + 1, n + 1):
            prod *= i
        return prod

    prod = 1.0
    for i in range(n + 1, d + 1):
        prod *= i
    return 1.0 / prod


def kval(m, ln):
    return math.sqrt(((2 * ln + 1) / (4 * math.pi)) * (fact_ratio(ln - m, ln + m)))


def associated_legendre(m, ln, x):
    if m < 0 or m > ln or np.max(np.abs(x)) > 1.0:
        return np.zeros_like(x)

    pmm = np.ones_like(x)
    if m > 0:
        somx2 = np.sqrt((1.0 + x) * (1.0 - x))
        fact = 1.0
        for i in range(1, m + 1):
            pmm = -pmm * fact * somx2
            fact = fact + 2

    if ln == m:
        return pmm

    pmmp1 = x * (2 * m + 1) * pmm
    if ln == m + 1:
        return pmmp1

    pll = np.zeros_like(x)
    for i in range(m + 2, ln + 1):
        pll = (x * (2 * i - 1) * pmmp1 - (i + m - 1) * pmm) / (i - m)
        pmm = pmmp1
        pmmp1 = pll
    return pll


def spherical_harmonic(m, ln, theta, phi):
    if m > 0:
        return math.sqrt(2.0) * kval(m, ln) * np.cos(m * phi) * associated_legendre(m, ln, np.cos(theta))
    elif m < 0:
        return math.sqrt(2.0) * kval(-m, ln) * np.sin(-m * phi) * associated_legendre(-m, ln, np.cos(theta))
    return kval(0, ln) * associated_legendre(0, ln, np.cos(theta))


def sample_spherical_directions(n):
    xv = np.random.rand(n, n)
    yv = np.random.rand(n, n)
    theta = np.arccos(1 - 2 * xv)
    phi = 2.0 * math.pi * yv

    phi = phi.reshape(-1)
    theta = theta.reshape(-1)

    vx = -np.sin(theta) * np.cos(phi)
    vy = -np.sin(theta) * np.sin(phi)
    vz = np.cos(theta)
    return np.stack([vx, vy, vz], 1), phi, theta


def get_sh_coeffs(order, phi, theta):
    shs = []
    for n in range(0, order + 1):
        for m in range(-n, n + 1):
            s = spherical_harmonic(m, n, theta, phi)
            shs.append(s)
    return np.stack(shs, 1)
