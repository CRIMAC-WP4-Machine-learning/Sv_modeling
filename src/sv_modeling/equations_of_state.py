
import numpy as np

GAS_PROPERTIES = {
    "Air": {
            "name" : "Air",
            "R": 287.0,       # J/(kg K)
            "T_cr": 132.5,    # K
            "P_cr": 3.77e6,   # Pa
        },
    "O2": {
        "name" : "Oxigen",
        "R": 259.8,       # J/(kg K)
        "T_cr": 154.8,    # K
        "P_cr": 5.08e6,   # Pa
    },
    "CO": {
        "name" : "Carbon monoxide",
        "R": 296.8,       # J/(kg K)
        "T_cr": 133.0,    # K
        "P_cr": 3.5e6,    # Pa
    },
    "CO2": {
        "name" : "Carbon dioxide",
        "R": 188.9,       # J/(kg K)
        "T_cr": 304.2,    # K
        "P_cr": 7.39e6,    # Pa
    },
    "CH4": {
        "name" : "Methane",
        "R": 518.2,       # J/(kg K)
        "T_cr": 191.1,    # K
        "P_cr": 4.64e6,    # Pa
    },
    "C2H6": {
        "name" : "Ethane",
        "R": 276.5,       # J/(kg K)
        "T_cr": 305.5,    # K
        "P_cr": 4.48e6,    # Pa
    }
}


def func_VanDerWaals(P_pa, T_celsius, gas="O2"):
    """
    Calculate gas density using the Van der Waals equation.

    P_pa      : Absolute pressure (Pa), scalar or NumPy array
    T_celsius : Temperature (°C), scalar or NumPy array
    gas       : "O2" or "CO"

    Returns
    -------
    rho : Gas density (kg/m³), scalar or NumPy array
    """

    gas = gas.upper()
    if gas not in GAS_PROPERTIES:
        raise ValueError(f"Unknown gas: {gas}")

    R = GAS_PROPERTIES[gas]["R"]
    T_cr = GAS_PROPERTIES[gas]["T_cr"]
    P_cr = GAS_PROPERTIES[gas]["P_cr"]

    P, T_c = np.broadcast_arrays(np.asarray(P_pa, dtype=float), np.asarray(T_celsius, dtype=float))
    T = T_c + 273.15

    if not np.all(np.isfinite(P)) or np.any(P <= 0):
        raise ValueError("Pressure must be finite and positive.")
    if not np.all(np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("Temperature must be above absolute zero.")

    # Van der Waals constants (SI units)
    a = 27 * R**2 * T_cr**2 / (64 * P_cr)
    b = R * T_cr / (8 * P_cr)

    # Cubic equation: P*v³ - (P*b + R*T)*v² + a*v - a*b = 0
    rho = np.empty(P.shape, dtype=float)

    for idx in np.ndindex(P.shape):
        p, t = P[idx], T[idx]
        coefficients = [p, -(p*b + R*t), a, -a*b]
        roots = np.roots(coefficients)

        # Select physically admissible real roots
        real_roots = roots[np.abs(roots.imag) <= 1e-9 * np.maximum(1, np.abs(roots.real))].real
        valid_roots = real_roots[real_roots > b]

        if valid_roots.size == 0:
            raise RuntimeError(f"No valid Van der Waals root for P={p}, T={t}.")

        # Gas-like branch: largest specific volume
        v = np.max(valid_roots)
        rho[idx] = 1.0 / v

    return rho.item() if rho.ndim == 0 else rho


# Optional wrappers preserving your original function names

# def func_VanDerWaals_Oxygen(P_pa, T_celsius):
#     return func_VanDerWaals(P_pa, T_celsius, gas="O2")


# def func_VanDerWaals_CarbonMonoxide(P_pa, T_celsius):
#     return func_VanDerWaals(P_pa, T_celsius, gas="CO")
