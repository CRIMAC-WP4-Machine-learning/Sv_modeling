
import numpy as np

GAS_PROPERTIES = {
    "Air": {
            "name" : "Air",
            "R": 287.0,       # J/(kg K)
            "k": 1.4,    # specific heat ratio 
        },
    "O2": {
        "name" : "Oxigen",
        "R": 259.8,       # J/(kg K)
            "k": 1.395,    # specific heat ratio 
        },
    "CO": {
        "name" : "Carbon monoxide",
        "R": 296.8,       # J/(kg K)
            "k": 1.4,    # specific heat ratio 
        },
    "CO2": {
        "name" : "Carbon dioxide",
        "R": 188.9,       # J/(kg K)
            "k": 1.289,    # specific heat ratio 
        },
    "CH4": {
        "name" : "Methane",
        "R": 518.2,       # J/(kg K)
            "k": 1.299,    # specific heat ratio 
        },
    "C2H6": {
        "name" : "Ethane",
        "R": 276.5,       # J/(kg K)
            "k": 1.186,    # specific heat ratio 
        }
}


def func_c_gas(T_celsius, gas="O2"):
    """
    Calculate gas sound speed using c = sqrt(kRT).

    T_celsius : Temperature (°C), scalar or NumPy array
    gas       : "O2" or "CO"

    Returns
    -------
    c_gas : Gas sound speed (m/s), scalar or NumPy array
    """

    gas = gas.upper()
    if gas not in GAS_PROPERTIES:
        raise ValueError(f"Unknown gas: {gas}")

    R = GAS_PROPERTIES[gas]["R"]
    k = GAS_PROPERTIES[gas]["k"]
    T = T_celsius + 273.15

    if not np.all(np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("Temperature must be above absolute zero.")

    c_gas = (k*R*T)**0.5
    

    return c_gas


