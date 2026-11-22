
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
    gas : str
        Gas type (case-insensitive): "Air", "O2", "CO", "CO2", "CH4", "C2H6".

    Returns
    -------
    c_gas : Gas sound speed (m/s), scalar or NumPy array
    """

    # Match gas names case-insensitively
    gas_lookup = {key.upper(): key for key in GAS_PROPERTIES}

    gas_key = gas_lookup.get(gas.strip().upper())

    if gas_key is None:
        raise ValueError(
            f"Unknown gas '{gas}'. Available gases: {', '.join(GAS_PROPERTIES)}"
        )

    R = GAS_PROPERTIES[gas_key]["R"]
    k = GAS_PROPERTIES[gas_key]["k"]
    T = np.asarray(T_celsius, dtype=float) + 273.15

    if not np.all(np.isfinite(T)) or np.any(T <= 0):
        raise ValueError("Temperature must be above absolute zero.")

    c_gas = np.sqrt(k*R*T)
    

    return c_gas.item() if c_gas.ndim == 0 else c_gas


