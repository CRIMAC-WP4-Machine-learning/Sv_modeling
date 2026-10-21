def func_FEM_BEM(_Dir, _FileName):
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    import pandas as pd
    import re

    File = os.path.join(_Dir,_FileName)
    with open(File, mode='r') as file:
        # lines = csv.reader(file, delimiter='\t')
        # # Skip the header row (first row)
        # next(lines)
        lines = file.readlines()[5:]
    
        # Loop through the rows and print them
        freq_bem = []
        TS_bem = []
        for row in lines:
            # print(row)
            split_row = row.split()
            # print(len(split_row))
            freq_bem.append(float(split_row[1]))  # Second column: freq (Hz)
            TS_bem.append(float(split_row[2]))  # Third column: TS_Global_var_Point

    # Convert to NumPy array
    freq_bem = np.array(freq_bem)
    TS_bem = np.array(TS_bem)

    return freq_bem, TS_bem