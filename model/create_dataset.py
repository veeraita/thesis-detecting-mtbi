import os
import sys
import numpy as np
from glob import glob

controls = ['%03d' % n for n in range(28, 48)]

def main(data_dir, output_dir):
    files = glob(os.path.join(data_dir), '*-stc-data.csv')

    norm_dataset = []
    cases_dataset = []
    for file in files:
        data = np.genfromtxt(file, delimiter=',')
        if file.startswith('sub-') or file[:3] in controls:
            norm_dataset.append(data)
        elif file[:3] not in controls:
            cases_dataset.append(data)
        else:
            continue

    norm_dataset_arr = np.array(norm_dataset)
    cases_dataset_arr = np.array(cases_dataset)

    np.savetxt(os.path.join(output_dir, 'normative_data.csv'), norm_dataset_arr, delimiter=",")
    np.savetxt(os.path.join(output_dir, 'cases_data.csv'), cases_dataset_arr, delimiter=",")


if __name__ == "__main__":
    main(*sys.argv[1:])