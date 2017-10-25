import numpy as np

from glob import glob

def read_file_as_array_tuple(filename, dtype=float):
    """
    Read in FAIMS datafile, returns a tuple of numpy arrays for the positive
    and negative ions respectively
    """
    positive_ion = np.genfromtxt(filename,
                                 dtype=dtype,
                                 delimiter='\t',
                                 skip_header=116,
                                 max_rows=51)

    negative_ion = np.genfromtxt(filename,
                                 dtype=dtype,
                                 delimiter='\t',
                                 skip_header=168,
                                 max_rows=51)

    return (positive_ion, negative_ion)

def read_files_as_array_tuple(file_list, dtype=float):
    """
    Read in a list of FAIMS datafiles, reutrns a tuple of numpy arrays for the
    positive and negative ions of shape len(file_list) x 512^2
    """
    positive_ion, negative_ion = [], []
    for filename in file_list:
        p, n = read_file_as_array_tuple(filename, dtype=dtype)
        positive_ion.append(p.flatten())
        negative_ion.append(n.flatten())

    positive_ion = np.vstack(positive_ion)
    negative_ion = np.vstack(negative_ion)

    return (positive_ion, negative_ion)

def read_faims_directory(path, pattern="export_matrix_0002.txt", dtype=float):
    """
    Read in a set of FAIMS files from a specific directory
    """
    file_list = glob(path + "*/" + pattern)
    positive_ion, negative_ion = read_files_as_array_tuple(file_list, dtype=dtype)
    labels = np.array([f.split('/')[-2] for f in file_list])

    return positive_ion, negative_ion, labels

