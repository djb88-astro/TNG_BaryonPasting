import os
import h5py
import numpy as np

def copy_file_struct(f, hmap, fo):
    if type(hmap) == h5py._hl.group.Group:
        name = hmap.name.encode('ascii')

        # Copy attributes associated with group
        atts = f[name].attrs.keys()
        if len(atts) > 0:
            group = fo.create_group(name)
            for v in atts: group.attrs[v] = f[name].attrs[v]

        # Now deal with subgroups and datasets
        for w in hmap:
            if type(f[name][w]) == h5py._hl.group.Group:
                atts = f[name][w].attrs.keys()
                if len(atts) > 0:
                    group = fo.create_group('{0}/{1}'.format(name, w))
                    for v in atts: group.attrs[v] = f[name][w].attrs[v]
                copy_file_struct(f, f[name][w], fo)
            elif type(f[name][w]) == h5py._hl.dataset.Dataset:
                tmp    = np.zeros(f[name][w][:].shape, dtype=f[name][w][:].dtype)
                tmp[:] = f[name][w][:]
                dset   = fo.create_dataset('{0}/{1}'.format(name.decode("utf-8"), w), data=tmp, dtype=f[name][w][:].dtype)
                del tmp
                atts = f[name][w].attrs.keys()
                if len(atts) > 0:
                    for v in atts: dset.attrs[v] = f[name][w].attrs[v]
    elif type(hmap) ==  h5py._hl.dataset.Dataset:
        name   = hmap.name.encode('ascii')
        tmp    = np.zeros(f[name][:].shape, dtype=f[name][:].dtype)
        tmp[:] = f[name][:]
        dset   = fo.create_dataset(name.decode("utf-8"), data=tmp, dtype=f[name][:].dtype)
        atts   = f[name].attrs.keys()
        if len(atts) > 0:
            for v in atts: dset.attrs[v] = f[name].attrs[v]
        del tmp
    return

def merge_outputs(outputs):
    """
    Merge all outputs from different processors into one file
    """

    print(' > Merging {0}*'.format(outputs))
    files = []
    for y in os.listdir('output/'):
        if y.startswith(outputs): files.append('output/{0}'.format(y))
    files.sort()

    flib = 'output/{0}.hdf5'.format(outputs)
    outf = h5py.File(flib, 'w')

    for y in files:
        if y == flib: continue
        print('  -{0}'.format(y))
        f = h5py.File(y, 'r')
        fk = sorted(f.keys())
        for z in fk:
            print('  --> {0}'.format(z))
            copy_file_struct(f, f[z], outf)
        f.close()
        os.remove(y)
    outf.close()
    del files, flib
    return
