'''Merge single embeddings matrices together.'''
import glob
import os
import scipy.io as sio

IN_DIR = '.'  # 'embeddings'
OUT_FILE = 'embeddings.mat'  # os.path.join(IN_DIR, 'embeddings.mat')

if __name__ == '__main__':
    if os.path.exists(OUT_FILE):
        import sys; sys.exit('Already merged: {}'.format(OUT_FILE))

    FILES = [os.path.join(IN_DIR, f) for f in glob.glob('{}/embeddings*[0-9].mat'.format(IN_DIR))]

    EMB_MATS = [sio.loadmat(f) for f in FILES]
    EMB_MATS = sorted(EMB_MATS, key=lambda e: int([k for k in e.keys() if k[0:2] == 'U_'][0][2:]))
    EMB_ALL = EMB_MATS[0]
    EMB_MATS = EMB_MATS[1:]

    for emb in EMB_MATS:
        key = [k for k in emb.keys() if k[0:2] == 'U_'][0]
        # print(key)
        EMB_ALL[key] = emb[key]

    sio.savemat(OUT_FILE, EMB_ALL)
