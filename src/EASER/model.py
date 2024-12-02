import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

def create_list_feature_pairs(XtX, threshold):
    AA = np.triu(np.abs(XtX))
    AA[np.diag_indices(AA.shape[0])] = 0.0
    ii_pairs = np.where((AA > threshold) == True)
    return ii_pairs

def create_matrix_Z(ii_pairs, X):
    MM = np.zeros((len(ii_pairs[0]), X.shape[1]), dtype=np.float64)
    MM[np.arange(MM.shape[0]), ii_pairs[0]] = 1.0
    MM[np.arange(MM.shape[0]), ii_pairs[1]] = 1.0
    CCmask = 1.0 - MM  
    MM = sparse.csc_matrix(MM.T)
    Z = X * MM
    Z = (Z == 2.0)
    Z = Z * 1.0
    return [Z, CCmask]

def train_higher(XtX, XtXdiag, lambdaBB, ZtZ, ZtZdiag, lambdaCC, CCmask, ZtX, rho, epochs):
    ii_diag = np.diag_indices(XtX.shape[0])
    XtX[ii_diag] = XtXdiag + lambdaBB
    PP = np.linalg.inv(XtX)

    ii_diag_ZZ = np.diag_indices(ZtZ.shape[0])
    ZtZ[ii_diag_ZZ] = ZtZdiag + lambdaCC + rho
    QQ = np.linalg.inv(ZtZ)

    CC = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
    DD = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
    UU = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)

    for iter in tqdm(range(epochs)):
        XtX[ii_diag] = XtXdiag
        BB = PP.dot(XtX - ZtX.T.dot(CC))
        gamma = np.diag(BB) / np.diag(PP)
        BB -= PP * gamma

        CC = QQ.dot(ZtX - ZtX.dot(BB) + rho * (DD - UU))
        DD = CC * CCmask  
        UU += CC - DD

    return [BB, DD]