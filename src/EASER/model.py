import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm

def create_list_feature_pairs(XtX, threshold):
    """
    상관 행렬의 비대칭적인 항목들 중에서 상관 값이 주어진 임계값 이상인 항목의 인덱스를 반환하는 함수.
    
    Args:
        XtX (numpy.ndarray): 데이터 행렬 X의 전치 행렬과 X의 곱으로 얻어진 행렬 (XtX = X.T @ X).
        threshold (float): 상관 값의 임계값. 이 값보다 큰 상관을 가진 항목만 선택된다.

    Returns:
        tuple: 두 개의 배열을 포함하는 튜플 (ii_pairs[0], ii_pairs[1])로, 상관 값이 threshold 이상인 항목들의 인덱스를 나타낸다.
    """
    AA = np.triu(np.abs(XtX))  # 행렬의 상삼각 행렬을 가져오되, 절댓값을 취한다.
    AA[np.diag_indices(AA.shape[0])] = 0.0  # 대각선 요소는 제외
    ii_pairs = np.where((AA > threshold) == True)  # 상관 값이 threshold 이상인 항목들의 인덱스를 반환
    return ii_pairs

def create_matrix_Z(ii_pairs, X):
    """
    특정 상관 쌍에 대해 선택된 항목들에 대한 행렬 Z를 생성하는 함수.
    
    Args:
        ii_pairs (tuple): (i, j) 형태의 인덱스 쌍으로, 상관 값이 임계값 이상인 항목들의 인덱스를 포함.
        X (numpy.ndarray): 원본 데이터 행렬.

    Returns:
        list: 두 개의 요소를 가진 리스트 [Z, CCmask]를 반환.
            - Z: 선택된 상관 쌍에 대해 원본 데이터와 연산한 결과 (희소 행렬 형태).
            - CCmask: 선택된 항목들에 대한 마스크 행렬.
    """
    MM = np.zeros((len(ii_pairs[0]), X.shape[1]), dtype=np.float64)  # 인덱스 쌍에 대해 희소 행렬을 생성
    MM[np.arange(MM.shape[0]), ii_pairs[0]] = 1.0  # 첫 번째 항목의 인덱스에 1 할당
    MM[np.arange(MM.shape[0]), ii_pairs[1]] = 1.0  # 두 번째 항목의 인덱스에 1 할당
    CCmask = 1.0 - MM  # 선택되지 않은 항목에 대해 마스크 생성
    MM = sparse.csc_matrix(MM.T)  # 희소 행렬로 변환
    Z = X * MM  # 원본 데이터와 희소 행렬을 곱하여 새로운 행렬 Z 생성
    Z = (Z == 2.0)  # Z 값이 2.0인 항목을 True로 설정
    Z = Z * 1.0  # True는 1.0으로 변환
    return [Z, CCmask]  # Z와 CCmask 반환

def train_higher(XtX, XtXdiag, lambdaBB, ZtZ, ZtZdiag, lambdaCC, CCmask, ZtX, rho, epochs):
    """
    고차원 모델 학습 함수. 주어진 매개변수들을 기반으로 반복적으로 파라미터를 업데이트하여 
    최적의 해를 찾는다.

    Args:
        XtX (numpy.ndarray): 데이터 행렬 X의 전치 행렬과 X의 곱.
        XtXdiag (numpy.ndarray): XtX의 대각선 요소들.
        lambdaBB (float): 정규화 파라미터로, XtX 행렬에 추가될 항목.
        ZtZ (numpy.ndarray): Z 행렬의 전치 행렬과 Z 행렬의 곱.
        ZtZdiag (numpy.ndarray): ZtZ의 대각선 요소들.
        lambdaCC (float): 정규화 파라미터로, ZtZ 행렬에 추가될 항목.
        CCmask (numpy.ndarray): CC 행렬에 대한 마스크.
        ZtX (numpy.ndarray): Z 행렬과 X 행렬의 전치 행렬의 곱.
        rho (float): 페널티 항에 대한 스칼라 값.
        epochs (int): 학습할 에포크 수.

    Returns:
        list: 학습된 결과인 [BB, DD]를 포함하는 리스트를 반환.
            - BB: 학습된 결과의 첫 번째 항목.
            - DD: 학습된 결과의 두 번째 항목.
    """
    ii_diag = np.diag_indices(XtX.shape[0])  # XtX의 대각선 인덱스를 추출
    XtX[ii_diag] = XtXdiag + lambdaBB  # XtX에 정규화 파라미터 추가
    PP = np.linalg.inv(XtX)  # XtX의 역행렬 계산

    ii_diag_ZZ = np.diag_indices(ZtZ.shape[0])  # ZtZ의 대각선 인덱스를 추출
    ZtZ[ii_diag_ZZ] = ZtZdiag + lambdaCC + rho  # ZtZ에 정규화 파라미터 및 rho 추가
    QQ = np.linalg.inv(ZtZ)  # ZtZ의 역행렬 계산

    # CC, DD, UU는 모델 학습을 위한 초기화된 파라미터들
    CC = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
    DD = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)
    UU = np.zeros((ZtZ.shape[0], XtX.shape[0]), dtype=np.float64)

    # 학습 반복
    for iter in tqdm(range(epochs)):
        XtX[ii_diag] = XtXdiag  # 대각선 값을 원래대로 복원
        BB = PP.dot(XtX - ZtX.T.dot(CC))  # 첫 번째 항목 계산
        gamma = np.diag(BB) / np.diag(PP)  # gamma 계산
        BB -= PP * gamma  # BB 값 업데이트

        CC = QQ.dot(ZtX - ZtX.dot(BB) + rho * (DD - UU))  # CC 값 업데이트
        DD = CC * CCmask  # DD는 CC와 CCmask의 곱
        UU += CC - DD  # UU는 CC와 DD의 차이값

    return [BB, DD]  # 학습된 BB와 DD 반환
