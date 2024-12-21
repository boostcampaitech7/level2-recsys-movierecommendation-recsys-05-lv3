import numpy as np
from scipy import sparse

class BaseSlimModel(object):
    """
    기본 SLIM 모델 클래스.

    Methods:
        fit(X): 입력 데이터 행렬 X를 학습하고 계수를 초기화합니다.
        predict(X): 학습된 계수를 바탕으로 점수를 예측합니다.
        recommend(X, top): 상위 N개의 추천 아이템을 반환합니다.

    Args:
        X (scipy.sparse matrix): 사용자-아이템 상호작용 행렬.
        top (int, optional): 추천할 아이템 수. 기본값은 20.
    """
    def fit(self, X):
        self.coef = np.identity(X.shape[1])

    def predict(self, X):
        return X.dot(self.coef)

    def recommend(self, X, top=20):
        scores = self.predict(X)
        top_items = np.argsort(scores, axis=1)[:, -top:]
        return top_items

class AdmmSlim(BaseSlimModel):
    """
    ADMMSLIM 모델 클래스.

    ADMM 최적화 방법을 사용하여 SLIM(스파스 선형 항등 모델)을 학습합니다.

    Args:
        lambda_1 (float, optional): L1 정규화 항의 가중치. 기본값은 1.
        lambda_2 (float, optional): L2 정규화 항의 가중치. 기본값은 500.
        rho (float, optional): ADMM의 페널티 매개변수. 기본값은 10000.
        positive (bool, optional): 계수를 양수로 제한할지 여부. 기본값은 True.
        n_iter (int, optional): 최대 반복 횟수. 기본값은 50.
        eps_rel (float, optional): 상대 허용 오차. 기본값은 1e-4.
        eps_abs (float, optional): 절대 허용 오차. 기본값은 1e-3.
        verbose (bool, optional): 학습 중 로그를 출력할지 여부. 기본값은 False.

    Methods:
        fit(X): 사용자-아이템 상호작용 행렬을 사용해 모델을 학습합니다.
        soft_thresholding(B, Gamma): 소프트 임계값 조정을 수행합니다.
        is_converged(B, C, C_old, Gamma): 학습이 수렴했는지 판단합니다.
    """

    def __init__(self, lambda_1=1, lambda_2=500, rho=10000,
                 positive=True, n_iter=50, eps_rel=1e-4, eps_abs=1e-3,
                 verbose=False):
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.rho = rho
        self.positive = positive
        self.n_iter = n_iter
        self.eps_rel = eps_rel
        self.eps_abs = eps_abs
        self.verbose = verbose

    def soft_thresholding(self, B, Gamma):
        if self.lambda_1 == 0:
            if self.positive:
                return np.abs(B)
            else:
                return B
        else:
            x = B + Gamma / self.rho
            threshold = self.lambda_1 / self.rho
            if self.positive:
                return np.where(threshold < x, x - threshold, 0)
            else:
                return np.where(threshold < x, x - threshold,
                                np.where(x < - threshold, x + threshold, 0))

    def is_converged(self, B, C, C_old, Gamma):
        B_norm = np.linalg.norm(B)
        C_norm = np.linalg.norm(C)
        Gamma_norm = np.linalg.norm(Gamma)

        eps_primal = self.eps_abs * B.shape[0] - self.eps_rel * np.max([B_norm, C_norm])
        eps_dual = self.eps_abs * B.shape[0] - self.eps_rel * Gamma_norm

        R_primal_norm = np.linalg.norm(B - C)
        R_dual_norm = np.linalg.norm(C  - C_old) * self.rho

        converged = R_primal_norm < eps_primal and R_dual_norm < eps_dual
        return converged

    def fit(self, X):
        XtX = X.T.dot(X)
        if sparse.issparse(XtX):
            XtX = XtX.todense().A

        if self.verbose:
            print(' --- init')
        identity_mat = np.identity(XtX.shape[0])
        diags = identity_mat * (self.lambda_2 + self.rho)
        P = np.linalg.inv(XtX + diags).astype(np.float32)
        B_aux = P.dot(XtX)

        Gamma = np.zeros_like(XtX, dtype=np.float32)
        C = np.zeros_like(XtX, dtype=np.float32)

        self.primal_residuals = []
        self.dual_residuals = []

        for iter in range(self.n_iter):
            if self.verbose:
                print(f'Iteration {iter+1}')
            C_old = C.copy()
            B_tilde = B_aux + P.dot(self.rho * C - Gamma)
            gamma = np.diag(B_tilde) / (np.diag(P) + 1e-8)
            B = B_tilde - P * gamma
            C = self.soft_thresholding(B, Gamma)
            Gamma = Gamma + self.rho * (B - C)

            R_primal_norm = np.linalg.norm(B - C)
            R_dual_norm = np.linalg.norm(C - C_old) * self.rho
            self.primal_residuals.append(R_primal_norm)
            self.dual_residuals.append(R_dual_norm)

            if self.verbose:
                print(f'     Primal Residual Norm: {R_primal_norm:.6f}')
                print(f'     Dual Residual Norm: {R_dual_norm:.6f}')

            if self.is_converged(B, C, C_old, Gamma):
                if self.verbose:
                    print(f' --- Converged at iteration {iter+1}. Stopped iteration.')
                break

        self.coef = C