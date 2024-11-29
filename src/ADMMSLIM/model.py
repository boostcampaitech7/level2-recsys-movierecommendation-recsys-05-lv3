import numpy as np
from scipy import sparse

class BaseSlimModel(object):
    def fit(self, X):
        self.coef = np.identity(X.shape[1])

    def predict(self, X):
        return X.dot(self.coef)

    def recommend(self, X, top=20):
        scores = self.predict(X)
        top_items = np.argsort(scores, axis=1)[:, -top:]
        return top_items

class AdmmSlim(BaseSlimModel):
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