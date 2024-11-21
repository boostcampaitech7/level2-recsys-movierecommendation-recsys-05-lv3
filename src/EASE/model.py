import torch

class EASE:
    def __init__(self, _lambda,  device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.B = None
        self._lambda = _lambda
        self.device = device

    def train(self, X):
        try:
            if not isinstance(X, torch.Tensor):
                X = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
            
            print("X shape:", X.shape)
            
            G = X.T @ X
            print("G shape:", G.shape)
            
            diag_indices = torch.arange(G.size(0), device=self.device)
            G[diag_indices, diag_indices] += self._lambda
            print("G after adding lambda on diag:", G.shape)
            
            P = torch.linalg.pinv(G)
            print("P shape:", P.shape)
            
            diag_values = torch.diag(P)
            self.B = P / (-diag_values.view(-1, 1))
            self.B[diag_indices, diag_indices] = 0
            print("Final B shape:", self.B.shape)
            
        except Exception as e:
            print("Error occurred:", e)

    def predict(self, X):
        if self.B is None:
            raise ValueError("Model has not been trained yet!")
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X.toarray(), dtype=torch.float32).to(self.device)
        return (X @ self.B).cpu().numpy()