import numpy as np
from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F


def swish(x):
    """
    Swish 활성화 함수.
    
    Swish 함수는 ReLU의 개선된 버전으로, 입력값에 대해 양수일 때는 ReLU처럼 동작하고,
    음수일 때는 부드럽게 변화하는 활성화 함수입니다.
    
    Args:
        x (torch.Tensor): 입력 텐서.
    
    Returns:
        torch.Tensor: Swish 활성화가 적용된 결과 텐서.
    """
    return x.mul(torch.sigmoid(x))


def log_norm_pdf(x, mu, logvar):
    """
    정규 분포의 로그 확률 밀도 함수 (Log-Normal PDF).
    
    주어진 `mu` (평균)와 `logvar` (로그 분산)을 사용하여 정규 분포의 로그 확률 밀도를 계산합니다.
    
    Args:
        x (torch.Tensor): 입력 값.
        mu (torch.Tensor): 정규 분포의 평균.
        logvar (torch.Tensor): 정규 분포의 로그 분산.
    
    Returns:
        torch.Tensor: 계산된 로그 확률 밀도.
    """
    return -0.5 * (logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    """
    혼합된 사전 분포(Composite Prior)를 정의하는 클래스.
    
    이 클래스는 표준 Gaussian, Posterior (인코더에서 나온), Uniform prior를 혼합하여 사용합니다.
    여러 종류의 사전 분포를 합성하여 복잡한 분포를 모델링합니다.
    
    Args:
        hidden_dim (int): 은닉층 차원.
        latent_dim (int): 잠재 변수 차원.
        input_dim (int): 입력 데이터 차원.
        mixture_weights (list): 각 사전 분포의 가중치.
    """
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        # 사전 분포들 초기화
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        # 이전 인코더 모델 (freeze)
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        """
        주어진 `x`와 `z`에 대해 Composite Prior을 계산합니다.
        
        Args:
            x (torch.Tensor): 입력 데이터 (사용자-아이템 상호작용).
            z (torch.Tensor): 잠재 변수.
        
        Returns:
            torch.Tensor: 혼합된 prior 확률 밀도의 로그값.
        """
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        # 각 prior 분포 계산
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        # 각 분포에 가중치 적용
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        # 혼합된 분포의 밀도 값 계산
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)


class Encoder(nn.Module):
    """
    VAE의 인코더 네트워크.
    
    입력 데이터를 잠재 공간(latent space)으로 인코딩하는 네트워크입니다.
    여러 개의 Fully Connected Layer와 Layer Normalization을 사용하여 안정적인 학습을 도와줍니다.
    
    Args:
        hidden_dim (int): 은닉층 차원.
        latent_dim (int): 잠재 변수 차원.
        input_dim (int): 입력 데이터 차원.
        eps (float): Layer Normalization에 사용할 작은 값.
    """
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        """
        인코더의 순전파 과정.
        
        입력 `x`를 잠재 공간으로 변환하는 과정입니다. Dropout을 사용하여 모델의 일반화 성능을 향상시킵니다.
        
        Args:
            x (torch.Tensor): 입력 데이터 (사용자-아이템 상호작용).
            dropout_rate (float): Dropout 비율.
        
        Returns:
            tuple: 잠재 변수의 평균 (`mu`)과 로그 분산 (`logvar`).
        """
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        
        return self.fc_mu(h5), self.fc_logvar(h5)


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) 모델.
    
    입력 데이터에 대한 잠재 변수를 인코딩하고, 이 잠재 변수를 통해 예측된 출력을 디코딩합니다.
    VAE는 불확실성을 모델링하며, KL Divergence와 ELBO 손실 함수를 사용하여 학습됩니다.
    
    Args:
        hidden_dim (int): 은닉층 차원.
        latent_dim (int): 잠재 변수 차원.
        input_dim (int): 입력 데이터 차원.
    """
    def __init__(self, hidden_dim, latent_dim, input_dim):
        super(VAE, self).__init__()

        # Encoder와 Decoder 정의
        self.encoder = Encoder(hidden_dim=hidden_dim, latent_dim=latent_dim, input_dim=input_dim)
        self.prior = CompositePrior(hidden_dim=hidden_dim, latent_dim=latent_dim, input_dim=input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        잠재 변수 `z`를 샘플링하는 reparameterization trick.
        
        학습 중에는 평균 `mu`와 로그 분산 `logvar`를 사용하여 `z`를 샘플링합니다. 
        테스트 중에는 평균 `mu`를 사용하여 `z`를 반환합니다.
        
        Args:
            mu (torch.Tensor): 평균.
            logvar (torch.Tensor): 로그 분산.
        
        Returns:
            torch.Tensor: 샘플링된 잠재 변수 `z`.
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, user_ratings, beta=None, gamma=1.0, dropout_rate=0.5, calculate_loss=True):
        """
        VAE 모델의 순전파 과정.

        입력된 사용자 평점 데이터(`user_ratings`)를 통해 잠재 변수 `z`를 샘플링하고, 이를 바탕으로 
        예측된 평점(`x_pred`)을 계산합니다. 또한, 모델이 학습 중일 경우 ELBO 손실을 계산하여 
        반환합니다.

        Args:
            user_ratings (torch.Tensor): 사용자 평점 입력 텐서. 크기 (batch_size, input_dim).
            beta (float, optional): ELBO 손실에서 KL divergence에 대한 가중치. 기본값은 `None`.
            gamma (float, optional): KL divergence에 대한 스케일링 팩터. 기본값은 `1.0`.
            dropout_rate (float, optional): 인코더에서 사용할 dropout 비율. 기본값은 `0.5`.
            calculate_loss (bool, optional): 손실 값을 계산하고 반환할지 여부. 기본값은 `True`.

        Returns:
            tuple: `calculate_loss`가 `True`인 경우, MLL(Marginal Log Likelihood)과 KLD(KL Divergence)의 
                   값과 음의 ELBO(ELBO loss) 값을 반환합니다.
                   `calculate_loss`가 `False`인 경우, 예측된 평점 `x_pred`를 반환합니다.
        """
        # 인코딩
        mu, logvar = self.encoder(user_ratings, dropout_rate=dropout_rate)
        z = self.reparameterize(mu, logvar)

        # 디코딩
        x_pred = self.decoder(z)

        if calculate_loss:
            # KL divergence에 대한 가중치 설정
            kl_weight = gamma * user_ratings.sum(dim=-1) if gamma else beta

            # MLL (Marginal Log Likelihood) 계산
            mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()

            # KLD (KL Divergence) 계산
            kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()

            # 음의 ELBO 손실 계산
            negative_elbo = -(mll - kld)

            return (mll, kld), negative_elbo
        else:
            return x_pred

    def update_prior(self):
        """
        현재 인코더의 파라미터를 사용하여 prior를 업데이트합니다.

        이 함수는 인코더 네트워크가 업데이트될 때마다, 그 파라미터를 복사하여 
        CompositePrior에 있는 `encoder_old`에 저장합니다. 이 업데이트된 prior는 
        모델 학습의 일부로 사용됩니다.
        """
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))