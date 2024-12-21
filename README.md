<p align="center">

  <h1> 🎞️ Movie Recommendation </h1>

  > 이 프로젝트는 사용자의 영화 시청 이력 데이터를 바탕으로 사용자가 다음에 시청할 영화 및 좋아할 영화를 예측하는 것이 목적입니다.


</p>

<br>


## 👨🏼‍💻 Members
공지원|김주은|류지훈|박세연|박재현|백상민|
:-:|:-:|:-:|:-:|:-:|:-:
<img src='https://avatars.githubusercontent.com/annakong23' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/kimjueun028' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/JihoonRyu00' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/SayOny' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/JaeHyun11' height=60 width=60></img>|<img src='https://avatars.githubusercontent.com/gagoory7' height=60 width=60></img>|
<a href="https://github.com/annakong23" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/kimjueun028" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/JihoonRyu00" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/SayOny" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>|<a href="https://github.com/JaeHyun11" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/>|<a href="https://github.com/gagoory7" target="_blank"><img src="https://img.shields.io/badge/GitHub-black.svg?&style=round&logo=github"/></a>

<br>

## 🛠️ 기술 스택 및 협업
  <img src="https://img.shields.io/badge/Python-3776AB?style=square&logo=Python&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/Pandas-150458?style=square&logo=Pandas&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/scikitlearn-F7931E?style=square&logo=scikitlearn&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=PyTorch&logoColor=white"/>&nbsp;

  <img src="https://img.shields.io/badge/Jira-0052CC?style=flat-square&logo=Jira&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/Confluence-0052CC?style=flat-square&logo=Jira&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/Notion-000000?style=square&logo=Notion&logoColor=white"/>&nbsp;
  <img src="https://img.shields.io/badge/Slack-4A154B?style=flat-square&logo=Slack&logoColor=white"/>&nbsp;


<br>

## 📁 Directory
```bash
project
├── README.md
├── main.py
├── config/
│   ├── model_config.yaml
│   └── model_weights.yaml
├── data/
├── notebook/
├── saved/
│   ├── output/
│   └── preprocessed/
└── src
    ├── ADMMSLIM/
    ├── BERT4Rec/
    ├── CDAE/
    ├── DeepFM/
    ├── EASE/
    ├── EASER/
    ├── FM/
    ├── LightGCN/
    ├── MultiVAE/
    ├── NCF/
    ├── RecVAE/
    ├── SASRec/
    └── ensemble/
```
<br>

# 🏃 How to run
## Config File

기본 config 파일은 아래와 같으며, 새로운 config 파일을 정의하셔도 됩니다.

또한, 모델 실행 시 Default 파라미터가 정의되어 있습니다.

__model_config.yaml__

```bash
seed : 0
device: cpu # 장치 설정
model: EASE # 기본 모델


model_args:
  모델명:
    파라미터1:
    파라미터2:

dataset :
  data_path : data/train/ # 학습 데이터 불러오는 곳
  output_path : saved/output # 예측한 결과 저장할 곳
  preprocessing_path : saved/preprocessed/ # 전처리된 파일이 저장될 곳

```

## 전처리 & 학습 & 예측

전처리 & 학습 & 예측을 동시에 하려면 다음 명령어를 사용하세요:

```bash
python main.py -c config/model_args -m Model -p param1 value1 param2 value2 ...
```

EASE 모델 실행을 원하면, 다음 명령어를 사용하세요:

```bash
python main.py -c config/model_config.yaml -m EASE -p _lambda 1000
```

EASER 모델 실행을 원하면, 다음 명령어를 사용하세요:

```bash
python main.py -c config/model_config.yaml -m EASER -p epochs 1000 rho 50000
```


자세한 파싱 정보는 main.py를 참고해주세요.

