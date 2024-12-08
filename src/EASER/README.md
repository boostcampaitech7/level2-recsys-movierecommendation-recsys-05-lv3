---

##### 이 코드는 아래 논문과 오픈소스 코드를 참고하였습니다.

Steck, Harald, and Dawen Liang. "Negative interactions for improved collaborative filtering Don’t go deeper, go higher." Proceedings of the 15th ACM Conference on Recommender Systems. 2021.

github.com/hasteck/Higher_Recsys_2021


---

## 🚀 EASER 사용 방법

### 1. 데이터 전처리
데이터를 전처리하려면 아래 명령어를 실행하세요:

```bash
python preprocessing.py
```

### 2. 모델 학습 및 추론
모델 학습 또는 추론을 수행하려면 아래 명령어를 실행하세요:

```bash
python run.py
```

---

## 📂 디렉토리 구조

```
프로젝트/
├── preprocessing.py   # 데이터 전처리 스크립트
├── model.py             # 모델 코드
├── run.py             # 모델 학습 및 추론 스크립트
└── README.md        
```


---

## ❓ FAQ

**Q1. 데이터 전처리 결과는 어디에 저장되나요?**  
A1. `preprocessing.py` 실행 후, 결과는 지정된 "../../saved" 디렉토리에 저장됩니다. 자세한 정보는 스크립트 내부를 참고하세요.

**Q2. 모델 학습 시 필요한 설정은 어디에서 변경할 수 있나요?**  
A2. `run.py` 내부의 설정 파일 또는 명령줄 인자를 통해 변경할 수 있습니다.
