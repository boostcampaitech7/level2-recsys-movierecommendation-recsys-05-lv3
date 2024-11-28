
---

## 🚀 DeepFM 사용 방법

### 전처리, 모델학습, 추론
전처리부터 모델 학습 그리고 추론을 수행하려면 아래 명령어를 실행하세요:

```bash
python run.py
```

---

## 📂 디렉토리 구조

```
프로젝트/
├── DeepFM.py  # 모듈화 이전의 코드
├── dataset.py   
├── inference.py   # top k개를 추론하는 스크립트
├── model.py             # 모델 코드
├── preprocessing.py   # 데이터 전처리 스크립트
├── run.py             # 모델 학습 및 추론 스크립트
├── train.py             # 모델 학습 및 추론 스크립트
└── README.md        
```


---

## ❓ FAQ

**Q1. 모델 학습 시 필요한 설정은 어디에서 변경할 수 있나요?**  
A1. `run.py` 내부의 코드 또는 명령줄 인자를 통해 변경할 수 있습니다.