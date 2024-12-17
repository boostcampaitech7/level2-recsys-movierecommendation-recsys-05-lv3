## 🚀 Light GCN 사용 방법

### 1. 데이터 전처리
데이터를 전처리하려면 아래 명령어를 실행하세요:

```bash
python preprocess.py
```

### 2. 모델 학습 및 추론
모델 학습 또는 추론을 수행하려면 아래 명령어를 실행하세요:

```bash
python main.py
```

---

## 📂 디렉토리 구조

```
프로젝트/
├── preprocess.py       # 데이터 전처리 스크립트
├── model.py            # 모델 코드
├── recommendation.py   # output 생성 코드
├── train.py            # 모델 학습 코드
├── utils.py            # 각종 기능
├── main.py             # 모델 학습 및 추론 스크립트
└── README.md        
```


---

## ❓ FAQ

**Q1. 데이터 전처리 결과는 어디에 저장되나요?**  
A1. `preprocess.py` 실행 후, 결과는 지정된 {output_path} 디렉토리에 저장됩니다. 자세한 정보는 스크립트 내부를 참고하세요.

**Q2. 모델 학습 시 필요한 설정은 어디에서 변경할 수 있나요?**  
A2. `main.py` 내부의 설정 파일을 통해 변경할 수 있습니다.