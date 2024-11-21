
---

## 🚀 ensemble 사용 방법

### hard voting

voting을 원하는 모델들의 예측 결과 파일들을 predictions/에 담아주세요.
이후 아래 명령을 실행하세요.

```bash
python hard_voting.py
```

최종 예측 출력은 finals에 저장됩니다.


## 📂 디렉토리 구조

```
프로젝트/
├── finals/   # 최종 예측 파일이 담기는 곳
├── predictions/     # 개별 모델들의 예측 결과를 담아두는 곳
├── hard_voting.py       # 하드 보팅 수행하기 위한 파일
└── README.md       
```

