
---

## 🚀 ensemble 사용 방법

### hard voting

voting을 원하는 모델들의 예측 결과 파일들을 predictions/에 담아주세요.<br>
파일명 앞에 숫자를 추가해 파일들에 우선순위를 부여하세요.<br>
예시)
```
predictions/
├── 01_A.csv
├── 02_D.csv
├── 03_B.csv
└── 04_C.csv
```
아이템이 같은 추천수를 가질 때 우선순위가 높은 csv에 있는 아이템을 우선적으로 추천합니다.<br>
이후 아래 명령을 실행하세요.

```bash
python hard_voting.py
```

모든 파일에 동일한 가중치를 적용할지를 선택하세요.<br>
```bash
모든 파일에 동일한 가중치를 적용하시겠습니까? (y/n): 
```
서로 다른 가중치를 적용하게 된다면 파일 별 가중치를 입력하게 됩니다.<br>
각 파일에 대한 가중치는 전체 파일에 대해 엽력된 가중치의 합에 대한 비율로 적용됩니다.
```bash
파일: 01_A.csv
01_A.csv의 가중치를 입력하세요:
```

최종 예측 출력(.csv)과 해당 예측에 사용된 가중치 정보(.json)는 finals에 저장됩니다.
```
finals/
├── {timestamp}.csv             # 최종 예측 파일이 담기는 곳
└── {timestamp}_weights.json    # 개별 모델들의 예측 결과를 담아두는 곳     
```

```json
# File: {timestamp}_weights.json

{
    "original_weights": {
        "01_A.csv": 1.0,
        "02_D.csv": 1.0,
        "03_B.csv": 1.0,
        "04_C.csv": 2.0
    },
    "weight_ratios": {
        "01_A.csv": 20.0,
        "02_D.csv": 20.0,
        "03_B.csv": 20.0,
        "04_C.csv": 40.0
    }
}
```


## 📂 디렉토리 구조

```
ensemble/
├── finals/   # 최종 예측 파일이 담기는 곳
├── predictions/     # 개별 모델들의 예측 결과를 담아두는 곳
├── hard_voting.py       # 하드 보팅 수행하기 위한 파일
└── README.md       
```

