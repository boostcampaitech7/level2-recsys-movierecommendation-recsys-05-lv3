## 🚀 ensemble 사용 방법

### hard voting

'../../config/model_weights.yaml'을 수정하여 앙상블 output 파일을 생성합니다.<br><br>
'id'의 value를 수정하여 파일 저장명을 설정할 수 있습니다. ('final_recommendations_{id}.csv'로 저장됩니다.)<br><br>
'output_dir'의 value를 수정하여 파일 저장 경로를 설정할 수 있습니다. <br><br>
'model_weights' 에는 각 모델명이 value로 존재합니다.<br>
'model_weights' 내부에서 모델의 순서를 바꿔 같은 가중치일 때의 우선순위를 설정할 수 있습니다.<br>
위쪽에 위치할수록 더 높은 우선순위를 갖습니다.<br>
각 모델 내부의 key의 value들을 수정할 수 있습니다.<br>
- 'use'의 value를 수정하여 해당 모델의 output의 사용여부를 설정합니다.<br>
- 'file'의 value를 수정하여 해당 모델의 파일명을 설정합니다.<br>
- 'weight'의 value를 수정하여 해당 모델의 가중치를 설정합니다.<br>


아래 명령을 실행하세요.

```bash
python hard_voting.py
```


최종 예측 출력(.csv)과 해당 예측에 사용된 가중치 정보(.log)는 'output_dir'에 저장됩니다.
```
output_dir/
├── final_recommendations_{id}.csv              # 최종 예측 파일
└── final_recommendations_{experiment_id}.log   # 예측할 때 사용된 파일별 가중치 정보    
```