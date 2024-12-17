import os
import pandas as pd

class Preprocessing:
    """
    사용자-아이템 데이터를 숫자로 변환하고, 고유한 사용자 및 아이템 ID를 매핑하여 저장하는 전처리 클래스입니다.
    이 클래스는 주어진 데이터에서 고유한 사용자 및 아이템 ID를 추출하고, 이를 숫자로 변환하여 저장하는 기능을 수행합니다.
    
    Attributes:
        data_dir (str): 원본 데이터가 저장된 디렉토리 경로.
        output_dir (str): 전처리 후 결과를 저장할 디렉토리 경로.
        data (DataFrame): 원본 데이터가 저장된 pandas DataFrame.
        unique_sid (ndarray): 고유한 아이템 ID 목록.
        unique_uid (ndarray): 고유한 사용자 ID 목록.
        show2id (dict): 아이템 ID에서 숫자로 매핑하는 딕셔너리.
        profile2id (dict): 사용자 ID에서 숫자로 매핑하는 딕셔너리.
    """
    
    def __init__(self, data_dir, output_dir):
        """
        클래스 초기화 메서드입니다. 데이터를 로드하고, 전처리 파이프라인을 실행합니다.
        
        Parameters:
            data_dir (str): 원본 데이터가 저장된 디렉토리 경로.
            output_dir (str): 전처리 후 결과를 저장할 디렉토리 경로.
        """
        self.data_dir = data_dir
        self.data = pd.read_csv(self.data_dir + 'train_ratings.csv')  # 원본 데이터 로드
        self.unique_sid = None
        self.unique_uid = None
        self.show2id = None
        self.profile2id = None

        self.output_dir = output_dir

        # 초기화 시 전처리 파이프라인 실행
        self.run()

    def prepare_mappings(self):
        """
        고유한 사용자 및 아이템 ID를 추출하고, 이를 숫자로 매핑하는 딕셔너리를 생성합니다.
        - 아이템 ID를 숫자 인덱스로 매핑하는 `show2id` 생성
        - 사용자 ID를 숫자 인덱스로 매핑하는 `profile2id` 생성
        """
        self.unique_sid = pd.unique(self.data['item'])  # 고유한 아이템 ID 추출
        self.unique_uid = pd.unique(self.data['user'])  # 고유한 사용자 ID 추출
        self.show2id = {sid: i for i, sid in enumerate(self.unique_sid)}  # 아이템 ID -> 숫자 매핑
        self.profile2id = {pid: i for i, pid in enumerate(self.unique_uid)}  # 사용자 ID -> 숫자 매핑

    def save_mappings(self):
        """
        생성된 고유 아이템 ID 및 사용자 ID 매핑을 텍스트 파일로 저장합니다.
        - 'unique_sid.txt': 고유한 아이템 ID 리스트
        - 'unique_uid.txt': 고유한 사용자 ID 리스트
        """
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)  # 출력 디렉토리가 없으면 생성

        # 고유 아이템 ID 저장
        with open(os.path.join(self.output_dir, 'unique_sid.txt'), 'w') as f:
            for sid in self.unique_sid:
                f.write(f'{sid}\n')

        # 고유 사용자 ID 저장
        with open(os.path.join(self.output_dir, 'unique_uid.txt'), 'w') as f:
            for uid in self.unique_uid:
                f.write(f'{uid}\n')

    def numerize(self, data):
        """
        주어진 데이터를 숫자 형식으로 변환합니다. 사용자 ID와 아이템 ID를 각각 매핑하여 숫자로 변환합니다.
        
        Parameters:
            data (DataFrame): 원본 데이터 (사용자-아이템 데이터)
        
        Returns:
            DataFrame: 변환된 사용자 및 아이템 ID 숫자형 데이터
        """
        # 사용자 ID를 숫자형으로 변환
        uid = data['user'].map(self.profile2id)
        # 아이템 ID를 숫자형으로 변환
        sid = data['item'].map(self.show2id)
        # 숫자로 변환된 사용자 및 아이템 ID를 DataFrame으로 반환
        return pd.DataFrame({'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

    def save_numerized_data(self):
        """
        변환된 숫자형 사용자 및 아이템 데이터를 'train.csv' 파일로 저장합니다.
        """
        # 데이터를 숫자형으로 변환
        train_data = self.numerize(self.data)
        # 변환된 데이터를 CSV로 저장
        train_data.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)

    def run(self):
        """
        전체 전처리 파이프라인을 실행하는 메서드입니다. 아래 작업을 순차적으로 수행합니다:
        1. 매핑 준비
        2. 매핑 저장
        3. 데이터 숫자형 변환 및 저장
        
        이 메서드는 클래스 초기화 시 자동으로 실행됩니다.
        """
        print("Preparing mappings...")
        self.prepare_mappings()
        print("Saving mappings...")
        self.save_mappings()
        print("Numerizing and saving data...")
        self.save_numerized_data()
        print("Preprocessing complete.")
