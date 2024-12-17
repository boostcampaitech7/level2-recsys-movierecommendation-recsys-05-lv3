import os
import pandas as pd

class Preprocessing:
    def __init__(self, data_dir, output_dir, threshold=None, min_items_per_user=1, min_users_per_item=0):
        """
        데이터셋을 로드하고, 최소 사용자 수 및 최소 아이템 수 조건에 맞는 데이터를 필터링하는 클래스.

        Args:
            data_dir (str): 원본 데이터셋이 위치한 디렉토리 경로.
            output_dir (str): 처리된 데이터를 저장할 출력 디렉토리 경로.
            threshold (float, optional): 평점 threshold (현재는 사용되지 않음).
            min_items_per_user (int): 사용자가 평가한 최소 아이템 수. 기본값은 1.
            min_users_per_item (int): 아이템이 평가된 최소 사용자 수. 기본값은 0.
        """
        self.dataset = data_dir + 'train_ratings.csv'
        self.output_dir = output_dir
        self.threshold = threshold
        self.min_uc = min_items_per_user
        self.min_sc = min_users_per_item
        self.raw_data = None

    def load_data(self):
        """
        데이터를 로드하여 `self.raw_data`에 저장하는 함수.

        주어진 경로에서 CSV 파일을 읽어오며, 파일이 없으면 예외 처리하여 에러 메시지를 출력합니다.
        """
        try:
            self.raw_data = pd.read_csv(self.dataset, header=0)
        except FileNotFoundError:
            print(f"Dataset file {self.dataset} not found.")

    def get_count(self, tp, id):
        """
        주어진 데이터프레임에서 특정 ID에 대한 카운트를 반환하는 함수.

        Args:
            tp (DataFrame): 데이터를 포함한 pandas DataFrame.
            id (str): 카운트를 세고자 하는 열 이름.

        Returns:
            pandas.Series: 각 ID에 대한 빈도 수.
        """
        playcount_groupbyid = tp.groupby(id).size()
        return playcount_groupbyid

    def filter_triplets(self, tp):
        """
        사용자 및 아이템의 최소 평가 수를 기준으로 데이터를 필터링하는 함수.

        Args:
            tp (DataFrame): 필터링할 데이터.

        Returns:
            tuple: 필터링된 데이터프레임, 사용자 카운트, 아이템 카운트.
        """
        # 아이템의 최소 평가자 수가 설정되어 있으면 필터링
        if self.min_sc > 0:
            itemcount = self.get_count(tp, 'item')
            tp = tp[tp['item'].isin(itemcount.index[itemcount >= self.min_sc])]
        
        # 사용자의 최소 평가 아이템 수가 설정되어 있으면 필터링
        if self.min_uc > 0:
            usercount = self.get_count(tp, 'user')
            tp = tp[tp['user'].isin(usercount.index[usercount >= self.min_uc])]
        
        # 필터링된 데이터의 사용자 및 아이템별 평가 수
        usercount, itemcount = self.get_count(tp, 'user'), self.get_count(tp, 'item')
        return tp, usercount, itemcount

    def process(self):
        """
        데이터를 처리하여 필요한 형식으로 변환하고, 필터링된 데이터를 출력 디렉토리에 저장하는 함수.

        데이터 로딩, 필터링, 사용자 및 아이템 ID 맵핑, 그리고 처리된 데이터를 파일로 저장하는 과정입니다.
        """
        if self.raw_data is None:
            print("No data loaded. Please check the dataset path.")
            return

        # 데이터 필터링
        self.raw_data, user_activity, item_popularity = self.filter_triplets(self.raw_data)

        # 희소도 계산 (sparsity)
        sparsity = 1. * len(self.raw_data) / (len(user_activity) * len(item_popularity))

        print("After filtering, there are %d interactions from %d users and %d movies (sparsity: %.3f%%)" % 
              (len(self.raw_data), len(user_activity), len(item_popularity), sparsity * 100))

        # 고유 사용자 ID 및 아이템 ID 리스트 생성
        unique_uid = user_activity.index
        unique_sid = pd.unique(self.raw_data['item'])
        
        # 아이템 및 사용자 ID 매핑 생성
        show2id = dict((sid, i) for (i, sid) in enumerate(unique_sid))
        profile2id = dict((pid, i) for (i, pid) in enumerate(unique_uid))

        # 출력 디렉토리 생성
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # unique_sid.txt 및 unique_uid.txt 파일로 저장
        with open(os.path.join(self.output_dir, 'unique_sid.txt'), 'w') as f:
            for sid in unique_sid:
                f.write('%s\n' % sid)
                
        with open(os.path.join(self.output_dir, 'unique_uid.txt'), 'w') as f:
            for uid in unique_uid:
                f.write('%s\n' % uid)

        def numerize(tp):
            """
            사용자 및 아이템 ID를 정수로 변환하는 함수.

            Args:
                tp (DataFrame): 변환할 데이터프레임.

            Returns:
                DataFrame: 정수로 변환된 사용자 및 아이템 ID를 포함하는 데이터프레임.
            """
            uid = list(map(lambda x: profile2id[x], tp['user']))
            sid = list(map(lambda x: show2id[x], tp['item']))
            return pd.DataFrame(data={'uid': uid, 'sid': sid}, columns=['uid', 'sid'])

        # 데이터를 정수로 변환하여 train.csv로 저장
        train_data = numerize(self.raw_data)
        train_data.to_csv(os.path.join(self.output_dir, 'train.csv'), index=False)
