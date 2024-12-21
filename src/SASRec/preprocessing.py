import os

import pandas as pd


def item2attributes(args):
    """
    모델에 사용할 아이템 속성 데이터를 생성하는 함수입니다.

    이 함수는 주어진 데이터셋 경로에서 장르, 감독, 작가, 연도 등의 정보를 불러와서 전처리하고,
    이를 결합하여 최종적으로 아이템에 대한 속성 정보를 생성합니다. 생성된 속성 정보는 
    'genre', 'new_year', 'director', 'writer'와 같은 항목들을 포함하며, 중복된 요소가 
    있는지 확인한 후, 최종적으로 JSON 파일로 저장됩니다.
    """
    
    data_path = args.dataset.data_path
    output_dir = f'{args.dataset.preprocessing_path}/SAS/'

    genres_df = pd.read_csv(f"{data_path}/genres.tsv", sep="\t")
    directors_df = pd.read_csv(f'{data_path}/directors.tsv', sep='\t')
    years_df = pd.read_csv(f"{data_path}/years.tsv", sep="\t")
    writers_df = pd.read_csv(f'{data_path}/writers.tsv', sep='\t')

    # 연도 데이터 전처리
    min_year = years_df['year'].min()
    years_df['decade_group'] = (years_df['year'] - min_year) // 10
    years_df['new_year'] = years_df['year'].apply(lambda x: 0 if x < 2004 else x - 2003)

    # 장르 데이터 전처리
    genres_df['genre'], _ = pd.factorize(genres_df['genre'])
    genres_grouped = genres_df.groupby("item")['genre'].apply(list).reset_index()

    # 연도 데이터와 병합
    tmp = pd.merge(genres_grouped, years_df[['item', 'new_year']], on='item', how='left')
    tmp['new_year'], _ = pd.factorize(tmp['new_year'])
    tmp['new_year'] += max(genres_df['genre']) + 1

    # 감독 데이터 전처리
    directors_df['director'], _ = pd.factorize(directors_df['director'])
    directors_df['director'] += max(tmp['new_year']) + 1
    directors_max = max(directors_df['director'])

    # 감독 데이터와 병합
    tmp = pd.merge(tmp, directors_df.groupby('item')['director'].apply(list), on='item', how='left')
    tmp['director'] = tmp['director'].apply(lambda x: x if isinstance(x, list) else [directors_max + 1 ])

    # 작가 데이터 전처리
    writers_df['writer'], _ = pd.factorize(writers_df['writer'])
    writers_df['writer'] += directors_max + 1 + 1 
    writers_max = max(writers_df['writer'])

    # 작가 데이터와 병합
    tmp = pd.merge(tmp, writers_df.groupby('item')['writer'].apply(list), on='item', how='left')
    tmp['writer'] = tmp['writer'].apply(lambda x: x if isinstance(x, list) else [writers_max + 1 + 1 + 1 ])

    # 최종 결합
    tmp['combined'] = tmp.apply(
        lambda row: row['genre'] + [row['new_year']] +
                    row['director'] + row['writer'],
        axis=1
    )

    # 중복 요소 검증
    if not tmp[tmp['combined'].apply(lambda x: len(x) != len(set(x)))].empty:
        print('Error: Duplicate elements found in combined list.')
    else:
        print('Generate Data' + '-' * 89)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # 최종 JSON 저장
    tmp.set_index('item')['combined'].to_json(f"{output_dir}/Ml_item2attributes.json")