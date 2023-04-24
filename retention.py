## Import Library & Settings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import warnings
import argparse
from scipy.interpolate import make_interp_spline
import argparse

warnings.filterwarnings('ignore')


def preprocess(user_competition, user_cohort) :
    """_summary_
    DB에서 추출한 데이터를 리텐션 분석에 활용될 데이터 형태로 가공하는 함수
    
    Args:
        user_competition (pd.DataFrame): user_id, competition_1, ..., competition_6
        user_cohort (pd.DataFrame): user_id, cohort

    Returns:
        (pd.DataFrame): 리텐션 분석에 활용될 데이터
    """
    data = user_competition.merge(user_cohort, on='user_id', how='left')

    retention_df = pd.DataFrame(columns=['cohort_1', 'cohort_2', 'cohort_3'], index=range(7), data=0)
    retention_df.loc[0] = [100, 100, 100]

    for cohort in [1,2,3] :
        temp = data[data['cohort']==cohort]
        for i in range(1, 7) :
            retention_df.loc[i, f'cohort_{cohort}'] = np.around(temp[f'competition_{i}'].sum() / len(temp) * 100)

    return retention_df


def plot(retention_df, f_name='retention.png') :
    x = retention_df.index
    x_smooth = np.linspace(x.min(), x.max(), 500)
    for y in retention_df.columns :
        # 스플라인 보간법으로 새로운 x, y 데이터 생성
        spl = make_interp_spline(x, retention_df[y], k=3)
        y_smooth = spl(x_smooth)

        plt.plot(x_smooth, y_smooth, label=y)

    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Retention')
    plt.savefig(f_name, dpi=200, facecolor='#ffffff')

    print(f'{args.f_name}에 저장되었습니다.')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='Retention Analysis')
    parser.add_argument('--f_name', type=str, default='retention.png',
                        help='저장할 파일명')
    args = parser.parse_args()

    user_competition = pd.read_csv('user_competition.csv')
    user_cohort = pd.read_csv('user_cohort.csv')

    retention_df = preprocess(user_competition, user_cohort)

    plot(retention_df, args.f_name)
    

    