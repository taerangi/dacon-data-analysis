# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import warnings
from scipy.interpolate import make_interp_spline

# Settings
warnings.filterwarnings(action='ignore')


def pie_plot(data, target_column, target_index, colors, labels, unit, f_name) :
    """_summary_

    Args:
        data (pd.DataFrame): pie 차트를 그리는데 사용될 소스 데이터
        target_column (str): ratio 값을 구할 때 사용할 column명
        target_index (int): text_num 값을 구할 떄 사용될 index 값
        colors (list): 그래프를 그릴 떄 사용될 색상들
        labels (list): 그래프를 그릴 때 표기 될 label 값들
        unit (str): 그래프에 표기될 변동치의 단위
        f_name (str): 저장할 파일명
    """
    ratio = np.around(data[target_column].value_counts().values / len(data) * 100, 1)
    wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 5}
    textprops={'fontsize': 15, 'color':'black', 'weight':'bold'}
    if target_column=='promotion_score' :
        text_num = ratio[0] - ratio[2]
    else :
        text_num = np.around(ratio[target_index],1)
    num_comparison = 41.3
    delta = np.around(text_num - num_comparison, 1)
    if delta > 0 :
        arrow = '▲'
        delta_color = 'green'
    else :
        arrow = '▼'
        delta_color = 'red'

    ## 한글 폰트
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    ## Plot
    plt.figure(figsize=(6,6))
    plt.pie(ratio,
        labels=labels,
        autopct='',
        startangle=90, 
        counterclock=False, 
        colors=colors, 
        wedgeprops=wedgeprops, 
        labeldistance=1.1,
        textprops=textprops
    )
    plt.text(-0.22, 0, f'{text_num}%', fontdict={'size':20, 'weight':'bold'})
    plt.text(-0.2, -0.15, f'{arrow} {delta}{unit}', fontdict={'size':12, 'weight':'bold', 'color':delta_color})

    ## Save
    plt.savefig(f_name, dpi=300, transparent=True)
    print(f'{f_name} 저장 완료')


def classify_promotion(x) :
    """_summary_

    Args:
        x (int): promotion score
    """
    if 0 <= x <= 6 :
        return 'Detractors'
    if 7 <= x <= 8 :
        return 'Passives'
    if 9 <= x <= 10 :
        return "Promoters"


def retention_plot(data, f_name='performance_retention.png') :
    """_summary_

    Args:
        data (pd.DataFrame): '대조군', '데이콘 Basic'
        f_name (str, optional): 저장할 파일명
    """
    x = data.index
    colors = ['#4c7bd7', '#ea6a56']
    columns = data.columns

    ## 한글 폰트
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8,6))
    for column, color in zip(columns, colors) :
        y = data[column]
        ## 스플라인 보간법으로 새로운 x, y 데이터 생성
        x_smooth = np.linspace(x.min(), x.max(), 500)
        spl = make_interp_spline(x, y, k=3)
        y_smooth = spl(x_smooth)
        ## 그래프 그리기
        plt.plot(x_smooth, y_smooth, label=column, linewidth=5, color=color, alpha=0.8)
    plt.legend(fontsize=15)
    plt.xlabel('Time', fontsize=15)
    plt.ylabel('Retention', fontsize=15)

    ## Save
    plt.savefig(f_name, dpi=300, transparent=True)
    print(f'{f_name} 저장 완료')


if __name__ == '__main__' :
    cohort_data = pd.read_csv('performance_cohort.csv')
    pie_plot(
        data=cohort_data, 
        target_column='cohort', 
        target_index=2,
        colors=['#28282B', 'grey', '#4c7bd7'],
        num_comparison = 41.3,
        labels = ['cohort_1', 'cohort_2', 'cohort_3\n(초급자)'],
        unit='%p',
        f_name='performance_cohort.png'
    )

    difficulty_data = pd.read_csv('performance_difficulty.csv')
    pie_plot(
        data=cohort_data, 
        target_column='difficulty', 
        target_index=1,
        colors=['grey', '#4c7bd7', '#28282B'],
        num_comparison = 57.2,
        labels = ['쉽다', '적당하다', '어렵다'],
        unit='%p',
        f_name='performance_difficulty.png'
    )

    nps_data = pd.read_csv('performance_nps.csv')
    nps_data['promotion_score'] = nps_data['promotion_score'].apply(lambda x: classify_promotion(x))
    pie_plot(
        data=nps_data, 
        target_column='promotion_score', 
        target_index=0,
        colors=['#4c7bd7', 'grey', '#ea6a56'],
        num_comparison = 36,
        labels = ['Promoters', 'Passives', 'Detractors'],
        unit='',
        f_name='performance_nps.png'
    )

    retention_data = pd.read_csv('performance_retention.csv')
    retention_plot(retention_data)
