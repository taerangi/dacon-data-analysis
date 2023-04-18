# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from konlpy.tag import Okt
import networkx as nx
from matplotlib import rc
from itertools import combinations
from tqdm import tqdm
import warnings
import pickle
import argparse

# Settings
warnings.filterwarnings(action='ignore')


class KeywordNetwork :
    """
    Useage:
        kn = KeywordNetwork(texts)
        kn.fit()
        kn.plot(ax_n=ax[0][1], color='#4169E1')
    """
    def __init__(self, texts, node_pos) :
        self.texts = texts # list or pd.Series or np.array
        self.graph = nx.Graph(), 
        self.node_df = pd.DataFrame(columns=['node', 'x', 'y', 'size']),
        self.relation_df = pd.DataFrame(columns=['from', 'to', 'relation']),
        self.node_pos = node_pos, # nx.spring_layout(nx.Graph())
        if isinstance(self.node_pos, tuple) :
            self.node_pos = self.node_pos[0]


    def preprocess_nouns(self, nouns) :
        """_summary_
        fit에서 명사 전처리에 활용되는 함수

        Args:
            nouns (list): 추출한 명사들이 담긴 리스트

        Returns:
            list: 전처리 완료한 명사들을 담은 리스트
        """
        stop_words = ['끼리', '로서', '통해', '생기', '때문', '얻지', '전반', '별로', '경우', '하나', '던데', '조금', '매우', '대한', '공유', '참고', '더욱', '자체']
        nouns = [n for n in nouns if len(n) > 1] # 단어의 길이가 1개인 것은 제외
        nouns = [noun for noun in nouns if noun not in stop_words] # 불용어 제외

        return nouns
    

    def fit(self, threshold=3) :
        """_summary_
        설문 응답 데이터를 분석하여 키워드 네트워크를 그리는 데에 필요한 소스 값들을 업데이트하는 함수.
        (graph, node_df, relation_df, node_pos)

        
        Args:
            responses (pd.Series): 설문 응답 내용
            threshold (int): 특정 횟수 이상 나온 단어만 남길 때 기준이 되는 값
        """
        okt = Okt()
        count_dict = Counter()
        
        ## tuple(empty dataframe) --> pd.DataFrame
        self.relation_df = self.relation_df[0]
        self.node_df = self.node_df[0]
        
        for words in tqdm(self.texts) :
            ## 출현 빈도 (count_dict)
            normalized_words = okt.normalize(words) # text normalization
            nouns = okt.nouns(normalized_words) # 명사만 추출
            nouns = self.preprocess_nouns(nouns)
            count_dict += Counter(nouns)

            ## 동시 출현 빈도 (relation_df)
            for comb in combinations(set(nouns), 2) : # 중복 제거 후 단어 2개씩 조합 뽑기 (순서 고려 X)
                comb = sorted(comb)
                index = np.where((self.relation_df['from']==comb[0]) & (self.relation_df['to']==comb[1]))[0]
                ## 값이 없을 경우 row 새로 추가
                if len(index)==0 : 
                    new_row = {'from': comb[0], 'to': comb[1], 'relation':1}
                    self.relation_df = self.relation_df.append(new_row, ignore_index=True)
                ## 기존에 값이 있을 경우 +1
                else : 
                    self.relation_df.loc[index, 'relation'] += 1

        ## 특정 횟수 이상 나온 단어만 남기기 (count_dict, relation_df)
        count_dict = {key: value for key, value in dict(count_dict).items() if value >= threshold}
        for i in self.relation_df.index :
            temp = self.relation_df.loc[i]
            if (temp['from'] not in count_dict.keys()) or (temp['to'] not in count_dict.keys()) :
                self.relation_df = self.relation_df.drop(i)
        self.relation_df = self.relation_df.reset_index(drop=True)

        ## graph
        self.graph = nx.from_pandas_edgelist(self.relation_df, 'from', 'to', create_using=nx.Graph())
        
        ## node_pos & count_dict --> node_df 
        for node, pos in self.node_pos.items() :
            new_row = {'node': node, 'x': pos[0], 'y': pos[1], 'size': float(count_dict[node])}
            self.node_df = self.node_df.append(new_row, ignore_index=True)
    

    def adjust_label_pos(self, size) :
        """_summary_
        node size에 비례하여 label의 y 좌표값을 조정해주는 함수

        Args:
            size (_type_): _description_
        """
        ## 세 개의 데이터 포인트
        x = np.array([2, 16, 45])
        y = np.array([0.035, 0.09, 0.27])

        ## 계수를 찾기 위한 행렬 연산
        A = np.vstack([x**2, x, np.ones(len(x))]).T
        a, b, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return a*(size**2) + b*size + c


    def plot(self, ax_n, color) :
        """_summary_
        Keword Network를 시각화하는 함수
        
        Args:
            ax_n (matplotlib.axes._axes.Axes): subplot에서 그래프를 그릴 위치
            color (str): 그래프 색
        """
        nx.draw(self.graph, 
            node_color=color, 
            node_size=self.node_df['size']**2.3 + 5,
            width=self.relation_df['relation']*0.3,
            edgecolors='white',
            linewidths=1.3,
            edge_color=color,
            alpha=0.8, #transparency
            pos=self.node_pos,
            ax=ax_n
        )

        label_pos = {}
        for node, x, y, size in self.node_df.values :
            # if size > ? : + more
            label_pos[node] = (x, y+self.adjust_label_pos(size))
        nx.draw_networkx_labels(self.graph, 
            ax=ax_n,
            pos=label_pos, 
            font_family='AppleGothic',
            font_size=12,
            font_weight='bold'
        )


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


def pie_plot(promotion, ax_n, color) :
    """_summary_

    Args:
        promotion (pd.Series): Promoters, Passives, Detractors
        ax_n (matplotlib.axes._axes.Axes): subplot에서 그래프를 그릴 위치
        color (str): 차트 내의 Promoters의 색깔
    """
    ratio = np.around(promotion.value_counts().values / len(promotion) * 100, 1)
    score = ratio[0] - ratio[2]
    labels = promotion.value_counts().keys()
    colors = [color, 'grey', '#28282B'] 
    wedgeprops={'width': 0.6, 'edgecolor': 'w', 'linewidth': 5}
    textprops={'fontsize': 11, 'color':'black', 'weight':'bold'}

    ax_n.text(-0.2, -0.05, score, fontdict={'size':16, 'weight':'bold'})
    wedges, texts, autotexts = ax_n.pie(ratio, 
        labels=labels, 
        autopct='%.1f%%', 
        startangle=90, 
        counterclock=False, 
        colors=colors, 
        wedgeprops=wedgeprops, 
        pctdistance=0.7,
        labeldistance=1.1,
        textprops=textprops
    )

    ## autopct 글자 색상 변경
    for autotext in autotexts:
        autotext.set_color('white')


def main(f_name, figsize=(20,14)) :
    """_summary_

    Args:
        f_name (str): 저장할 파일명
        figsize (tuple): 전체 그림 크기
    """
    ## 한글 폰트
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    ## data load
    data = pd.read_csv('nps_source.csv')
    low_data = data[data['level']=='low']
    high_data = data[data['level']=='high']
    with open('low_node_pos.pickle', 'rb') as f:
        low_node_pos = pickle.load(f)
    with open('high_node_pos.pickle', 'rb') as f:
        high_node_pos = pickle.load(f)

    ## 밑그림
    fig, ax = plt.subplots(2, 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 3]})
    fig.text(0.215, 0.08, 'NPS', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.65, 0.08, 'Keyword Network', ha='center', fontsize=14, fontweight='bold')
    fig.text(0.05, 0.705, '초급자', va='center', fontsize=14, fontweight='bold')
    fig.text(0.05, 0.285, '숙련자', va='center', fontsize=14, fontweight='bold')

    ## low nps
    low_promotion = low_data['promotion_score'].apply(lambda x: classify_promotion(x))
    pie_plot(low_promotion, ax_n=ax[0][0], color='#4169E1')

    ## high nps
    high_promotion = high_data['promotion_score'].apply(lambda x: classify_promotion(x))
    pie_plot(high_promotion, ax_n=ax[1][0], color='#DB7093')

    ## low keyword network
    kn_low = KeywordNetwork(low_data['text'], node_pos=low_node_pos)
    kn_low.fit()
    kn_low.plot(ax_n=ax[0][1], color='#4169E1')

    ## high keyword network
    kn_high = KeywordNetwork(high_data['text'], node_pos=high_node_pos)
    kn_high.fit()
    kn_high.plot(ax_n=ax[1][1], color='#DB7093')

    plt.savefig(f_name, dpi=200, facecolor='#eeeeee')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='NPS Survey Analysis')
    parser.add_argument('--f_name', type=str, default='nps_analysis.png',
                        help='저장할 파일명')
    args = parser.parse_args()

    main(args.f_name)

    print(f'{args.f_name}에 저장되었습니다.')
