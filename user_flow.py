## Import Library & Settings
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


class user_flow :
    def __init__(self, data, cohort) :
        """_summary_

        Args:
            data (pd.DataFrame): source, target, value, index_s, index_t
        """
        self.data = data
        self.cohort = cohort
        self.barh_data = {'conversion': pd.DataFrame(), 'exit': pd.DataFrame()}


    def append_y(self, y_list, total_first_col, start, end, index_type) :
        """_summary_
        sankey_plot에서 node의 위치를 그릴 때 사용될 y값을 계산하는 함수

        Useage:
            e.g.
            y_list = append_y(y_list, start, end, 'index_s')


        Args:
            y_list (list): y값을 추가할 리스트
            start (int): 그래프 상에서 동일한 열에 놓일 노드들의 index start
            end (int): 그래프 상에서 동일한 열에 놓일 노드들의 index end
            index_type (str): 'index_s' or 'index_t'

        Returns:
            (list)
        """
        query = (self.data[index_type]>=start) & (self.data[index_type]<=end)
        total = self.data[query]['value'].sum()
        pre_value = -0.1 + (1 - total/total_first_col)/2 
        pre_half_len = 0

        for i in range(start, end+1): 
            node_count = self.data[self.data[index_type]==i]['value'].sum()
            half_len = (node_count/total_first_col)/2
            value = pre_value + pre_half_len + half_len + 0.03
            y_list.append(value)
            pre_half_len = half_len
            pre_value = value
        
        return y_list


    def sankey_plot(self, width=800, height=500) :
        """_summary_

        Args:
            f_name (str): 저장할 파일명
            width (int, optional): 저장할 이미지의 너비
            height (int, optional): 저장할 이미지의 높이
        """
        label_list = [
            'Contest Platform', 'Google AD', 'Naver AD', 'SNS AD', 'etc', # 0, 1, 2, 3, 4 (Acquisition)
            'View_Easy', 'View_Moderate', 'View_Difficult', # 5, 6, 7 (View Competition)
            'Join_Easy', 'Join_Moderate', 'Join_Difficult', # 8, 9, 10 (Join Competition)
            'Rejoin_Easy', 'Rejoin_Moderate', 'Rejoin_Difficult', # 11, 12, 13 (Rejoin Competition)
        ]

        ## Get y_list (node의 y 좌표)
        total_first_col = self.data[self.data['index_s']<=4]['value'].sum()
        y_list = []
        y_list = self.append_y(y_list, total_first_col, 0, 4, 'index_s')
        node_ranges = [(5, 7), (8, 10), (11, 13)]
        for start, end in node_ranges :
            y_list = self.append_y(y_list, total_first_col, start, end, 'index_t')

        ## Sankey Diagram 객체 생성
        fig_sankey = go.Sankey(
            node = dict(
                pad = 20,
                thickness = 20,
                label = label_list,
                x = [0.01,0.01,0.01,0.01,0.01,0.33,0.33,0.33,0.66,0.66,0.66,1,1,1],
                y = y_list,
                # color = ["green", "orange"] * 4
            ),
            link = dict(
                ## source[n] -> target[n] = value[n]
                source = self.data['index_s'],
                target = self.data['index_t'],
                value = self.data['value'],
                color = '#e6e8ed'
            )
        )

        ## Plot
        fig = go.Figure()
        fig.add_trace(fig_sankey)
        fig.update_layout({
            'margin': {
                    't': 30,
                    'b': 30,
                    'l': 50,
                    'r': 50
            },
            'font': {'size':12},
            'width': width,
            'height': height
        })

        ## Save as image file
        f_name = f'sankey_{self.cohort}.png'
        fig.write_image(f_name)
        print(f'User FLow - Sankey Diagram: "{f_name}" 저장 완료.')


    def update_conversion_data(self, threshold=4) :
        """_summary_
        conversion의 barh_plot에 사용될 테이터(barh_data['conversion'])를 업데이트하는 함수

        Args:
            threshold (int, optional): Defaults to 4.
        """
        total = self.data[self.data['index_s'] <= threshold]['value'].sum()

        conversion_df = pd.DataFrame(index=['Join', 'Rejoin'], data=0, columns=['Easy', 'Moderate', 'Difficult'])
        for step in conversion_df.index :
            for difficulty in conversion_df.columns :
                conversion_count = self.data[self.data['target']==f'{step}_{difficulty}']['value'].sum()
                conversion_df.loc[step, difficulty] = conversion_count/total*100

        self.barh_data['conversion'] = conversion_df
    

    def update_exit_data(self) :
        """_summary_
        conversion의 barh_plot에 사용될 테이터(barh_data['exit'])를 업데이트하는 함수
        """
        exit_df = pd.DataFrame(index=['View', 'Join'], data=0, columns=['Easy', 'Moderate', 'Difficult'])

        for step in exit_df.index :
            for difficulty in exit_df.columns :
                node_count = self.data[self.data['target']==f'{step}_{difficulty}']['value'].sum()
                link_count = self.data[self.data['source']==f'{step}_{difficulty}']['value'].sum()
                exit_df.loc[step, difficulty] = np.around((1-link_count/node_count)*100, 1)

        exit_df.index = ['Join', 'Rejoin']
        
        self.barh_data['exit'] = exit_df


    def compute_pos(self, step, height, i, models):
        """_summary_
        barh_plot에 사용되는 함수로 plt.barh() 함수의 y 값을 계산

        Args:
            step (np.array): 
            height (float): 
            i (int): 
            difficulty (np.array): 

        Returns:
            [float, float]
        """
        index = np.arange(len(step)) 
        n = len(models) # 3
        correction = i - 0.5*(n-1)

        return index + height * correction


    def present_width(self, ax, bar):
        """_summary_
        barh_plot에 사용되는 함수로 barh의 너비를 출력
        
        Args:
            ax (matplotlib.axes._axes.Axes): 
            bar (matplotlib.container.BarContainer): _description_
        """
        for rect in bar:
            witdh = rect.get_width()
            posx = witdh*1.01
            posy = rect.get_y()+rect.get_height()*0.5
            ax.text(posx, posy, '%.1f' % witdh + '%', rotation=0, ha='left', va='center')
    

    def barh_plot(self, df_type) :
        """_summary_
        Competition 난이도에 따른 단계별 전환율, 이탈률을 수평 막대 그래프로 시각화하는 함수
        
        Args:
            barh_df (pd.DataFrame): 데이터 시각화에 필요한 소스 데이터
            df_type (str): 'conversion' or 'exit'
        """
        ## bar plot으로 나타낼 데이터 입력
        barh_df = self.barh_data[df_type]
        barh_dict = {}
        for c in barh_df.columns :
            barh_dict[c] = list(barh_df[c])
        step = barh_df.index # ['Join', 'Rejoin']

        ## matplotlib의 figure 및 axis 설정
        fig, ax = plt.subplots(1, 1, figsize=(7,5)) 
        colors = ['salmon', 'orange', 'cadetblue'] 
        height = 0.25

        ## bar plot
        for i, difficulty in enumerate(barh_df.columns):
            pos = self.compute_pos(step, height, i, barh_df.columns)
            bar = ax.barh(pos, barh_dict[difficulty], height=height*0.95, label=difficulty, color=colors[i])
            self.present_width(ax, bar) # bar너비 출력

        ## axis 세부설정
        ax.set_xlim([0, 100])
        ax.set_yticks(range(len(step)))
        ax.set_yticklabels(step, fontsize=10)	
        plt.gca().invert_yaxis()
        ax.legend()

        ## save as image file
        f_name = f'barh_{df_type}_{self.cohort}.png'
        plt.savefig(f_name, format='png', dpi=300)
        print(f'User FLow - Barh: "{f_name}" 저장 완료.')


if __name__ == '__main__' :
    for cohort in [1, 2] :
        cohort_data = pd.read_csv(f'user_flow_{cohort}.csv')
        cohort_user_flow = user_flow(cohort_data, cohort)
        
        ## Sankey Plot
        cohort_user_flow.sankey_plot()

        ## Barh Plot - Conversion
        cohort_user_flow.update_conversion_data()
        cohort_user_flow.barh_plot('conversion')

        ## Barh Plot - Exit
        cohort_user_flow.update_exit_data()
        cohort_user_flow.barh_plot('exit')