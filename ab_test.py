## Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.stats import kstest, norm, binom, t, chi2_contingency

## Settings
warnings.filterwarnings('ignore')


class ABTest() :
    def __init__(self, data) :
        """_summary_

        Args:
            data (pd.Series): AB Test에 사용될 소스 데이터
        """
        self.data = data
        self.var_dict = {'A':{}, 'B':{}}
        for group in ['A', 'B'] :
            group_data = data[group]
            n = group_data.sum()
            p = group_data[1]/n
            mean = n*p
            variance = n*p*(1-p)
            std_dev = np.sqrt(variance)
            x = np.arange(binom.ppf(0.01, n, p), binom.ppf(0.99, n, p))

            self.var_dict[group]['n'] = n
            self.var_dict[group]['p'] = p
            self.var_dict[group]['mean'] = mean
            self.var_dict[group]['variance'] = variance
            self.var_dict[group]['std_dev'] = std_dev
            self.var_dict[group]['x'] = x
            

    def normality_test(self, group, color, alpha=0.05) :
        """_summary_
        Kolmogorov-Smirnov Test

        Args:
            group (str): A or B
            color (str): 이항분포 CDF 그래프의 색
            alpha (float, optional): 유의수준 Defaults to 0.05.
        """
        ## 가설 수립
        h_0 = f'그룹 {group}의 데이터셋이 정규분포를 따른다.' # 귀무가설
        h_1 = f'그룹 {group}의 데이터셋이 정규분포를 따르지 않는다.' # 대립가설

        g_dict = self.var_dict[group]

        ## 변수 세팅 (이항분포)
        cdf_binom = binom.cdf(g_dict['x'], g_dict['n'], g_dict['p']) # 이항분포 CDF
        cdf_norm = norm.cdf(g_dict['x'], g_dict['mean'], g_dict['std_dev']) # 정규분포 CDF
        f_name = f'normality_{group}.png'

        ## Kolmogorov-Smirnov Test
        D, p_value = kstest(cdf_binom, cdf_norm)

        ## 그래프 그리기
        plt.figure(figsize=(8,6))
        plt.plot(x, cdf_binom, 'X', color=color, markersize=9, label='Binomial CDF')
        plt.plot(x, cdf_norm, '-', color='red', lw=4, label='Normal CDF')
        plt.legend(fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.savefig(f_name, dpi=300)
        print(f'{f_name} 저장 완료')

        ## 결과 출력
        print(f'{group}그룹 데이터셋에 대해 Normality Test를 수행합니다.')
        print(f'H0(귀무가설): {h_0}')
        print(f'H1(귀무가설): {h_1}')
        print('\n')
        print(f'D: {D}')
        print(f'p_value: {p_value}')
        if p_value > alpha:
            print(f'귀무가설({h_0})을 기각하지 않는다.')
        else :
            print(f'귀무가설({h_0})을 기각하고, 대립가설({h_1})을 채택한다.')


    def t_test(self, alpha=0.05) :
        """_summary_
        이분산 독립표본, 단측 검정
        Args:
            alpha (float, optional): 유의수준 Defaults to 0.05.
        """
        ## 가설 수립
        h_0 = '커뮤니티 랜딩페이지(B)와 기존 랜딩페이지(A)의 대회 참가 전환율의 평균은 동일하다. (CR_A = CR_B)' # 귀무가설
        h_1 = '커뮤니티 랜딩페이지(B)가 기존 랜딩페이지(A)보다 대회 참가 전환율의 평균이 작다. (CR_A > CR_B)' # 대립가설

        ## 변수 세팅
        a = self.var_dict['A']
        b = self.var_dict['B']
        distance = a['mean'] - b['mean']
        se = np.sqrt(a['std_dev']**2 + b['std_dev']**2) # standard error
        t_value = distance/se
        dof = a['n'] + b['n'] - 2  # 자유도
        t_critical = t.ppf(1-alpha, dof)
        p_value = 1 - t.cdf(t_value, dof)
        threshold = data['B'][1] + (t_critical*b['std_dev'])
        threshold_index = np.where(a['x']>threshold)[0][0]
        pmf_binom_a = binom.pmf(a['x'], a['n'], a['p'])
        pmf_binom_b = binom.pmf(b['x'], b['n'], b['p'])
        mean_cr_a = a['mean']/a['n']*100
        mean_cr_b = b['mean']/b['n']*100
        threshold_percent = threshold/a['n']*100
        fill_x = a['x'][threshold_index:]
        fill_y = pmf_binom_a[threshold_index:]
        f_name = 't_test.png'

        ## 그래프 그리기
        plt.plot(a['x']/a['n']*100, pmf_binom_a, color='#4c7bd7', linewidth=3, alpha=1, label='A') #'#4c7bd7', 'grey', '#ea6a56'
        plt.plot(b['x']/b['n']*100, pmf_binom_b, color='#ea6a56', linewidth=3, alpha=1, label='B')
        plt.axvline(x=mean_cr_a, color='lightgray', linewidth=1)
        plt.axvline(x=mean_cr_b, color='lightgray', linewidth=1)
        plt.axvline(x=threshold_percent, color='lightgray', linewidth=1)
        plt.fill_between(fill_x/a['n']*100, min(pmf_binom_b), fill_y, alpha=0.4, color='skyblue')
        plt.text(mean_cr_a+0.04, max(pmf_binom_b)/2, f'CR A: {np.round(mean_cr_a, 2)}%', fontweight='bold', color='gray', rotation=90, va='center', ha='center')
        plt.text(mean_cr_b+0.04, max(pmf_binom_b)/2, f'CR B: {np.round(mean_cr_b, 2)}%', fontweight='bold', color='gray', rotation=90, va='center', ha='center')
        plt.text(threshold_percent+0.04, max(pmf_binom_b)/2, 'Confidence Level 95%', fontweight='bold', color='gray', rotation=90, va='center', ha='center')
        plt.legend()
        xticks = plt.xticks()[0][1:-1]
        plt.xticks(xticks, list(map(lambda x: str(x) + '%', xticks)))
        plt.xlabel('Conversion Rate', labelpad=15)
        plt.ylabel('Probability', labelpad=15)
        plt.tight_layout()
        plt.savefig('t_test.png', dpi=300)
        print(f'{f_name} 저장 완료')

        ## 결과 출력
        print('t-test를 수행합니다.')
        print(f'H0(귀무가설): {h_0}')
        print(f'H1(귀무가설): {h_1}')
        print('\n')
        print(f't_value: {t_value}')
        print(f'p_value: {p_value}')
        if p_value > alpha:
            print(f'귀무가설({h_0})을 기각하지 않는다.')
        else :
            print(f'귀무가설({h_0})을 기각하고, 대립가설({h_1})을 채택한다.')


    def chi_squared_test(self, alpha=0.05) :
        """_summary_
        이원 독립성 검정

        Args:
            alpha (float, optional): _description_. Defaults to 0.05.
        """
        ## 가설 수립
        h_0 = '랜딩페이지와 대회 참가 전환 간에는 연관성이 없다. (상호 독립이다.)' # 귀무가설
        h_1 = '랜딩페이지와 대회 참가 전환 간에는 연관성이 있다.' # 대립가설
        
        ## 데이터 입력
        obs = (self.data['A'].sort_index(ascending=False), self.data['B'].sort_index(ascending=False))

        ## 카이제곱 검정 수행
        chi2, p_value, dof, expected = chi2_contingency(obs)

        ## 결과 출력
        print('t-test를 수행합니다.')
        print(f'H0(귀무가설): {h_0}')
        print(f'H1(귀무가설): {h_1}')
        print('\n')
        print(f'Chi-squared statistic: {chi2}')
        print(f'p_value: {p_value}')
        if p_value > alpha:
            print(f'귀무가설({h_0})을 기각하지 않는다.')
        else :
            print(f'귀무가설({h_0})을 기각하고, 대립가설({h_1})을 채택한다.')


if __name__ == '__main__' :
    data = pd.read_csv('ab_test.csv')
    
    ## 인스턴스 생성
    ab_test = ABTest(data)
    
    ## Noramlity Test
    ab_test.normality_test('A', '#52BDFF') # '#0080FF'
    ab_test.normality_test('B', '#4CBB17') # '#28AE89'

    ## t-test
    ab_test.t_test()

    ## Chi-squared Test
    ab_test.chi_squared_test()

