## Import Library & Settings
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import warnings
import argparse

warnings.filterwarnings('ignore')


def elbowmethod(data, param_init='random', param_n_init=10, param_max_iter=300):
    """_summary_
    Elbow Method를 적용하여 K-means Clustering에 활용될 optimal k를 찾는 함수. (시각화)
    Args:
        data (np.array): pca를 완료한 데이터
        param_init (str, optional): Defaults to 'random'.
        param_n_init (int, optional): Defaults to 10.
        param_max_iter (int, optional): Defaults to 300.
    """
    distortions = []
    for i in range(1, 10):
        km = KMeans(n_clusters=i, init=param_init, n_init=param_n_init, max_iter=param_max_iter, random_state=0)
        km.fit(data)
        distortions.append(km.inertia_)

    plt.plot(range(1, 10), distortions, marker='o')
    plt.xlabel('Number of Cluster')
    plt.ylabel('Distortion') # WCSS
    plt.savefig('elbow_method.png', dpi=300, facecolor='#ffffff')

    print('elbow_method.png 저장을 완료했습니다.')


def biplot(score, coeff, pcax, pcay, cluster, labels=None):
    """_summary_

    Args:
        score (np.array): pca를 완료한 데이터
        coeff (np.array): 
        pcax (int): 
        pcay (int): 
        cluster (np.array): K-means Clustering을 통해 군집화한 결과값
        labels (_type_, optional): . Defaults to None.
    """
    pca1=pcax-1
    pca2=pcay-1
    xs = score[:,pca1]
    ys = score[:,pca2]
    n=score.shape[1]
    scalex = 1.0/(xs.max()- xs.min())
    scaley = 1.0/(ys.max()- ys.min())

    ## 한글 폰트
    rc('font', family='AppleGothic')
    plt.rcParams['axes.unicode_minus'] = False

    plt.scatter(xs*scalex,ys*scaley, alpha=0.8, c=cluster)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,pca1], coeff[i,pca2], color='r', width=0.005, head_width=0.08)
        if labels is None:
            plt.text(coeff[i,pca1]* 1.3, coeff[i,pca2] * 1.3, "Var"+str(i+1), color='r', ha='center', va='center', fontsize=11)
        else:
            plt.text(coeff[i,pca1]* 1.3, coeff[i,pca2] * 1.3, labels[i], color='r', ha='center', va='center', fontsize=11)
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{} (실력)".format(pcax))
    plt.ylabel("PC{} (활동량)".format(pcay))

    plt.savefig('user_segmentation.png', dpi=300, facecolor='#ffffff')
    print('elbow_method.png 저장을 완료했습니다.')


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='NPS Survey Analysis')
    parser.add_argument('--elbowmethod', type=bool, default=True,
                        help='elbowmethod 수행 여부')
    parser.add_argument('--biplot', type=bool, default=True,
                        help='biplot 수행 여부')
    args = parser.parse_args()

    ## 데이터 로드
    data = pd.read_csv('user_segmentation.csv')
    data = data.drop(['user_id'], axis=1)

    ## 정규화
    scaler = StandardScaler()
    s_data = scaler.fit_transform(data)

    ## PCA
    pca = PCA()
    pca.fit(s_data)
    pca_data = pca.transform(s_data)

    ## Elbow Method
    if args.elbowmethod :
        elbowmethod(pca_data)

    ## K-means Clustering
    optimal_k = 3
    km = KMeans(n_clusters=optimal_k, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=42)
    km_y = km.fit_predict(pca_data)

    ## Biplot
    if args.biplot :
        biplot(pca_data, pca.components_, 1, 2, km_y, labels=data.columns)
