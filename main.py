from sklearn.preprocessing import StandardScaler
from src.evaluation import compute_score
from src.LDPSC import LDP_SC #our algorithm
import scipy.io as scio
import numpy as np

if __name__ == '__main__':
    #load data'
    data = scio.loadmat('./data/realworld/Pendigits.mat')
    x=data['data']
    y=data['labels']
    s_x=StandardScaler()
    x=s_x.fit_transform(x)
    y=y.reshape(y.shape[0])
    cluster_num=len(np.unique(y))# the number of data
    
    #start clustering'
    y_predict=LDP_SC(x,cluster_num,12)#data,cluster number, Number of neighbors
    ARI,NMI,ACC=compute_score(y_predict,y)#score
    print('ARI: {a:.2f}\nNMI: {b:.2f}\nACC: {c:.2f}'.format(a = ARI,b = NMI,c = ACC))
