#调用第三方库
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise

def get_K_NN_Rho(x,k):
    nbrs = NearestNeighbors(n_neighbors=k).fit(x)
    distances, indices = nbrs.kneighbors(x)
    distances/=np.max(distances)
    return np.sum(np.exp(-distances**2),axis=1),distances, indices

def get_pre_cluster(x,rho,distances, indices,prun=False):
    def pruning(node,target_pre_center):
        node_ind=indices[node]
        if len(np.where(target_pre_center==node)[0])<=0:
            return
        for i in np.where(target_pre_center==node)[0]:
            if i in node_ind:
                pruning(i,target_pre_center)
            else:
                target_pre_center[i]=-1
                pruning(i,target_pre_center)
        
    n=len(x)
    pre_center=np.ones(n,dtype=np.int0)*-1
    sort_rho=np.flipud(np.argsort(rho))

    for i in range(n):
        min_dis=0
        dis=np.inf
        find=False
        for j,index in enumerate(indices[i]):
            if index==i:
                continue
            if rho[index]>rho[i] : 
                if distances[i,j]<dis:
                    find=True
                    min_dis=j
                    dis=distances[i,min_dis]
        if find:
            pre_center[i]=indices[i,min_dis]

    if prun:
        for i in np.where(pre_center==-1)[0]:
            pruning(i,pre_center)
    zero_index=[]
    for i in range(n):
        if pre_center[i]==-1:
            zero_index.append(i)
    for i in sort_rho:
        if pre_center[i]==-1 or pre_center[i] in zero_index:
            continue
        pre_center[i]=pre_center[pre_center[i]]
    return pre_center

def calLaplacianMatrix(adjacentMatrix,dia):

    # compute the Degree Matrix: D=sum(A)
    degreeMatrix = np.sum(adjacentMatrix, axis=1)

    # print degreeMatrix

    # compute the Laplacian Matrix: L=D-A
    laplacianMatrix = np.diag(degreeMatrix) - adjacentMatrix

    # print laplacianMatrix
    # normailze
    # D^(-1/2) L D^(-1/2)
    dia_sum=degreeMatrix*dia
    # dia_sum=dia
    sqrtDegreeMatrix = np.diag(1.0 / (dia_sum ** (0.5)))
    return np.dot(np.dot(sqrtDegreeMatrix, laplacianMatrix), sqrtDegreeMatrix)

def renormalization(newX):
    Y = newX.copy()
    squar_x=np.sum(newX**2,axis=1)**0.5
    squar_x=squar_x.reshape([len(newX),1])
    return Y/squar_x

def LDP_SC(x,cluster_num,k,norm=True,filter_nois=False,nois_den=0.2):

    rho,distances, indices=get_K_NN_Rho(x,k)
    pre_center=get_pre_cluster(x,rho,distances, indices,prun=False)
    pre_center=pre_center.astype(np.int0)
    t=[]
    for i in range(len(pre_center)):
        if pre_center[i]==-1:
            t.append(i)
    for i in range(len(pre_center)):
        if pre_center[i]==-1:
            pre_center[i]=i
    
    nois=set()
    if filter_nois:
        x_2=x[rho>nois_den]
        result=LDP_SC(x_2,cluster_num,k,filter_nois=False)
        y_last=np.ones_like(pre_center)*-1
        y_last[rho>nois_den]=result
        return y_last
    
    #计算各指标
    if 1:
        G=nx.Graph()
        for i in t:
            # if i in nois:
            #     continue
            G.add_node(i)
        #SNN and dist
        if 1:
            cluster_nbr=[]
            for i in t:
                a=set()
                for j in np.where(pre_center==i)[0]:
                    if j in nois:
                        continue
                    a.update(indices[j])
                cluster_nbr.append(a-nois)

            snn=np.zeros([len(t),len(t)])
            for t_i in range(len(t)):
                for t_j in range(len(t)):
                    if t_i==t_j:
                        snn[t_i,t_j]=0
                        continue
                    intersection=cluster_nbr[t_i] & cluster_nbr[t_j]
                    snn[t_i,t_j]=len(intersection)
                    if len(intersection)>0:
                        sum_rho=0
                        for ind_i,i in enumerate(intersection):
                            sum_rho+=rho[i]
                        G.add_edge(t[t_i],t[t_j],snn=len(intersection),dist=np.linalg.norm(x[t[t_i]]-x[t[t_j]],2),sum_rho=sum_rho)

        #计算mar
        mar=np.zeros([len(t),len(t)])
        for index_i,i in enumerate(t):
            for index_j,j in enumerate(t):
                if index_i>=index_j:
                    continue
                if snn[index_i,index_j]==0:
                    mar[index_i,index_j]=np.inf
                else:
                    set_i=np.where(pre_center==i)[0]
                    set_j=np.where(pre_center==j)[0]
                    # x_set,y_set=np.meshgrid(set_i,set_j)
                    # mar_t=np.linalg.norm((x[x_set]-x[y_set]),axis=2).flatten()
                    mar_t=pairwise.euclidean_distances(x[set_i],x[set_j]).flatten()
                    mar_t.sort()
                    mar_k=4
                    if len(mar_t)<mar_k:
                        mar_k=len(mar_t)
                    for t_i in mar_t[:mar_k]:
                        mar[index_i,index_j]+=t_i
                mar[index_j,index_i]=mar[index_i,index_j]

        for u, v in G.edges:
            G.edges[u,v]['mar']=mar[t.index(u),t.index(v)]

        #gap
        for i in t:
            set_i=np.where(pre_center==i)[0]
            if len(set_i)>1:
                distMatrix = pairwise.euclidean_distances(x[set_i])
                distMatrix+=np.diag(np.ones(len(set_i))*np.inf)
                min_d_i=np.min(distMatrix,axis=0)
                rho_set_i=rho[set_i]
                G.nodes[i]['gap']=np.sum( min_d_i*rho_set_i/np.sum(rho_set_i))
            else:
                G.nodes[i]['gap']=0

    for u, v in G.edges:
        tar_2=G.edges[u,v]['mar']/(G.nodes[u]['gap']*G.nodes[v]['gap']+1e-6)**0.5
        tar_3=G.edges[u,v]['dist']
        tar_5=G.edges[u,v]['snn']
        G.edges[u,v]['weight']=tar_5/(1+tar_2)/(1+tar_3)

    sim=np.zeros([len(t),len(t)])
    for u, v in G.edges:
        sim[t.index(u),t.index(v)]=sim[t.index(v),t.index(u)]=G.edges[u,v]['weight']

    t_num=np.zeros_like(t)
    for ind_i,i in enumerate(t):
        t_num[ind_i]=len(np.where(pre_center==i)[0])

    x_center=x[np.array(t)]


    A_t=pairwise.rbf_kernel(x_center,gamma=1)
    if np.max(sim)>0:
        sim_2=sim/np.max(sim)+np.identity(len(t))
    else:
        sim_2=0
    A=A_t+sim_2*1000
    # A=sim_2
    A/=np.max(A)

    Laplacian = calLaplacianMatrix(A,t_num)

    tzz, V = np.linalg.eig(Laplacian)
    tzz = zip(tzz, range(len(tzz)))
    tzz = sorted(tzz, key=lambda tzz:tzz[0])
    H = np.vstack([V[:,i] for (v, i) in tzz[:cluster_num]]).T
    H_real=H.real
    if norm:
        H_norm=renormalization(H_real)
        sp_kmeans = KMeans(n_clusters=cluster_num).fit(H_norm)
    else:
        sp_kmeans = KMeans(n_clusters=cluster_num).fit(H_real)
    result=sp_kmeans.labels_

    y_last=np.ones_like(pre_center)*-1
    for ind_i,i in enumerate(t):
        y_last[pre_center==i]=result[ind_i]

    return y_last