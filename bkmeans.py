#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np
import scipy as sp
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.utils import shuffle


# In[37]:


# open docs file and read its lines
with open("train.dat", "r", encoding="utf8") as fh:
    lines = fh.readlines() 
print(len(lines))


# In[38]:


docs = list()
for row in lines:
    docs.append(row.rstrip().split(" "))


# In[39]:



data = list()
occurence = list()
for dat in docs:
    data_index = list()
    occurence_index = list()
    # pick alternate elements start from index 0
    for i in range(0, len(dat), 2):      
        data_index.append(int(dat[i]))
    # pick alternate elements start from index 1
    for j in range(1, len(dat), 2):     
        occurence_index.append(int(dat[j]))
    data.append(data_index)
    occurence.append(occurence_index)

print(data[0])
print(occurence[0])


# In[40]:


from collections import Counter
from scipy.sparse import csr_matrix
def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


# In[41]:


mat = build_matrix(docs)


# In[42]:


# scale matrix
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat


# In[43]:


# normalize
def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat


# In[44]:


# initializing the clusters
def initialCentroids(matrix):
    matrixShuffled = shuffle(matrix, random_state=0)
    return matrixShuffled[:2,:]


# In[45]:


mat2 = csr_idf(mat, copy=True)
mat3 = csr_l2normalize(mat2, copy=True)
mat3_pca = mat3.copy()
print(mat.shape)
print(mat3.shape)


# In[46]:


# pca decomposition
from sklearn.decomposition import PCA, IncrementalPCA
pca = PCA(n_components=1500)
principalComponents = pca.fit_transform(mat3_pca.toarray())


# In[47]:


mat3=principalComponents


# In[48]:


def select_clusters(mat, centroids):
    
    c1 = list()
    c2 = list()
    
    similarityMatrix = mat.dot(centroids.T)
    for i in range(similarityMatrix.shape[0]):
        similarityRow = similarityMatrix[i]
        
        similaritySorted = np.argsort(similarityRow)[-1]
        
        if similaritySorted == 0:
            c1.append(i)
        else:
            c2.append(i)
        
    return c1, c2


# In[54]:


# calculating the centroids
def recomputeCentroid(matrix, clusters):
    centroids = list()
    
    for i in range(0,2):
        cluster = matrix[clusters[i],:]
        clusterMean = cluster.mean(0)
        centroids.append(clusterMean)
        
    centroids_array = np.asarray(centroids)
    
    return centroids_array


# In[55]:


from sklearn.cluster import KMeans

def kmeans(mat, n_iter):
    centroids = initialCentroids(mat)

    for _ in range(n_iter):
        clusters = list()
        c1, c2 = select_clusters(mat, centroids)
        
        if len(c1) > 1:
            clusters.append(c1)
        if len(c2) > 1:
            clusters.append(c2)
        
        centroids = recomputeCentroid(mat, clusters)
        
    return c1, c2


# In[56]:


def calculateSSE(mat, clusters):
    
    sseList = list()
    sseArray = []
    for cluster in clusters:
        rmse = np.sum(np.square(mat[cluster,:] - np.mean(mat[cluster,:])))
        sseList.append(rmse)
    sseArray = np.asarray(sseList)
    remove_cluster_idx = np.argsort(sseArray)[-1]
    return remove_cluster_idx


# In[57]:


from sklearn.cluster import KMeans
def bisecting_kmeans(mat, k, n_iter):
    
    clusters = list()
    
    initialcluster = list()
    # appending dimensions to initialcluster
    for i in range(mat.shape[0]):
        initialcluster.append(i)
    
    clusters.append(initialcluster)

    while len(clusters) < k:# 3

        remove_cluster_idx = calculateSSE(mat, clusters)
        removed_cluster = clusters[remove_cluster_idx]
        
        c1, c2 = kmeans(mat[removed_cluster,:], n_iter)
        del clusters[remove_cluster_idx]
        
        
        real_c1 = list()
        real_c2 = list()
        for index in c1:
            real_c1.append(removed_cluster[index])
            
        for index in c2:
            real_c2.append(removed_cluster[index])
        
        clusters.append(real_c1)
        clusters.append(real_c2)
    
    labels = [0] * mat.shape[0]

    for index, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = index + 1
    return labels


# In[58]:


from sklearn.metrics import silhouette_score
k_values = list()
scores = list()

for k in range(3, 21, 2):
    labels = bisecting_kmeans(mat3, k, 10)
    if (k == 7):        
        outputFile = open("output.dat", "w")
        for docuId, index in enumerate(labels, start=1):
            row = str(index)
            outputFile.write(row +'\n')
        outputFile.close()

    sscore = silhouette_score(mat3, labels)
    k_values.append(k)
    scores.append(sscore)
    print ("For K= %d silhouette_coefficient Score is %f" %(k, sscore))


# In[59]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_values, scores)
plt.xlabel('Number of Clusters k')
plt.ylabel('Silhouette Score')


# In[ ]:




