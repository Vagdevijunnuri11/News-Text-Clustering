#!/usr/bin/env python
# coding: utf-8

# In[15]:


import numpy as np
import random
from collections import defaultdict
from scipy.sparse import csr_matrix
from sklearn.utils import shuffle


# In[16]:


# open docs file and read its lines
with open("train.dat", "r", encoding="utf8") as fh:
    lines = fh.readlines() 
print(len(lines))


# In[17]:


def csr_read(fname, ftype="csr", nidx=1):

    with open(fname) as f:
        lines = f.readlines()
    if ftype == "csr":
        nrows = len(lines)
        ncols = 0 
        nnz = 0 
        for i in range(nrows):
            p = lines[i].split()
            if len(p) % 2 != 0:
                raise ValueError("Invalid CSR matrix. Row %d contains %d numbers." % (i, len(p)))
            nnz += len(p)/2
            for j in range(0, len(p), 2): 
                cid = int(p[j]) - nidx
                if cid+1 > ncols:
                    ncols = cid+1
    else:
        raise ValueError("Invalid sparse matrix ftype '%s'." % ftype)
    val = np.zeros(int(nnz), dtype=np.float)
    ind = np.zeros(int(nnz), dtype=np.int)
    ptr = np.zeros(int(nrows+1), dtype=np.long)
    n = 0 
    for i in range(nrows):
        p = lines[i].split()
        for j in range(0, len(p), 2): 
            ind[n] = int(p[j]) - nidx
            val[n] = float(p[j+1])
            n += 1
        ptr[i+1] = n 
    
    assert(n == nnz)
    
    return csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.float)


# In[18]:


def csr_idf(matrix, copy=False, **kargs):

    if copy is True:
        matrix = matrix.copy()
    nrows = matrix.shape[0]
    nnz = matrix.nnz
    ind, val, ptr = matrix.indices, matrix.data, matrix.indptr
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else matrix


# In[19]:


def csr_l2normalize(matrix, copy=False, **kargs):

    if copy is True:
        matrix = matrix.copy()
    nrows = matrix.shape[0]
    nnz = matrix.nnz
    ind, val, ptr = matrix.indices, matrix.data, matrix.indptr
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue
        rsum = float(1.0/np.sqrt(rsum))
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return matrix


# In[20]:


def initialCentroids(matrix):
    matrixShuffled = shuffle(matrix, random_state=0)
    return matrixShuffled[:2,:]


# In[21]:


def similarity(matrix, centroids):
    similarities = matrix.dot(centroids.T)
    return similarities


# In[22]:


def findClusters(matrix, centroids):
    
    clusterA = list()
    clusterB = list()
    
    similarityMatrix = similarity(matrix, centroids)
    
    for index in range(similarityMatrix.shape[0]):
        similarityRow = similarityMatrix[index]
        similaritySorted = np.argsort(similarityRow)[-1]
        
        if similaritySorted == 0:
            clusterA.append(index)
        else:
            clusterB.append(index)
        
    return clusterA, clusterB


# In[23]:


def recomputeCentroid(matrix, clusters):
    centroids = list()
    
    for i in range(0,2):
        cluster = matrix[clusters[i],:]
        clusterMean = cluster.mean(0)
        centroids.append(clusterMean)
        
    centroids_array = np.asarray(centroids)
    
    return centroids_array


# In[24]:


def kmeans(matrix, numberOfIterations):
    
    centroids = initialCentroids(matrix)
    
    for _ in range(numberOfIterations):
        
        clusters = list()
        
        clusterA, clusterB = findClusters(matrix, centroids)
        
        if len(clusterA) > 1:
            clusters.append(clusterA)
        if len(clusterB) > 1:
            clusters.append(clusterB)
            
        centroids = recomputeCentroid(matrix, clusters)
        
    return clusterA, clusterB


# In[25]:


def calculateSSE(mat, clusters):
    
    sseList = list()
    sseArray = []
    for cluster in clusters:
        rmse = np.sum(np.square(mat[cluster,:] - np.mean(mat[cluster,:])))
        sseList.append(rmse)
    sseArray = np.asarray(sseList)
    removed_cluster_idx = np.argsort(sseArray)[-1]
    return removed_cluster_idx


# In[26]:


from sklearn.cluster import KMeans
def bisecting_kmeans(matrix, k, numberOfIterations):
    
    clusters = list()
    initialcluster = list()
    for i in range(matrix.shape[0]):
        initialcluster.append(i)    
    clusters.append(initialcluster)
    
    while len(clusters) < k:
        dropClusterIndex = calculateSSE(matrix, clusters)
        droppedCluster = clusters[dropClusterIndex]        
        clusterA, clusterB = kmeans(matrix[droppedCluster,:], numberOfIterations)
        del clusters[dropClusterIndex]
        
        actualClusterA = list()
        actualClusterB = list()
        for index in clusterA:
            actualClusterA.append(droppedCluster[index])
            
        for index in clusterB:
            actualClusterB.append(droppedCluster[index])
        
        clusters.append(actualClusterA)
        clusters.append(actualClusterB)
    
    labels = [0] * matrix.shape[0]

    for index, cluster in enumerate(clusters):
        for idx in cluster:
            labels[idx] = index + 1
    return labels


# In[27]:


csrMatrix = csr_read('train.dat', ftype="csr", nidx=1)
csrIDF = csr_idf(csrMatrix, copy=True)
csrL2Normalized = csr_l2normalize(csrIDF, copy=True)
denseMatrix = csrL2Normalized.toarray()
print("Dense:",denseMatrix)


# In[28]:


labels = bisecting_kmeans(denseMatrix, 7, 10)
# write result to output file
outputFile = open("output.dat", "w")
for index in labels:
        outputFile.write(str(index) +'\n')
outputFile.close()


# In[ ]:



