#!/usr/bin/env python
# coding: utf-8

# In[20]:


import numpy as np
import scipy as sp
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.utils import shuffle


# In[21]:


# open docs file and read its lines
with open("train.dat", "r", encoding="utf8") as fh:
    lines = fh.readlines() 
print(len(lines))

# import numpy as np
# x = np.fromfile("train.dat")


# In[22]:


lines[4]


# In[24]:


docs = list()
for row in lines:
    docs.append(row.rstrip().split(" "))


# In[25]:


# sperate indices and values
# from the doc create two lists containing the word-id and frequencies separately
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

#print the unique data in first doc and corresponding occurences
print(data[0])
print(occurence[0])


# In[6]:


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


def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )


# In[7]:


mat = build_matrix(docs)
csr_info(mat)


# In[8]:


# scale matrix and normalize its rows
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
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

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


# In[9]:


mat2 = csr_idf(mat, copy=True)
mat3 = csr_l2normalize(mat2, copy=True)
mat3_pca = mat3.copy()
print(mat.shape)
print(mat3.shape)


# In[10]:


# perform PCA decomposition on the data
from sklearn.decomposition import PCA, IncrementalPCA
pca = PCA(n_components=1500)
principalComponents = pca.fit_transform(mat3_pca.toarray())


# In[11]:


mat3=principalComponents


# In[55]:


def select_clusters(mat, centroids):
    
    c1 = list()
    c2 = list()
    
    similarityMatrix = mat.dot(centroids.T)
    print("similarityMatrix: ",similarityMatrix)
    for i in range(similarityMatrix.shape[0]):
        similarityRow = similarityMatrix[i]
        
        #Sort the index of the matrix in ascending order of value
        similaritySorted = np.argsort(similarityRow)[-1]
        
        if similaritySorted == 0:
            c1.append(i)
        else:
            c2.append(i)
        
    return c1, c2


# In[56]:


from sklearn.cluster import KMeans

def kmeans(mat, n_iter):
    shuffledMatrix = shuffle(mat, random_state=0)
    centroids = shuffledMatrix[:2,:]

    for _ in range(n_iter):
        clusters = list()
        c1, c2 = select_clusters(mat, centroids)
        
        if len(c1) > 1:
            clusters.append(c1)
        if len(c2) > 1:
            clusters.append(c2)
        
        centroid_list = list()     
        for i in range(0,2):
            cluster = mat[clusters[i],:]
            cluster_mean = cluster.mean(0)
            centroid_list.append(cluster_mean)

        centroids = np.asarray(centroid_list)
        #centroids = np.asarray(kmeans)
        
    return c1, c2


# In[57]:


def calculateSSE(mat, clusters):
    
    sseList = list()
    sseArray = []
    for cluster in clusters:
        rmse = np.sum(np.square(mat[cluster,:] - np.mean(mat[cluster,:])))
        # print("rmse: ",rmse)
        sseList.append(rmse)
    # print("sselist: ",sseList)    
    sseArray = np.asarray(sseList)# 1,6,8,9,22
    # print("sseArray: ",sseArray)
    remove_cluster_idx = np.argsort(sseArray)[-1]
    # print("remove_cluster_idx: ",remove_cluster_idx)        
    return remove_cluster_idx


# In[61]:


from sklearn.cluster import KMeans
def bisecting_kmeans(mat, k, n_iter):
    
    clusters = list()
    
    initialcluster = list()
    # appending dimensions to initialcluster
    for i in range(mat.shape[0]):
        initialcluster.append(i)
    
    clusters.append(initialcluster)
    # print(mat.shape[0])
    # print(clusters)
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


# In[62]:


from sklearn.metrics import silhouette_score
k_values = list()
scores = list()

for k in range(3, 21, 2):
    labels = bisecting_kmeans(mat3, k, 10)
    if (k == 7):
        # write to output file
        outputFile = open("output.dat", "w")
        for docuId, index in enumerate(labels, start=1):
            row = str(index) #  str(docuId) + ',' +
            outputFile.write(row +'\n')
        outputFile.close()

    sscore = silhouette_score(mat3, labels)
    k_values.append(k)
    scores.append(sscore)
    print ("For K= %d silhouette_coefficient Score is %f" %(k, sscore))


# In[17]:


# Plotting silhoutte score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_values, scores)
plt.xlabel('Number of Clusters k')
plt.ylabel('Silhouette Score')


# In[19]:


# How to get the ideal number of documents
from sklearn.decomposition import PCA, IncrementalPCA
mat3_pca.shape
pca = PCA(n_components=7200)

principalComponents = pca.fit_transform(mat3_pca.toarray())

principalComponents

principalComponents.shape

cs = np.cumsum(pca.explained_variance_/np.sum(pca.explained_variance_))
cs[cs>0.7] 

for e in range(len(cs)):
    if cs[e]> 0.99:
        print(e)
        break

cs[2087]


# In[ ]:




