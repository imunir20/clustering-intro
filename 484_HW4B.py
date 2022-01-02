import numpy as np
import pandas as pd
import random
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)


def writePredictions(fName):
    writeFile = open(fName, 'w')
    for j in range(n - 1):
        writeFile.write(str(pLabels[j][0]) + '\n')
    writeFile.write(str(pLabels[n - 1][0]))
    writeFile.close()


def plotEvaluation():
    #print("SILHOUETTE SCORE WAS " + str(silhouette_score(features, pLabels, metric='cosine')))
    #x = [3, 5, 7, 9, 11, 13, 15, 17, 19]
    x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    y = []
    k = 2
    for w in range(10):
        clusterSums = np.zeros((k, data.shape[1]))   # Keeps track of sums for each clusters and its dimensions (length is k x dimensions)
        clusterNums = []
        for z in range(k):
            clusterNums.append(0)
        performKMC()
        score = silhouette_score(features, pLabels, metric='cosine')
        print('For k = ' + str(k) + ", SCORE WAS " + str(score))
        y.append(score)
        k += 2
    
    plt.plot(x, y)
    plt.xlabel('Value of K')
    plt.ylabel('Silhouette Score')
    plt.title('Part 2: Silhouette Score of varying levels of K in K-Means Clustering')
    plt.show()


def initCentroids(data):
    kmCents = np.zeros((k, data.shape[1]))
    first = random.randint(0, n)
    #kmCents[0] = data.iloc[first].to_numpy()
    kmCents[0] = data[first]
    #data = data.drop(first)
    data = np.delete(data, first, axis=0)
    current = 1

    for r in range(k - 1):
        distances = np.square(pairwise_distances(data, kmCents[:current]))
        minimums = np.amin(distances, axis=1)
        minSum = np.sum(minimums)
        probs = minimums/minSum
        newCent = random.uniform(0.0, 1.0)
        print('New cent\'s probability was ' + str(newCent))

        if probs[0] >= newCent:
            print("Chosen centroid had a probability of " + str(probs[0]))
            #kmCents[current] = data.iloc[0].to_numpy()
            kmCents[current] = data[0]
            #data = data.drop(0)
            data = np.delete(data, 0, axis=0)
        else:
            for s in range(n-1):
                probs[s + 1] = probs[s] + probs[s + 1]
                if probs[s + 1] >= newCent:
                    print("Chosen centroid had a probability of " + str(probs[s + 1]))
                    #kmCents[current] = data.iloc[s + 1].to_numpy()
                    kmCents[current] = data[s + 1]
                    #data = data.drop(s + 1)
                    data = np.delete(data, s + 1, axis=0)
                    break
        current += 1
    return kmCents

#print(data)

def performKMC():
    centroids = initCentroids(data)

    minCentroid = 0.0
    currDist = 0.0
    it = 0
    #oldCentroids = centroids
    #oldCentroids = oldCentroids.to_numpy()
    

    while True:
        dm = pairwise_distances(features, centroids, metric='cosine')

        for t in range(10):
            clustersX[t].clear()
            clustersY[t].clear()
        
        for g in range(dm.shape[0]):      # For each point, assign it to a cluster (label), add to that cluster sum and total amount of points accordingly
            minCentroid = dm[g][0]        # Default cluster that this point is assigned to is the first cluser (tentative to change)
            pLabels[g][0] = 1
            for h in range(dm.shape[1] - 1):      # Finding the point's cluster (minimum distance out of all of those points' columns)
                currDist = dm[g][h+1]
                if currDist < minCentroid:
                    minCentroid = currDist
                    pLabels[g][0] = h + 2

            clusterNums[pLabels[g][0] - 1] += 1
            #clusterSums[pLabels[g][0] - 1] += features.iloc[g]
            clusterSums[pLabels[g][0] - 1] += features[g]
            clustersX[pLabels[g][0] - 1].append(features[g][0])
            clustersY[pLabels[g][0] - 1].append(features[g][1])

        oldCentroids = centroids
        #if it != 0 :
        #    oldCentroids = centroids
        #oldCentroids = oldCentroids.to_numpy()
        for j in range(clusterSums.shape[0]):
            #print(clusterSums[j])
            #print(clusterNums[j])
            clusterSums[j] = (clusterSums[j])/clusterNums[j]
            #centroids = clusterSums
        centroids = clusterSums
        
        it += 1
        print("Iteration " + str(it))
        print("Old Centroids")
        print(oldCentroids)
        print("Current Centroids")
        print(centroids)

        if np.array_equal(oldCentroids, centroids) == True:
            print("Done!")
            break



clustersX = [[], [], [], [], [], [], [], [], [], []]
clustersY = [[], [], [], [], [], [], [], [], [], []]

k = 10    # K in K-means clustering
centroids = pd.DataFrame()
clusterNums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]   # Keeps track of number of points assigned to each cluster (length is k)

data = pd.read_csv('484HW4_Image_Test_Data.txt', header=None)        # TRAIN DATA FILES READ
#scaler = MinMaxScaler()
#data = scaler.fit_transform(data)
pca = PCA(n_components=2)
data = pca.fit_transform(data)

#print(data)

features = data

n = data.shape[0]
pLabels = np.zeros((n, 1), dtype=int)     # Keeps track of predicted labels (length is n)
clusterSums = np.zeros((k, data.shape[1]))   # Keeps track of sums for each clusters and its dimensions (length is k x dimensions)

# Perform K-means clustering
performKMC()


print(clustersX)
print(clustersY)
plt.scatter(clustersX[0], clustersY[0], color = 'red')
plt.scatter(clustersX[1], clustersY[1], color = 'blue')
plt.scatter(clustersX[2], clustersY[2], color = 'green')
plt.scatter(clustersX[3], clustersY[3], color = 'orange')
plt.scatter(clustersX[4], clustersY[4], color = 'yellow')
plt.scatter(clustersX[5], clustersY[5], color = 'purple')
plt.scatter(clustersX[6], clustersY[6], color = 'maroon')
plt.scatter(clustersX[7], clustersY[7], color = 'pink')
plt.scatter(clustersX[8], clustersY[8], color = 'magenta')
plt.scatter(clustersX[9], clustersY[9], color = 'navy')

plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.title('K-mean clusters (k = 10): PCA 1 vs PCA 2')
plt.show()


#plotEvaluation()

print('Beginning writing')
fileName = '484_HW4_Attempt14.txt'
writePredictions(fileName)
print('Done writing')

