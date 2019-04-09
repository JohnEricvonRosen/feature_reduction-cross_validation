import numpy as np
import itertools
import random as rand
from matplotlib import pyplot as plt

# create graph to determine necessary eigenvectors
def make_graph(S):
    sum = []
    for i in range(len(S)):
        if i == 0:
            sum.append(S[i])
        else:
            sum.append(sum[i-1] + S[i])
    for i, x in enumerate(sum):
         sum[i] = x/(i+1)
    plt.plot(range(0, 240), sum)
    plt.show()
        
        

# this is read and parse
def readfile(filename, index, training=True):
    values = []
    if training:
        lower = index*200
        upper = index*200+100
    else:
        lower = index*200
        upper= index*200+100
    with open(filename, "r") as f:
        for line in itertools.islice(f, lower,upper):
                values.append(line.split())
    rand.shuffle(values)
    return values

def cross_validate(U, training_data, m):
    for j in range(5):
        V = np.zeros((200,240), dtype=np.float64)
        T = np.zeros((800,240), dtype=np.float64)
        for i in range(0, 10):
            temp = training_data[i*100+j*20:i*100+j*20+20]
            V[i*20:i*20 + 20] = temp
            temp1 = training_data[i*100:i*100+j*20]
            temp2 = training_data[i*100+j*20+20:i*100+100]
            temp = np.concatenate((temp1, temp2), axis=0)
            T[i*80:i*80 + 80] = temp
    return V, T

def main():
    #For all digits in the training set do:
    trai_data = []
    test_data = []
    for i in range(0, 10):
        #Parsing data into training set and test set. Random order within each class.
        temp1 = readfile("mfeat-pix.txt", i, True)
        temp2 = readfile("mfeat-pix.txt", i, False)
        trai_data += temp1
        test_data += temp2

    training_data = np.asarray(trai_data, dtype=np.float64)

    #mean of training set is computed
    sum_vector = np.sum(training_data, axis=0)
    mu = np.divide(sum_vector, 1000)

    #training set centered and then correct transpose      
    X = np.subtract(training_data, mu).T

    #C is computed 1/N XX' and then SVD of C
    C = 1/1000 * (X @ X.T)
    U, S, V = np.linalg.svd(C)
        
    # Hand made m. Finding range of m by looking for dissimilarities for each feature 
    # and taking the first m features where the dissimilarity > 2%. 
    var = np.square(S)
    dissimilarity = np.array([np.sum(var[m:]) / np.sum(var) for m in range(241)])
    for m, dis in enumerate(dissimilarity):
        if dis < 0.02:
            break
        
    # make_graph(S)
    cross_validate(U, training_data, m)

    

if __name__ == "__main__":
    main()
