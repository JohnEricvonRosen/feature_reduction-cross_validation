import numpy as np
import itertools
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
    values = list()
    if training:
        lower = index*200
        upper = index*200+100
    else:
        lower = index*200+100
        upper= index*200+200
    with open(filename, "r") as f:
        for line in itertools.islice(f, lower,upper):
                values.append(line.split())
    return np.asarray(values, dtype=np.float64)

def cross_validate(U, test_data, L):
    for j in range(10):
        V = test_data[j*100:j*100+100]
        T = np.concatenate((test_data[0:j*100], test_data[j*100+100:]), axis=0)
        for m in range(10, 51): #m = 10, ... ,50
            pass

def main():
    mk = []
    fk = []
    #For all digits in the training set do:
    for i in range(0, 10):
        #Parsing data into training set and test set 
        training_data = np.empty((0,240), dtype=np.float64)
        test_data = np.empty((0,240), dtype=np.float64)

        training_data = np.vstack((training_data, readfile("mfeat-pix.txt", i, True)))
        test_data = np.vstack((test_data, readfile("mfeat-pix.txt", i, False)))
    
        #mean of training set is computed
        sum_vector = np.sum(training_data, axis=0)
        mu = np.divide(sum_vector, 100)

        #training set centered        
        X = training_data - mu

        #C is computed 1/N XX' and then SVD of C
        C = 1/100 * (X.T @ X)
        U, S, V = np.linalg.svd(C)
        
        var = np.square(S)

        dissimilarity = np.array([np.sum(var[m:]) / np.sum(var) for m in range(241)])

        for m, dis in enumerate(dissimilarity):
            if dis < 0.02:
                mk.append(m)
                fk.append(U[:m])
                break
        
    print(str(mk))
    make_graph(S)
        # cross_validate(U, test_data, 5)

    

if __name__ == "__main__":
    main()
