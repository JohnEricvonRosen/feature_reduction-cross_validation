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

#quadratic loss function
def loss(Dx, y):
    return np.linalg.norm(Dx - y)**2

def mse(Dx, f, T, z):
    temp = []
    for i, x in enumerate(T):
        temp1 = Dx @ f @ x
        temp1 = np.divide(temp1, np.linalg.norm(temp1))
        temp.append(np.linalg.norm(z[i] - temp1)**2)
    return sum(temp)/len(T)

def miss(Dx, f, V, y):
    count = 0
    for i, x in enumerate(V):
        temp = np.argmax(Dx @ f @ x)
        if temp != y[i]:
            count += 1
    temp = count/len(V)
    return temp

def cross_validate(U, D, training_data, res, krange, mu):
    z = []
    zi = []
    y = []
    yi = []
    MSEtrain = []
    MSEtest = []
    MISStrain = []
    MISStest = []

    for i in range(10):
        for j in range(80):
            if i == 0:
                z.append(np.array([1,0,0,0,0,0,0,0,0,0]))
            elif i == 1:
                z.append(np.array([0,1,0,0,0,0,0,0,0,0]))
            elif i == 2:
                z.append(np.array([0,0,1,0,0,0,0,0,0,0]))
            elif i == 3:
                z.append(np.array([0,0,0,1,0,0,0,0,0,0]))
            elif i == 4:
                z.append(np.array([0,0,0,0,1,0,0,0,0,0]))
            elif i == 5:
                z.append(np.array([0,0,0,0,0,1,0,0,0,0]))
            elif i == 6:
                z.append(np.array([0,0,0,0,0,0,1,0,0,0]))
            elif i == 7:
                z.append(np.array([0,0,0,0,0,0,0,1,0,0]))
            elif i == 8:
                z.append(np.array([0,0,0,0,0,0,0,0,1,0]))
            elif i == 9:
                z.append(np.array([0,0,0,0,0,0,0,0,0,1]))
            zi.append(i)
        for j in range(20):
            if i == 0:
                y.append(np.array([1,0,0,0,0,0,0,0,0,0]))
            elif i == 1:
                y.append(np.array([0,1,0,0,0,0,0,0,0,0]))
            elif i == 2:
                y.append(np.array([0,0,1,0,0,0,0,0,0,0]))
            elif i == 3:
                y.append(np.array([0,0,0,1,0,0,0,0,0,0]))
            elif i == 4:
                y.append(np.array([0,0,0,0,1,0,0,0,0,0]))
            elif i == 5:
                y.append(np.array([0,0,0,0,0,1,0,0,0,0]))
            elif i == 6:
                y.append(np.array([0,0,0,0,0,0,1,0,0,0]))
            elif i == 7:
                y.append(np.array([0,0,0,0,0,0,0,1,0,0]))
            elif i == 8:
                y.append(np.array([0,0,0,0,0,0,0,0,1,0]))
            elif i == 9:
                y.append(np.array([0,0,0,0,0,0,0,0,0,1]))
            yi.append(i)
    #Split the training data into 5 folds
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
        
        Dopt = []
        Rval = []
        for k, modelclass in enumerate(D):
            #Step 2.2.1 of cross-validation
            for foo in modelclass:
                values = []
                descisionloss = 0

                for j, x in enumerate(T):
                    descision = foo @ U[:krange[k]+1] @ (x - mu)
                    descision = np.divide(descision, np.linalg.norm(descision))
                    descisionloss += loss(descision, z[j])

                descisionloss = descisionloss/len(T)
                values.append(descisionloss)
            Dopt.append(modelclass[np.argmin(values)])

            #Step 2.2.2 of cross-validation
            Rvalj = []
            descisionloss = 0
            for j, x in enumerate(V):
                descision = Dopt[k] @ U[:krange[k]+1] @ (x - mu) 
                descision = np.divide(descision, np.linalg.norm(descision))
                descisionloss += loss(descision, y[j])
                Rvalj.append(descisionloss/200)

            #Step 3
            Rval.append(1/5 * sum(Rvalj))

            MSEtrain.append(mse(Dopt[k], U[:krange[k]+1], T, z))
            MSEtest.append(mse(Dopt[k], U[:krange[k]+1], V, y))
            MISStrain.append(miss(Dopt[k], U[:krange[k]+1], T, zi))
            MISStest.append(miss(Dopt[k], U[:krange[k]+1], V, yi))
    #Step 4
    mopt = np.argmin(Rval)

    #Step 5
    values = []
    for foo in D[mopt]:
        descisionloss = 0

        for j, x in enumerate(training_data):
            descision = foo @ U[:krange[mopt]+1] @ (x - mu)
            descision = np.divide(descision, np.linalg.norm(descision))
            descisionloss += loss(descision, res[j])

        descisionloss = descisionloss/len(training_data)
        values.append(descisionloss)

    alphaopt = np.argmin(values)
    return mopt, alphaopt, MSEtrain, MSEtest, MISStrain, MISStest
            

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
    U, S, _ = np.linalg.svd(C)
        
    # Hand made k. Finding range of k by looking for dissimilarities for each feature 
    # and taking the first k features where the dissimilarity > 2%. 
    var = np.square(S)
    dissimilarity = np.array([np.sum(var[m:]) / np.sum(var) for m in range(241)])
    for m, dis in enumerate(dissimilarity):
        if dis < 0.02:
            break
    
    #NOTE: Random choice of range for k.
    #Creating Model Class inclusion sequence D with k ranging from k-5 to k+5
    D = []
    if m < 5:
        krange = range(0, m+5)
    else:
        krange = range(m-5, m+5)

    for k in krange:
        #Creating f(x) with 1.0 padding
        fx = np.array([np.append(U[:k] @ x, 1.0) for x in X.T])

        #Creating z
        z = np.empty((1000, 10))
        for i in range(10):
            for j in range(100):
                if i == 0:
                    temp = np.array([1,0,0,0,0,0,0,0,0,0])
                elif i == 1:
                    temp = np.array([0,1,0,0,0,0,0,0,0,0])
                elif i == 2:
                    temp = np.array([0,0,1,0,0,0,0,0,0,0])
                elif i == 3:
                    temp = np.array([0,0,0,1,0,0,0,0,0,0])
                elif i == 4:
                    temp = np.array([0,0,0,0,1,0,0,0,0,0])
                elif i == 5:
                    temp = np.array([0,0,0,0,0,1,0,0,0,0])
                elif i == 6:
                    temp = np.array([0,0,0,0,0,0,1,0,0,0])
                elif i == 7:
                    temp = np.array([0,0,0,0,0,0,0,1,0,0])
                elif i == 8:
                    temp = np.array([0,0,0,0,0,0,0,0,1,0])
                elif i == 9:
                    temp = np.array([0,0,0,0,0,0,0,0,0,1])
                z[i*100+j] = temp

        #NOTE: Random choice of range for alpha
        #Ridge rigression with alpha ranging from 4 to 20 
        wopt = []
        for alpha in range(4, 20):
            wopt.append(np.transpose(np.transpose(1/1000 * (fx.T @ fx) + alpha**2 * np.identity(k+1)) @ (1/1000 * (fx.T @ z))))
        D.append(wopt)

    # make_graph(S)
    mopt, alphaopt, MSEtrain, MSEtest, MISStrain, MISStest = cross_validate(U, D, training_data, z, krange, mu)
    print("Model Dm%dj%d" % (krange[mopt], alphaopt))
    

    #MSEtrain, MSEtest, MISStrain, MISStest are 1d arrays. the first 10 values are for a certain alpha
    #and then for each alpha there are k different values. So it's like
    # 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4 ,5 ,...
    #You'll need to make a new graph for each alpha I think. So you'll group all the
    #1s together and all the 2s...
    

if __name__ == "__main__":
    main()
