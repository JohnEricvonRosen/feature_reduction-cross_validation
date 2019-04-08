import numpy as np
import itertools

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

def cross_validate(U, test_data):
    return 'yo'

def main():
    training_data = []
    test_data = []

    #Parsing data into training set and test set 
    for i in range(0, 10):
        training_data.append(readfile("DigitsBasicRoutines/mfeat-pix.txt", i, True))
        test_data.append(readfile("DigitsBasicRoutines/mfeat-pix.txt", i, False))
    
    #mean of training set is computed
    sum_vector = np.zeros((240,), dtype=np.float64)
    for num_set in training_data:
        for patern in num_set:
            np.add(sum_vector, patern, out=sum_vector, casting='unsafe')
    mu = np.divide(sum_vector, 1000)

    #training set centered
    alist = []
    for num_set in training_data:
        for patern in num_set:
            alist.append(np.abs(np.subtract(patern, mu)))
    X = np.array(alist)

    #C is computed 1/N XX' and then SVD of C
    C = np.multiply(1/1000,np.dot(X, np.transpose(X)))
    U, S, V = np.linalg.svd(C)

    #Returns nxm sized matrix Um
    Um = cross_validate(U, test_data)
    print(Um)

if __name__ == "__main__":
    main()
