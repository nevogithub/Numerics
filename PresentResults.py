import numpy as np
import scipy.io.matlab
import matplotlib.pyplot as plt

f = scipy.io.matlab.loadmat("mnist.mat", struct_as_record=False)

def load_images(digit):
    return [f['training'][0][0].images[:, :, i] for i in range(60000) if f['training'][0][0].labels[i][0] == digit]

N = 4000
imagesPerDigit = [
    load_images(d) for d in range(10)
]

def section1():
    digit1 = 0
    trainFailCount = 0
    testFailCount = 0
    fails = []
    x = np.load("{}.npy".format(digit1))
    for digit2 in range(1, 10):
        imagesPerDigit1 = imagesPerDigit[digit1]
        imagesPerDigit2 = imagesPerDigit[digit2]
        A = np.zeros(shape=(2*N, 28**2))
        b = np.zeros(shape=2*N)
        for i in range(N):
            b[2*i] = 1
            b[2*i + 1] = -1
            A[2*i] = np.reshape(imagesPerDigit1[i],(1,28*28))
            A[2*i+1] = np.reshape(imagesPerDigit2[i],(1,28*28))
        A = np.block([A, np.ones(shape=(2*N, 1))])
        A_train = A[0:N]
        b_train = b[0:N]
        A_test = A[N:2 * N, :]
        b_test = b[N:2 * N]
        predTrain = np.sign(A_train @ x[:, digit2])
        predTest = np.sign(A_test @ x[:, digit2])
        for i in range(N):
            if b_train[i] != predTrain[i]:
                trainFailCount += 1
                fails.append(A_train[i][:-1].reshape((28,28)))
            if b_test[i] != predTest[i]:
                testFailCount += 1
                fails.append(A_test[i][:-1].reshape((28,28)))
    print(trainFailCount)
    print(testFailCount)
    for i in range(5):
        plt.imshow(fails[i], cmap='gray')
        plt.show()


def section2():
    for digit1 in range(0, 10):
        x = np.load("{}.npy".format(digit1))
        trainFailCount = 0
        testFailCount = 0
        showImage = True
        numbersShown = set()
        for digit2 in (d for d in range(0, 10) if d != digit1):
            imagesPerDigit1 = imagesPerDigit[digit1]
            imagesPerDigit2 = imagesPerDigit[digit2]
            A = np.zeros(shape=(2*N, 28**2))
            b = np.zeros(shape=2*N)
            for i in range(N):
                b[2*i] = 1
                b[2*i + 1] = -1
                A[2*i] = np.reshape(imagesPerDigit1[i],(1,28*28))
                A[2*i+1] = np.reshape(imagesPerDigit2[i],(1,28*28))
            A = np.block([A, np.ones(shape=(2*N, 1))])
            A_train = A[0:N]
            b_train = b[0:N]
            A_test = A[N:2 * N, :]
            b_test = b[N:2 * N]
            predTrain = np.sign(A_train @ x[:, digit2])
            predTest = np.sign(A_test @ x[:, digit2])
            for i in range(N):
                if b_train[i] != predTrain[i]:
                    trainFailCount += 1
                    if showImage and digit2 not in numbersShown:
                        plt.imshow(A_train[i][:-1].reshape((28,28)), cmap='gray')
                        plt.show()
                        numbersShown.add(digit2)
                        showImage = False
                if b_test[i] != predTest[i]:
                    testFailCount += 1
                    if showImage and digit2 not in numbersShown:
                        plt.imshow(A_test[i][:-1].reshape((28,28)), cmap='gray')
                        plt.show()
                        numbersShown.add(digit2)
                        showImage = False
        print(digit1)
        print("Train fail count: {}".format(trainFailCount))
        print("Test fail count: {}".format(testFailCount))
        print()


section1()






