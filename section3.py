import numpy as np
import scipy.io.matlab
import matplotlib.pyplot as plt

f = scipy.io.matlab.loadmat("mnist.mat", struct_as_record=False)

def load_images(digit):
    return [f['training'][0][0].images[:, :, i] for i in range(60000) if f['training'][0][0].labels[i][0] == digit]

def load_section3():
    return np.roll(np.array([f['test'][0][0].images[:, :, i].reshape(28*28) for i in range(10000)]), 0)

def load_test_labels():
    return [f['test'][0][0].labels[i][0] for i in range(10000)]

imagesPerDigit = [
    load_images(d) for d in range(10)
]

def load_other_digits(digit):
    return [f['training'][0][0].images[:, :, i] for i in range(60000) if f['training'][0][0].labels[i][0] != digit]

def create_section3():
    N = 5400
    imagesPerDigit = [
        load_images(d)[:N] for d in range(10)
    ]
    for digit1 in range(0, 10):
        x = []
        xFile = open("{}.npy".format(digit1), mode="wb")
        imagesPerDigit1 = imagesPerDigit[digit1]
        otherDigits = load_other_digits(digit1)
        A = np.zeros(shape=(2*N, 28**2))
        b = np.zeros(shape=2*N)
        for i in range(N):
            b[2*i] = 1
            A[2*i] = np.reshape(imagesPerDigit1[i],(28*28))
            A[2*i + 1] = np.reshape(otherDigits[i], (28 * 28))
            b[2*i + 1] = -1
        A = np.block([A, np.ones(shape=(2*N, 1))])
        x = np.linalg.pinv(A) @ b
        np.save(xFile, x)
        xFile.close()


def section3():
    images = load_section3()
    images = np.block([images, np.ones(shape=(len(images), 1))])
    labels = load_test_labels()
    failCount = 0
    results = []
    x = [np.load("{}.npy".format(i)) for i in range(0, 10)]
    for i in range(len(images)):
        for digit1 in range(0, 10):
            results.append( images[i] @ x[digit1])
        pred = np.argmax(results)
        if pred != labels[i]:
            failCount += 1
            if failCount < 6:
                plt.imshow(images[i][:-1].reshape((28, 28)), cmap='gray')
                plt.show()
        results = []
    print((len(images) - failCount) / len(images) * 100.0)

create_section3()
section3()










