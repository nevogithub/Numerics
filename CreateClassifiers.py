import numpy as np
import scipy.io.matlab

f = scipy.io.matlab.loadmat("mnist.mat", struct_as_record=False)

def load_images(digit):
    return [f['training'][0][0].images[:, :, i] for i in range(60000) if f['training'][0][0].labels[i][0] == digit]

N = 4000
imagesPerDigit = [
    load_images(d) for d in range(10)
]
for digit1 in range(0, 10):
    x = []
    xFile = open("{}.npy".format(digit1), mode="wb")
    for digit2 in range(10):
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
        x.append(np.linalg.pinv(A_train) @ b_train)
    x = np.column_stack(x)
    np.save(xFile, x)
    xFile.close()




