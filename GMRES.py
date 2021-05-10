import numpy as np
import matplotlib.pyplot as plt


def gmres(A, b, x, n):
    # for plot
    xline, ylineRes = [], []

    for i in range(n):
        # for residual plot
        xline.append(i)
        r = b - A @ x
        curr_res_norm = np.linalg.norm(r)
        ylineRes.append(curr_res_norm)

        # apply iteration
        rkt = np.transpose(r)
        Ar = A @ r
        alpha = (rkt @ Ar) / (rkt @ np.transpose(A) @ Ar)
        x = x + alpha * r

    # plot creation
    plt.semilogy(xline, ylineRes)
    plt.title('Residual norm vs. the iterations')
    plt.xlabel('iterations')
    plt.ylabel('residual norm')
    plt.show()


def main():
    A = np.array([[5, 4, 4, -1, 0], [3, 12, 4, -5, -5],
                  [-4, 2, 6, 0, 3], [4, 5, -7, 10, 2], [1, 2, 5, 3, 10]])
    b = np.array([1, 1, 1, 1, 1])
    x = np.zeros(5)

    gmres(A, b, x, 50)


if __name__ == '__main__':
    main()
