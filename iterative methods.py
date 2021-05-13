import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def jacobi(A, b, x, w, n):
    # for plot
    xline, ylineRes, ylineCon = [], [], []

    # # dominant diagonal
    # D = np.diagflat(np.diag(A))

    # block preconditioner
    M1 = A[0:3, 0:3]
    M2 = A[3:10, 3:10]
    M_inv = np.block([[np.linalg.inv(M1), np.zeros((3, 7))], [np.zeros((7, 3)), np.linalg.inv(M2)]])

    # block preconditioner
    # M1 = A[0:3, 0:3]
    # M2 = A[3:7, 3:7]
    # M3 = A[7:10, 7:10]
    # M_inv = np.block([[np.linalg.inv(M1), np.zeros((3, 4)), np.zeros((3, 3))],
    #                   [np.zeros((4, 3)), np.linalg.inv(M2), np.zeros((4, 3))],
    #                   [np.zeros((3, 3)), np.zeros((3, 4)), np.linalg.inv(M3)]])

    for i in range(n):
        # for residual plot
        xline.append(i)
        curr_res_norm = np.linalg.norm(A @ x - b)
        ylineRes.append(curr_res_norm)

        # apply iteration
        # x = x + w * np.linalg.inv(D) @ (b - A @ x)
        x = x + w * M_inv @ (b - A @ x)

        # for convergence factor plot
        new_res_norm = np.linalg.norm(A @ x - b)
        ylineCon.append(new_res_norm / curr_res_norm)

        # Convergence criterion
        # if (new_res_norm/ np.linalg.norm(b)) < 1*pow(10, -5):
        #     break

        # five digits accuracy
        if new_res_norm < 1 * pow(10, -5):
            print(i)
            break

    return x, xline, ylineRes, ylineCon


def gauss_seidel(A, b, x, n):
    # for plot
    xline, ylineRes, ylineCon = [], [], []

    # lower triangular matrix L+D creation
    D = np.diagflat(np.diag(A))
    L = np.tril(A)

    for i in range(n):
        # for residual plot
        xline.append(i)
        curr_res_norm = np.linalg.norm(A @ x - b)
        ylineRes.append(curr_res_norm)

        # apply iteration
        x = x + np.linalg.inv(L + D) @ (b - A @ x)

        # for convergence factor plot
        new_res_norm = np.linalg.norm(A @ x - b)
        ylineCon.append(new_res_norm / curr_res_norm)

        # Convergence criterion
        if (new_res_norm / np.linalg.norm(b)) < 0.1:
            break

    return x, xline, ylineRes, ylineCon


def SD(A, b, x, n):
    # for plot
    xline, ylineRes, ylineCon = [], [], []

    r = b - A @ x
    for i in range(n):
        # for residual plot
        xline.append(i)
        curr_res_norm = np.linalg.norm(r)
        ylineRes.append(curr_res_norm)

        # apply iteration
        # weight
        Ar = A @ r
        alpha = np.dot(r, r) / np.dot(r, Ar)
        x = x + alpha * r
        r = r - alpha * Ar

        # for convergence factor plot
        new_res_norm = np.linalg.norm(r)
        ylineCon.append(new_res_norm / curr_res_norm)

        # Convergence criterion
        if (new_res_norm / np.linalg.norm(b)) < 0.1:
            break

    return x, xline, ylineRes, ylineCon


def CG(A, b, x, n):
    # for plot
    xline, ylineRes, ylineCon = [], [], []

    r = b - A @ x
    p = r
    for i in range(n):
        # for residual plot
        xline.append(i)
        curr_res_norm = np.linalg.norm(r)
        ylineRes.append(curr_res_norm)

        # apply iteration
        Ap = A @ p
        alpha = np.dot(r, r) / np.dot(p, Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        # for convergence factor plot
        new_res_norm = np.linalg.norm(r)
        ylineCon.append(new_res_norm / curr_res_norm)

        # Convergence criterion
        if (new_res_norm / np.linalg.norm(b)) < 0.1:
            break

        beta = - np.dot(r, A @ p) / np.dot(p, A @ p)
        p = r + beta * p

    return x, xline, ylineRes, ylineCon


def create_plots(title, xline, ylineRes, ylineCon):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

    ax1.set_title('convergence graph of residual')
    ax1.set_xlabel('iterations')
    ax1.set_ylabel('residual')
    ax1.semilogy(xline, ylineRes)

    ax2.set_title('convergence factor graph')
    ax2.set_xlabel('iterations')
    ax2.set_ylabel('convergence factor')
    ax2.semilogy(xline, ylineCon)

    plt.suptitle(title)
    plt.show()


def main():
    # # create A sparse array
    # n = 256
    # A = random(n, n, 5 / n, dtype=float)
    # v = np.random.rand(n)
    # v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
    # A = A.transpose() * v * A + 0.1 * sparse.eye(n)
    # A = A.toarray()
    #
    # b = np.random.rand(n)
    # x = np.zeros(n)
    #
    # x, xline, ylineRes, ylineCon = jacobi(A, b, x, 1, 100)
    # create_plots('Standard Jacobi method', xline, ylineRes, ylineCon)
    #
    # x, xline, ylineRes, ylineCon = jacobi(A, b, x, 0.1, 100)
    # create_plots('Jacobi method with weight = 0.1', xline, ylineRes, ylineCon)
    #
    # x, xline, ylineRes, ylineCon = gauss_seidel(A, b, x, 100)
    # create_plots('Gauss-Seidel method', xline, ylineRes, ylineCon)
    #
    # x, xline, ylineRes, ylineCon = SD(A, b, x, 100)
    # create_plots('Steepest Descent method', xline, ylineRes, ylineCon)
    #
    # x, xline, ylineRes, ylineCon = CG(A, b, x, 100)
    # create_plots('Conjugate Gradient Jacobi method', xline, ylineRes, ylineCon)

    L = np.array([[2, -1, -1, 0, 0, 0, 0, 0, 0, 0], [-1, 2, -1, 0, 0, 0, 0, 0, 0, 0],
                  [-1, -1, 3, -1, 0, 0, 0, 0, 0, 0], [0, 0, -1, 5, -1, 0, -1, 0, -1, -1],
                  [0, 0, 0, -1, 4, -1, -1, -1, 0, 0], [0, 0, 0, 0, -1, 3, -1, -1, 0, 0],
                  [0, 0, 0, -1, -1, -1, 5, -1, 0, -1], [0, 0, 0, 0, -1, -1, -1, 4, 0, -1],
                  [0, 0, 0, -1, 0, 0, 0, 0, 2, -1], [0, 0, 0, -1, 0, 0, -1, -1, -1, 4]])

    b = np.array([1, -1, 1, -1, 1, -1, 1, -1, 1, -1])
    x = np.zeros(10)

    # x, xline, ylineRes, ylineCon = jacobi(L, b, x, 1, 100)
    # create_plots('Standard Jacobi method', xline, ylineRes, ylineCon)
    # print(x)

    # rows and columns swap
    # rows 4 8 swap
    # temp = L[3, :].copy()
    # L[3, :] = L[7, :]
    # L[7, :] = temp
    #
    # # columns 4 8 swap
    # temp = L[:, 3].copy()
    # L[:, 3] = L[:, 7]
    # L[:, 7] = temp

    x, xline, ylineRes, ylineCon = jacobi(L, b, x, 0.65, 100)
    create_plots('Weighted Jacobi method', xline, ylineRes, ylineCon)
    print(x)


if __name__ == '__main__':
    main()
