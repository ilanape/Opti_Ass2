import numpy as np
from scipy.sparse import random
import scipy.sparse as sparse
import matplotlib.pyplot as plt


def jacobi(A, b, x, w, n):
    # for plot
    ylineRes, ylineCon = [], []

    # dominant diagonal
    D = np.diagflat(np.diag(A))

    for i in range(n):
        # for residual plot
        curr_res_norm = np.linalg.norm(A @ x - b)
        ylineRes.append(curr_res_norm)

        # guess vector update
        x = x + w * np.linalg.inv(D) @ (b - A @ x)

        # for convergence factor plot
        new_res_norm = np.linalg.norm(A @ x - b)
        ylineCon.append(new_res_norm / curr_res_norm)
    return x, ylineRes, ylineCon


def gauss_seidel(A, b, x, n):
    # for plot
    ylineRes, ylineCon = [], []

    # lower triangular + diagonal
    D = np.diagflat(np.diag(A))
    L = np.tril(A)

    for i in range(n):
        # for residual plot
        curr_res_norm = np.linalg.norm(A @ x - b)
        ylineRes.append(curr_res_norm)

        # guess vector update
        x = x + np.linalg.inv(L + D) @ (b - A @ x)

        # for convergence factor plot
        new_res_norm = np.linalg.norm(A @ x - b)
        ylineCon.append(new_res_norm / curr_res_norm)
    return x, ylineRes, ylineCon


def SD(A, b, x, n):
    # for plot
    ylineRes, ylineCon = [], []

    r = b - A @ x
    for i in range(n):
        # for residual plot
        curr_res_norm = np.linalg.norm(A @ x - b)
        ylineRes.append(curr_res_norm)

        # guess vector update
        alpha = np.dot(r, r) / np.dot(r, A @ r)
        x = x + alpha * r
        r = r - alpha * A @ r

        # for convergence factor plot
        new_res_norm = np.linalg.norm(A @ x - b)
        ylineCon.append(new_res_norm / curr_res_norm)
    return x, ylineRes, ylineCon


def CG(A, b, x, n):
    # for plot
    ylineRes, ylineCon = [], []

    r = b - A @ x
    p = r
    for i in range(n):
        # for residual plot
        curr_res_norm = np.linalg.norm(A @ x - b)
        ylineRes.append(curr_res_norm)

        # guess vector update
        alpha = np.dot(r, r) / np.dot(p, A @ p)
        x = x + alpha * p
        r = r - alpha * A @ p
        beta = - np.dot(r, A @ p) / np.dot(p, A @ p)
        p = r + beta * p

        # for convergence factor plot
        new_res_norm = np.linalg.norm(A @ x - b)
        ylineCon.append(new_res_norm / curr_res_norm)
    return x, ylineRes, ylineCon


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


# create A sparse array
n = 256
A = random(n, n, 5 / n, dtype=float)
v = np.random.rand(n)
v = sparse.spdiags(v, 0, v.shape[0], v.shape[0], 'csr')
A = A.transpose() * v * A + 0.1 * sparse.eye(n)
A = A.toarray()

b = np.random.rand(n)
x = np.zeros(n)

# iterations x line
xline = [*list(range(0, 100))]

x, ylineRes, ylineCon = jacobi(A, b, x, 1, 100)
create_plots('Standart Jacobi method', xline, ylineRes, ylineCon)

x, ylineRes, ylineCon = jacobi(A, b, x, 0.1, 100)
create_plots('Jacobi method with weight = 0.1', xline, ylineRes, ylineCon)

x, ylineRes, ylineCon = gauss_seidel(A, b, x, 100)
create_plots('Gauss-Seidel method', xline, ylineRes, ylineCon)

x, ylineRes, ylineCon = SD(A, b, x, 100)
create_plots('Steepest Descent method', xline, ylineRes, ylineCon)

x, ylineRes, ylineCon = CG(A, b, x, 100)
create_plots('Conjugate Gradient Jacobi method', xline, ylineRes, ylineCon)
