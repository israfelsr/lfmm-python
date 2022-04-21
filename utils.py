# common imports
import numpy as np
import scipy

def center_columns(x):
    return (x-np.expand_dims(np.mean(x, axis=0),0))

def compute_pmatrix(x, lambda_value):
    n, d = x.shape
    svd = np.linalg.svd(x)
    d_lambda = np.ones(n)
    d_lambda[:d] = np.sqrt(lambda_value / (lambda_value + svd[1]))
    D = np.diag(d_lambda)
    D_inv = np.diag(1/d_lambda)
    sqrt_p = np.dot(D, svd[0].T)
    sqrt_p_inv = np.dot(svd[0], D_inv)
    return (sqrt_p, sqrt_p_inv)

def compute_svds(P, Y, k):
    res_svds = scipy.sparse.linalg.svds(np.dot(P,Y), k=k, tol=10e-10)
    u = np.flip(res_svds[0], axis=1)
    d = np.flip(res_svds[1])
    v = np.flip(res_svds[2].T, axis=1)
    return (u, d, v)

def compute_B_ridge(A, X, lambda_value):
    D = np.diag(np.ones(X.shape[1]))
    B = np.linalg.solve(np.dot(X.T, X) + lambda_value * D, A).T
    return B

def sum2_lm(Y, X, B):
    n = Y.shape[0]
    p = Y.shape[1]
    err2 = np.zeros(p)
    aux = 0.0
    for j in range(p):
        for i in range(n):
            aux = Y[i,j] - np.dot(X[i,:], B[j,:])
            err2[j] += aux ** 2 
    return err2

def compute_pvalue_from_tscore(score, df):
    return 2 * (1 - scipy.stats.t.cdf(abs(score),df=df))

def compute_pvalue_from_zscore2(score2, df=1):
    return 1 - scipy.stats.chi2.cdf(score2, df=df)

def compute_gif(score):
    return np.nanmedian(score**2) / scipy.stats.chi2.ppf(0.5,df=1)