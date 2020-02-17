# from __future__ import division, print_function, absolute_import

import os
import warnings
import sys
import numpy as np

from scipy import linalg
import scipy.special as special

import arrayfire as af

import pycuda
import pycuda.autoinit
import pycuda.compiler
from pycuda import gpuarray
from pycuda import driver
from skcuda import cublas

###############################################################################
def compute_af(X, Y): #, ranges
    # # Check some preconditions to verify we're doing something sensical.
    # # Doesn't cover all cases, but catches obvious mistakes.
    # assert len(X[0]) == len(X[1]), 'Your sequences in X should all be the same length.'
    # assert len(Y[0]) == len(Y[1]), 'Your sequences in Y should all be the same length.'
    # assert len(X[0]) == len(Y[0]), 'Your sequences in X and Y should all be the same length.'

    n = len(X)
    m = len(Y)
    X_af = af.Array(X.ctypes.data, X.T.shape, X.dtype.char)
    Y_af = af.Array(Y.ctypes.data, Y.T.shape, Y.dtype.char)

    # Create the correlation matrix and buckets.
    matrix = np.zeros(shape=(n, m), dtype=np.float64, order='C')#

    for i in range(n):
        for j in range(m):
            # Compute the coefficient.
            coeff = af.corrcoef(X_af[:,i], Y_af[:,j])
            matrix[i][j] = coeff

    return matrix
    
def _pearson(X, Y):
    n = len(X)
    m = len(Y)
    p = len(X[0])

    # Load the kernel and compile it.
    kernel_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pearson.cu')
    f = open(kernel_file, 'r')
    kernel = pycuda.compiler.SourceModule(f.read())
    f.close()
    pearson_cuda = kernel.get_function('pearson')
    # reduction_cuda = kernel.get_function('reduction')

    # CUDA parameters that seem to work well. The number of threads per tile
    # (the tile_size) should be a power of 2 for the parallel reduction to
    # work right!
    threads_per_block = 16 #64t/b*64b/g = 4096t/g
    blocks_per_tile = 64 #max 4096b/g with 16t/b
    tile_size = threads_per_block * blocks_per_tile #1024t/g
    num_tiles = (n // tile_size + 1, m // tile_size + 1)

    # # Copy the ranges into a numpy array.
    # ranges_np = np.zeros(shape=(len(ranges), 2), dtype=np.float32, order='C')#
    # for i in range(len(ranges)):
    #     np.put(ranges_np[i], range(2), ranges[i])

    # Create a zero-initialized chunk of memory for the per-thread buckets and
    # copy it to the GPU.
    matrix = np.zeros(shape=(n, m), dtype=np.float64, order='C')
    matrix_gpu = gpuarray.to_gpu(matrix)
    # buckets = np.zeros(shape=(tile_size * tile_size * len(ranges), 1), dtype=np.uint64, order='C')
    # buckets_gpu = gpuarray.to_gpu(buckets)

    # Do a kernel launch for each tile, copying the appropriate chunks of the
    # input arrays into X and Y for each launch.
    for s in range(num_tiles[0]):
        for t in range(num_tiles[1]):
            num_A = tile_size
            remain_X = n - (s * tile_size)
            num_A = num_A if num_A < remain_X else remain_X

            A = np.zeros(shape=(num_A, p), dtype=np.float64, order='C')#
            for i in range(num_A):
                np.put(A[i], range(p), X[(s * tile_size) + i])

            num_B = tile_size
            remain_Y = m - (t * tile_size)
            num_B = num_B if num_B < remain_Y else remain_Y

            B = np.zeros(shape=(num_B, p), dtype=np.float64, order='C')#
            for j in range(num_B):
                np.put(B[j], range(p), Y[(t * tile_size) + j])

            pearson_cuda(matrix_gpu.gpudata, # buckets_gpu.gpudata,
                        #  driver.In(ranges_np), np.uint32(len(ranges)),
                         driver.In(A), driver.In(B),
                         np.uint32(tile_size), np.uint32(s), np.uint32(t),
                         np.uint32(n), np.uint32(m), np.uint32(p),
                         block=(threads_per_block, threads_per_block, 1),
                         grid=(blocks_per_tile, blocks_per_tile))

    #         progress = (s * num_tiles[1] + t) * 100.0 / (num_tiles[0] * num_tiles[1])
    #         sys.stdout.write('\rComputing correlations %.2f%% (s=%d, t=%d)' % (progress, s, t))
    #         sys.stdout.flush()

    # print('\rComputing correlations 100.000%          ')
    # sys.stdout.write('Merging buckets... ')
    # sys.stdout.flush()

    # Copy buckets back from GPU.
    matrix_gpu.get(matrix)

    return matrix

###############################################################################
def _pearson_cuda(X, Y):
    n = len(X)
    m = len(Y)
    p = len(X[0])

    # standardize
    # threshold = 1e-13
    Xmean = X.mean(axis=1)
    Xm = X - Xmean.reshape(n, 1)
    normXm = linalg.norm(Xm, axis=1)
    # if normXm < threshold*abs(Xmean[0]):
    #     warnings.warn(PearsonRNearConstantInputWarning())
    # if normXm != 0:
    Xm /= normXm.reshape(n, 1)
    # else:
    #     Xm *= 0
    Ymean = Y.mean(axis=1)
    Ym = Y - Ymean.reshape(m, 1)
    normYm = linalg.norm(Ym, axis=1)
    # if normYm < threshold*abs(Ymean[0]):
    #     warnings.warn(PearsonRNearConstantInputWarning())
    # if normYm != 0:
    Ym /= normYm.reshape(m, 1)
    # else:
    #     Ym *= 0

    # Create a zero-initialized chunk of memory for the per-thread buckets and copy it to the GPU.
    matrix = np.zeros(shape=(n, m), dtype=np.float64, order='F')
    v = np.zeros(shape=(n), dtype=np.float64, order='F')
    vector_gpu = gpuarray.to_gpu(v)

    # set up values for CuBLAS context
    alpha = 1.0
    beta = 0.0
    trans = cublas._CUBLAS_OP['N']
    lda = n
    incx = 1
    incy = 1

    # Prepare input matrices for GPU.
    XT = Xm.T.copy()
    A_gpu = gpuarray.to_gpu(XT)
 
    handle = cublas.cublasCreate()
    for j in range(m):
        x = Ym[j].copy()
        x_gpu = gpuarray.to_gpu(x)
        cublas.cublasDgemv(handle, trans, n, p,
                           alpha, A_gpu.gpudata, lda,
                           x_gpu.gpudata, incx,
                           beta, vector_gpu.gpudata, incy)
        vector_gpu.get(v)
        matrix[:, j] = v

    cublas.cublasDestroy(handle)

    return matrix

###############################################################################
def pearson_cuda(X, Y, mode='none'):
    # Check some preconditions to verify we're doing something sensical.
    # Doesn't cover all cases, but catches obvious mistakes.
    if X.ndim == 1:
        p = len(X)
        X = X.reshape(1, p)
    if Y.ndim == 1:
        p = len(Y)
        Y = Y.reshape(1, p)

    # if Y.ndim == 2:
        # assert len(X[0]) == len(X[1]), 'Your sequences in X should all be the same length.'
    # assert len(Y[0]) == len(Y[1]), 'Your sequences in Y should all be the same length.'
    assert len(X[0]) == len(Y[0]), 'Your sequences in X and Y should all be the same length.'

    if mode=='cublasDgemm':
        matrix = _pearson_cublasDgemm(X, Y)
    elif mode=='cublasDgemv':
        matrix = _pearson_cublasDgemv(X, Y)
    else:
        matrix = _pearson(X, Y)

    # else:
    #     raise(ValueError("Y.ndim must be 1 or 2"))
    
    return matrix

###############################################################################
def _pearson_cublasDgemv(X, Y):
    n = len(X)
    m = len(Y)
    p = len(X[0])

    # standardize
    # threshold = 1e-13
    Xmean = X.mean(axis=1)
    Xm = X - Xmean.reshape(n, 1)
    normXm = linalg.norm(Xm, axis=1)
    # if normXm < threshold*abs(Xmean[0]):
    #     warnings.warn(PearsonRNearConstantInputWarning())
    # if normXm != 0:
    Xm /= normXm.reshape(n, 1)
    # else:
    #     Xm *= 0
    Ymean = Y.mean(axis=1)
    Ym = Y - Ymean.reshape(m, 1)
    normYm = linalg.norm(Ym, axis=1)
    # if normYm < threshold*abs(Ymean[0]):
    #     warnings.warn(PearsonRNearConstantInputWarning())
    # if normYm != 0:
    Ym /= normYm.reshape(m, 1)
    # else:
    #     Ym *= 0

    # Create a zero-initialized chunk of memory for the per-thread buckets and copy it to the GPU.
    matrix = np.zeros(shape=(n, m), dtype=np.float64, order='F')
    v = np.zeros(shape=(n), dtype=np.float64, order='F')
    vector_gpu = gpuarray.to_gpu(v)

    # set up values for CuBLAS context
    alpha = 1.0
    beta = 0.0
    trans = cublas._CUBLAS_OP['N']
    lda = n
    incx = 1
    incy = 1

    # Prepare input matrices for GPU.
    XT = Xm.T.copy()
    X_gpu = gpuarray.to_gpu(XT)

    handle = cublas.cublasCreate()
    for j in range(m):
        y = Ym[j].copy()
        y_gpu = gpuarray.to_gpu(y)
        cublas.cublasDgemv(handle, trans, n, p, 
                           alpha, X_gpu.gpudata, lda,
                           y_gpu.gpudata, incx,
                           beta, vector_gpu.gpudata, incy)
        vector_gpu.get(matrix[:, j])

    cublas.cublasDestroy(handle)

    return matrix

###############################################################################
def _pearson_cublasDgemm(X, Y):
    n = len(X)
    m = len(Y)
    p = len(X[0])

    # standardize
    # threshold = 1e-13
    Xmean = X.mean(axis=1)
    Xm = X - Xmean.reshape(n, 1)
    normXm = linalg.norm(Xm, axis=1)
    # if normXm < threshold*abs(Xmean[0]):
    #     warnings.warn(PearsonRNearConstantInputWarning())
    # if normXm != 0:
    Xm /= normXm.reshape(n, 1)
    # else:
    #     Xm *= 0
# XT /= X.std(axis=1).reshape(n, 1)
    Ymean = Y.mean(axis=1)
    Ym = Y - Ymean.reshape(m, 1)
    normYm = linalg.norm(Ym, axis=1)
    # if normYm < threshold*abs(Ymean[0]):
    #     warnings.warn(PearsonRNearConstantInputWarning())
    # if normYm != 0:
    Ym /= normYm.reshape(m, 1)
    # else:
    #     Ym *= 0
    # YT /= Y.std(axis=1).reshape(m, 1)

    # Create a zero-initialized chunk of memory for the per-thread buckets and copy it to the GPU.
    matrix = np.zeros(shape=(n, m), dtype=np.float64, order='F')
    matrix_gpu = gpuarray.to_gpu(matrix)

    # set up values for CuBLAS context
    alpha = 1.0
    beta = 0.0
    transa = cublas._CUBLAS_OP['N']
    transb = cublas._CUBLAS_OP['N']
    lda = n
    ldb = p
    ldc = n

    # Prepare input matrices for GPU.
    XT = Xm.T.copy()
    X_gpu = gpuarray.to_gpu(XT)
    Y_gpu = gpuarray.to_gpu(Ym)

    handle = cublas.cublasCreate()
    cublas.cublasDgemm(handle, transa, transb, n, m, p,
                       alpha, X_gpu.gpudata, lda,
                       Y_gpu.gpudata, ldb,
                       beta, matrix_gpu.gpudata, ldc)
    cublas.cublasDestroy(handle)
    matrix_gpu.get(matrix)

    return matrix
