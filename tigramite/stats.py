# from __future__ import division, print_function, absolute_import

import os
import warnings
import sys
import numpy as np

from scipy import linalg
import scipy.special as special

import pycuda
import pycuda.autoinit
import pycuda.compiler
from pycuda import gpuarray
from pycuda import driver
from skcuda import cublas

# #import math
# if sys.version_info.major >= 3 and sys.version_info.minor >= 5:
#     from math import gcd
# else:
#     from fractions import gcd

# __all__ = ['find_repeats', 'gmean', 'hmean', 'mode', 'tmean', 'tvar',
#            'tmin', 'tmax', 'tstd', 'tsem', 'moment', 'variation',
#            'skew', 'kurtosis', 'describe', 'skewtest', 'kurtosistest',
#            'normaltest', 'jarque_bera', 'itemfreq',
#            'scoreatpercentile', 'percentileofscore',
#            'cumfreq', 'relfreq', 'obrientransform',
#            'sem', 'zmap', 'zscore', 'iqr', 'gstd', 'median_absolute_deviation',
#            'sigmaclip', 'trimboth', 'trim1', 'trim_mean', 'f_oneway',
#            'PearsonRConstantInputWarning', 'PearsonRNearConstantInputWarning',
#            'pearsonr', 'fisher_exact', 'spearmanr', 'pointbiserialr',
#            'kendalltau', 'weightedtau',
#            'linregress', 'siegelslopes', 'theilslopes', 'ttest_1samp',
#            'ttest_ind', 'ttest_ind_from_stats', 'ttest_rel', 'kstest',
#            'chisquare', 'power_divergence', 'ks_2samp', 'mannwhitneyu',
#            'tiecorrect', 'ranksums', 'kruskal', 'friedmanchisquare',
#            'rankdata', 'rvs_ratio_uniforms',
#            'combine_pvalues', 'wasserstein_distance', 'energy_distance',
#            'brunnermunzel', 'epps_singleton_2samp']



# def pearsonr(x, y):
#     r"""
#     Pearson correlation coefficient and p-value for testing non-correlation.

#     The Pearson correlation coefficient [1]_ measures the linear relationship
#     between two datasets.  The calculation of the p-value relies on the
#     assumption that each dataset is normally distributed.  (See Kowalski [3]_
#     for a discussion of the effects of non-normality of the input on the
#     distribution of the correlation coefficient.)  Like other correlation
#     coefficients, this one varies between -1 and +1 with 0 implying no
#     correlation. Correlations of -1 or +1 imply an exact linear relationship.
#     Positive correlations imply that as x increases, so does y. Negative
#     correlations imply that as x increases, y decreases.

#     The p-value roughly indicates the probability of an uncorrelated system
#     producing datasets that have a Pearson correlation at least as extreme
#     as the one computed from these datasets.

#     Parameters
#     ----------
#     x : (N,) array_like
#         Input
#     y : (N,) array_like
#         Input

#     Returns
#     -------
#     r : float
#         Pearson's correlation coefficient
#     p-value : float
#         two-tailed p-value

#     Warns
#     -----
#     PearsonRConstantInputWarning
#         Raised if an input is a constant array.  The correlation coefficient
#         is not defined in this case, so ``np.nan`` is returned.

#     PearsonRNearConstantInputWarning
#         Raised if an input is "nearly" constant.  The array ``x`` is considered
#         nearly constant if ``norm(x - mean(x)) < 1e-13 * abs(mean(x))``.
#         Numerical errors in the calculation ``x - mean(x)`` in this case might
#         result in an inaccurate calculation of r.

#     See Also
#     --------
#     spearmanr : Spearman rank-order correlation coefficient.
#     kendalltau : Kendall's tau, a correlation measure for ordinal data.

#     Notes
#     -----

#     The correlation coefficient is calculated as follows:

#     .. math::

#         r = \frac{\sum (x - m_x) (y - m_y)}
#                  {\sqrt{\sum (x - m_x)^2 \sum (y - m_y)^2}}

#     where :math:`m_x` is the mean of the vector :math:`x` and :math:`m_y` is
#     the mean of the vector :math:`y`.

#     Under the assumption that x and y are drawn from independent normal
#     distributions (so the population correlation coefficient is 0), the
#     probability density function of the sample correlation coefficient r
#     is ([1]_, [2]_)::

#                (1 - r**2)**(n/2 - 2)
#         f(r) = ---------------------
#                   B(1/2, n/2 - 1)

#     where n is the number of samples, and B is the beta function.  This
#     is sometimes referred to as the exact distribution of r.  This is
#     the distribution that is used in `pearsonr` to compute the p-value.
#     The distribution is a beta distribution on the interval [-1, 1],
#     with equal shape parameters a = b = n/2 - 1.  In terms of SciPy's
#     implementation of the beta distribution, the distribution of r is::

#         dist = scipy.stats.beta(n/2 - 1, n/2 - 1, loc=-1, scale=2)

#     The p-value returned by `pearsonr` is a two-sided p-value.  For a
#     given sample with correlation coefficient r, the p-value is
#     the probability that abs(r') of a random sample x' and y' drawn from
#     the population with zero correlation would be greater than or equal
#     to abs(r).  In terms of the object ``dist`` shown above, the p-value
#     for a given r and length n can be computed as::

#         p = 2*dist.cdf(-abs(r))

#     When n is 2, the above continuous distribution is not well-defined.
#     One can interpret the limit of the beta distribution as the shape
#     parameters a and b approach a = b = 0 as a discrete distribution with
#     equal probability masses at r = 1 and r = -1.  More directly, one
#     can observe that, given the data x = [x1, x2] and y = [y1, y2], and
#     assuming x1 != x2 and y1 != y2, the only possible values for r are 1
#     and -1.  Because abs(r') for any sample x' and y' with length 2 will
#     be 1, the two-sided p-value for a sample of length 2 is always 1.

#     References
#     ----------
#     .. [1] "Pearson correlation coefficient", Wikipedia,
#            https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
#     .. [2] Student, "Probable error of a correlation coefficient",
#            Biometrika, Volume 6, Issue 2-3, 1 September 1908, pp. 302-310.
#     .. [3] C. J. Kowalski, "On the Effects of Non-Normality on the Distribution
#            of the Sample Product-Moment Correlation Coefficient"
#            Journal of the Royal Statistical Society. Series C (Applied
#            Statistics), Vol. 21, No. 1 (1972), pp. 1-12.

#     Examples
#     --------
#     >>> from scipy import stats
#     >>> a = np.array([0, 0, 0, 1, 1, 1, 1])
#     >>> b = np.arange(7)
#     >>> stats.pearsonr(a, b)
#     (0.8660254037844386, 0.011724811003954649)

#     >>> stats.pearsonr([1, 2, 3, 4, 5], [10, 9, 2.5, 6, 4])
#     (-0.7426106572325057, 0.1505558088534455)

#     """
#     n = len(x)
#     if n != len(y):
#         raise ValueError('x and y must have the same length.')

#     if n < 2:
#         raise ValueError('x and y must have length at least 2.')

#     x = np.asarray(x)
#     y = np.asarray(y)

#     # If an input is constant, the correlation coefficient is not defined.
#     if (x == x[0]).all() or (y == y[0]).all():
#         warnings.warn(PearsonRConstantInputWarning())
#         return np.nan, np.nan

#     # dtype is the data type for the calculations.  This expression ensures
#     # that the data type is at least 64 bit floating point.  It might have
#     # more precision if the input is, for example, np.longdouble.
#     dtype = type(1.0 + x[0] + y[0])

#     if n == 2:
#         return dtype(np.sign(x[1] - x[0])*np.sign(y[1] - y[0])), 1.0

#     xmean = x.mean(dtype=dtype)
#     ymean = y.mean(dtype=dtype)

#     # By using `astype(dtype)`, we ensure that the intermediate calculations
#     # use at least 64 bit floating point.
#     xm = x.astype(dtype) - xmean
#     ym = y.astype(dtype) - ymean

#     # Unlike np.linalg.norm or the expression sqrt((xm*xm).sum()),
#     # scipy.linalg.norm(xm) does not overflow if xm is, for example,
#     # [-5e210, 5e210, 3e200, -3e200]
#     normxm = linalg.norm(xm)
#     normym = linalg.norm(ym)

#     threshold = 1e-13
#     if normxm < threshold*abs(xmean) or normym < threshold*abs(ymean):
#         # If all the values in x (likewise y) are very close to the mean,
#         # the loss of precision that occurs in the subtraction xm = x - xmean
#         # might result in large errors in r.
#         warnings.warn(PearsonRNearConstantInputWarning())

#     # print(xm/normxm)
#     # print(ym/normym)

#     r = np.dot(xm/normxm, ym/normym)

#     # Presumably, if abs(r) > 1, then it is only some small artifact of
#     # floating point arithmetic.
#     r = max(min(r, 1.0), -1.0)

#     # As explained in the docstring, the p-value can be computed as
#     #     p = 2*dist.cdf(-abs(r))
#     # where dist is the beta distribution on [-1, 1] with shape parameters
#     # a = b = n/2 - 1.  `special.btdtr` is the CDF for the beta distribution
#     # on [0, 1].  To use it, we make the transformation  x = (r + 1)/2; the
#     # shape parameters do not change.  Then -abs(r) used in `cdf(-abs(r))`
#     # becomes x = (-abs(r) + 1)/2 = 0.5*(1 - abs(r)).  (r is cast to float64
#     # to avoid a TypeError raised by btdtr when r is higher precision.)
#     ab = n/2 - 1
#     prob = 2*special.btdtr(ab, ab, 0.5*(1 - abs(np.float64(r))))

#     return r, prob

###############################################################################
def _pearson_cuda_2dim(X, Y):
    n = len(X)
    m = len(Y)
    p = len(X[0])

    # standardize
    # X -= X.mean(axis=1).reshape(n, 1)
    # X /= X.std(axis=1).reshape(n, 1)
    # Y -= Y.mean(axis=1).reshape(m, 1)
    # Y /= Y.std(axis=1).reshape(m, 1)
    XT = X - X.mean(axis=1).reshape(n, 1)
    XT /= X.std(axis=1).reshape(n, 1)
    YT = Y - Y.mean(axis=1).reshape(m, 1)
    YT /= Y.std(axis=1).reshape(m, 1)

    # Create a zero-initialized chunk of memory for the per-thread buckets and copy it to the GPU.
    matrix = np.zeros(shape=(n, m), dtype=np.float64, order='F')    # (n, m) == (10, 20)
    v = np.zeros(shape=(n), dtype=np.float64, order='F')         # (n, 1) == (10,)
    vector_gpu = gpuarray.to_gpu(v)

    # set up values for CuBLAS context
    alpha = 1.0 / p  # Why??????????????????????
    beta = 0.0
    trans = cublas._CUBLAS_OP['N']
    lda = n
    incx = 1
    incy = 1

    # Prepare input matrices for GPU.
    # print(f'shape(A)={XT.shape}')
    # print(f'shape(x)={YT.shape}')
    # print(f'shape(b)={v.shape}')
    # print(f'A=XT={XT}')
    XT = XT.T.copy()
    A_gpu = gpuarray.to_gpu(XT)
    # print(f'A_gpu={A_gpu}')
    # print(f'trans={trans}')
    # print(f'n={n}')
    # print(f'p={p}')
    # print(f'alpha={alpha}')
    # print(f'lda={lda}')
    # print(f'beta={beta}')
    # print(f'incx={incx}')
    # print(f'incy={incy}')
 
    handle = cublas.cublasCreate()
    for j in range(m):
        x = YT[j].copy()
        # print(f'x=YT[{j}]={x}')
        x_gpu = gpuarray.to_gpu(x)
        # print(f'x_gpu={x_gpu}')
        # print(f'shape(v)={v.shape}')
        # print(f'v={v}')
        cublas.cublasDgemv(handle, trans, n, p,
                           alpha, A_gpu.gpudata, lda,
                           x_gpu.gpudata, incx,
                           beta, vector_gpu.gpudata, incy)
        vector_gpu.get(v)
        # print(f'v={v}')
        matrix[:, j] = v

    #     progress = i * 100.0 / m
    #     sys.stdout.write('\rComputing correlations %.2f%% ' % progress)
    #     sys.stdout.flush()

    # print('\rComputing correlations 100.000%          ')

        # print(v.shape)
    # print(matrix.shape)
    cublas.cublasDestroy(handle)

    return matrix#, merged

###############################################################################
def pearson_cuda(X, Y):
    # Check some preconditions to verify we're doing something sensical.
    # Doesn't cover all cases, but catches obvious mistakes.
    assert X.ndim == 2, 'X should be matrix.'
    if Y.ndim == 2:
 #       assert len(X[0]) == len(X[1]), 'Your sequences in X should all be the same length.'
        assert len(Y[0]) == len(Y[1]), 'Your sequences in Y should all be the same length.'
        assert len(X[0]) == len(Y[0]), 'Your sequences in X and Y should all be the same length.'

        matrix = _pearson_cuda_2dim(X, Y)

    elif Y.ndim == 1:
        # assert len(X[0]) == len(X[1]), 'Your sequences in X should all be the same length.'
        assert len(X[0]) == len(Y), 'Your sequences in X and Y should all be the same length.'
        p = len(X[0])

        matrix = _pearson_cuda_2dim(X, Y.reshape(1, p))

    else:
        raise(ValueError("Y.ndim must be 1 or 2"))
    
    return matrix
    