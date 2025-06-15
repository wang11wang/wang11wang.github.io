---
title: sklearn设置环境降低cpu负载,提升计算速度
date: 2024-12-21
---
当使用scikit-learn中的算法的时候，会涉及到并行计算，以加快计算速度；
scikit-learn的并行有几种实现
- 高层次的并行: [joblib](https://joblib.readthedocs.io/en/latest/)
- 调用C或者Cython实现的并行，使用OpenMP
- 调用Numpy或者SciPy的array操作，使用BLAS

## joblib
scikit-learn中如果使用joblib进行多线程或者多进程加速，默认使用多进程，在某些情况下使用多线程进行加速； 那么一般会有n_jobs的参数，默认是None 使用1个进程进行计算，设置n_jobs=-1会使用全部的进程进行计算；

如果要手动指定使用多线程计算
``` python
from joblib import parallel_backend

with parallel_backend('threading', n_jobs=2):
    # Your scikit-learn code here
```

## OpenMP
OpenMP会默认使用尽可能多的线程进行计算，通常和CPU的逻辑核数相等的核数进行计算，可以通过<font color="red">OMP_NUM_THREADS</font>环境变量进行控制,如
``` bash
$ OMP_NUM_THREADS=4 python my_script.py
```
或者
``` bash
$ export OMP_NUM_THREADS=4
python my_script.py
```

## 来自numpy,scipy的并行
scikit-learn依赖了很多numpy和scipy的计算，numpy和scipy内部使用线性代数库(BLAS & LAPACK)进行并行计算，比如： MKL, OpenBLAS or BLIS
- MKL_NUM_THREADS 设置MKL的线程个数
- OPENBLAS_NUM_THREADS 设置OpenBLAS的线程个数
- BLIS_NUM_THREADS 设置BLIS的线程个数

需要注意的是，OpenMP也可能会影响BLAS & LAPACK的实现，可以通过以下命令进行检查：
``` bash
$ OMP_NUM_THREADS=2 python -m threadpoolctl -i numpy scipy
```

## spawning过多的线程
一般情况下，使用和cpu核心数相当的线程数是合适的，使用更多的核心数不会提升性能，性能反而还会降低

假设你有1个8核CPU，假设同时使用HistGradientBoostingClassifier(由OpenMP并行)和GridSearchCV（由joblib并行） with n_jobs=8。那么每个假设你有1个8核CPU，假设同时使用HistGradientBoostingClassifier实例会spawn8个线程，最终会有8 * 8 = 64个线程，会导致线程数远超过物理核心数。

线程超额也可能会是joblib内嵌套MKL、 OpenBLAS 或 BLIS导致的。

自从joblib >= 0.14开始，默认使用loky作为joblib的backend，joblib会告诉子进程限制线程数，以避免超额；

## Best Practice
直接设置以下的环境，不要依赖于joblib的默认设置或者自动设置：
``` bash
$ export OMP_NUM_THREADS=1
$ export MKL_NUM_THREADS=1
$ export OPENBLAS_NUM_THREADS=1
$ export BLIS_NUM_THREADS=1
```
