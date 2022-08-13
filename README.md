# EigenRand : The Fastest C++11-compatible random distribution generator for Eigen

EigenRand is a header-only library for [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), providing vectorized random number engines and vectorized random distribution generators. 
Since the classic Random functions of Eigen relies on an old C function `rand()`, 
there is no way to control random numbers and no guarantee for quality of generated numbers. 
In addition, Eigen's Random is slow because `rand()` is hard to vectorize.

EigenRand provides a variety of random distribution functions similar to C++11 standard's random functions, 
which can be vectorized and easily integrated into Eigen's expressions of Matrix and Array.

You can get 5~10 times speed by just replacing old Eigen's Random or unvectorizable c++11 random number generators with EigenRand.

## Features

* C++11-compatible Random Number Generator
* 5~10 times faster than non-vectorized functions
* Header-only (like Eigen)
* Can be easily integrated with Eigen's expressions
* Currently supports only x86, x86-64(up to AVX2), and ARM64 NEON (experimental) architecture.

## Requirement

* Eigen 3.3.4 ~ 3.4.0
* C++11-compatible compilers

## Build for Test & Benchmark
You can build a test binary to verify if EigenRand is working well.
First, make sure you have Eigen 3.3.4~3.4.0 installed in your compiler include folder. Also make sure you have cmake 3.9 or higher installed.
After then, you can build it following:
```console
$ git clone https://github.com/bab2min/EigenRand
$ cd EigenRand
$ git clone https://github.com/google/googletest
$ pushd googletest && git checkout v1.8.x && popd
$ mkdir build && cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
$ ./test/EigenRand-test # Binary for unit test
$ ./EigenRand-accuracy # Binary for accuracy test of univariate random distributions
$ ./EigenRand-benchmark # Binary for performance test of univariate random distributions
$ ./EigenRand-benchmark-mv # Binary for performance test of multivariate random distributions
```

You can specify additional compiler arguments including target machine options (e.g. -mavx2, -march) like:
```console
$ cmake -DCMAKE_BUILD_TYPE=Release -DEIGENRAND_CXX_FLAGS="-march=native" ..
```

## Documentation

https://bab2min.github.io/eigenrand/

## Functions

### Random distributions for real types

| Function | Generator | Scalar Type | Description | Equivalent to |
|:---:|:---:|:---:|:---:|:---:|
| `Eigen::Rand::balanced` | `Eigen::Rand::BalancedGen` | float, double | generates real values in the [-1, 1] range | `Eigen::DenseBase<Ty>::Random` for floating point types |
| `Eigen::Rand::beta` | `Eigen::Rand::BetaGen` | float, double | generates real values on a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) |  |
| `Eigen::Rand::cauchy` | `Eigen::Rand::CauchyGen` | float, double | generates real values on the [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution). | `std::cauchy_distribution` |
| `Eigen::Rand::chiSquared` | `Eigen::Rand::ChiSquaredGen` | float, double | generates real values on a [chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution). | `std::chi_squared_distribution` |
| `Eigen::Rand::exponential` | `Eigen::Rand::ExponentialGen` | float, double | generates real values on an [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution). | `std::exponential_distribution` |
| `Eigen::Rand::extremeValue` | `Eigen::Rand::ExtremeValueGen` | float, double | generates real values on an [extreme value distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution). | `std::extreme_value_distribution` |
| `Eigen::Rand::fisherF` | `Eigen::Rand::FisherFGen` | float, double | generates real values on the [Fisher's F distribution](https://en.wikipedia.org/wiki/F_distribution). | `std::fisher_f_distribution` |
| `Eigen::Rand::gamma` | `Eigen::Rand::GammaGen` | float, double | generates real values on a [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution). | `std::gamma_distribution` |
| `Eigen::Rand::lognormal` | `Eigen::Rand::LognormalGen` | float, double | generates real values on a [lognormal distribution](https://en.wikipedia.org/wiki/Lognormal_distribution). | `std::lognormal_distribution` |
| `Eigen::Rand::normal` | `Eigen::Rand::StdNormalGen`, `Eigen::Rand::NormalGen` | float, double | generates real values on a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution). | `std::normal_distribution` |
| `Eigen::Rand::studentT` | `Eigen::Rand::StudentTGen` | float, double | generates real values on the [Student's t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution). | `std::student_t_distribution` |
| `Eigen::Rand::uniformReal` | `Eigen::Rand::UniformRealGen` | float, double | generates real values in the `[0, 1)` range. | `std::generate_canonical` |
| `Eigen::Rand::weibull` | `Eigen::Rand::WeibullGen` | float, double | generates real values on the [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution). | `std::weibull_distribution` |

### Random distributions for integer types

| Function | Generator | Scalar Type | Description | Equivalent to |
|:---:|:---:|:---:|:---:|:---:|
| `Eigen::Rand::binomial` | `Eigen::Rand::BinomialGen` | int | generates integers on a [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). | `std::binomial_distribution` |
| `Eigen::Rand::discrete` | `Eigen::Rand::DiscreteGen` | int | generates random integers on a discrete distribution. | `std::discrete_distribution` |
| `Eigen::Rand::geometric` | `Eigen::Rand::GeometricGen` | int | generates integers on a [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution). | `std::geometric_distribution` |
| `Eigen::Rand::negativeBinomial` | `Eigen::Rand::NegativeBinomialGen` | int | generates integers on a [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution). | `std::negative_binomial_distribution` |
| `Eigen::Rand::poisson` | `Eigen::Rand::PoissonGen` | int | generates integers on the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution). | `std::poisson_distribution` |
| `Eigen::Rand::randBits` | `Eigen::Rand::RandbitsGen` | int | generates integers with random bits. | `Eigen::DenseBase<Ty>::Random` for integer types |
| `Eigen::Rand::uniformInt` | `Eigen::Rand::UniformIntGen` | int | generates integers in the `[min, max]` range. | `std::uniform_int_distribution` |

### Multivariate distributions for real vectors and matrices

| Generator | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::MultinomialGen` | generates integer vectors on a [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution) | [scipy.stats.multinomial in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html#scipy.stats.multinomial) |
| `Eigen::Rand::DirichletGen` | generates real vectors on a [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) | [scipy.stats.dirichlet in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html#scipy.stats.dirichlet) |
| `Eigen::Rand::MvNormalGen` | generates real vectors on a [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | [scipy.stats.multivariate_normal in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal) |
| `Eigen::Rand::WishartGen` | generates real matrices on a [Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution) | [scipy.stats.wishart in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html#scipy.stats.wishart) |
| `Eigen::Rand::InvWishartGen` | generates real matrices on a [inverse Wishart distribution](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution) | [scipy.stats.invwishart in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html#scipy.stats.invwishart) |


### Random number engines

|  | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::Vmt19937_64` | a vectorized version of Mersenne Twister algorithm. It generates two 64bit random integers simultaneously with SSE2 & NEON and four integers with AVX2. | `std::mt19937_64` |
| `Eigen::Rand::P8_mt19937_64` | a vectorized version of Mersenne Twister algorithm. Since it generates eight 64bit random integers simultaneously, the random values are the same regardless of architecture. | |

## Performance
The following charts show the relative speed-up of EigenRand compared to references(equivalent functions of C++ std or Eigen).

![Perf_no_vect](/doxygen/images/perf_no_vect.png)
![Perf_no_vect](/doxygen/images/perf_sse2.png)
![Perf_no_vect](/doxygen/images/perf_avx.png)
![Perf_no_vect](/doxygen/images/perf_avx2.png)

The following charts are about multivariate distributions.
![Perf_no_vect](/doxygen/images/perf_mv_part1.png)
![Perf_no_vect](/doxygen/images/perf_mv_part2.png)


The following result is a measure of the time in seconds it takes to generate 1M random numbers. 
It shows the average of 20 times.

### Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz (Ubuntu 16.04, gcc5.4)

|  | C++ std (or Eigen) | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|---:|
| `balanced`* | 9.0 | 5.9 | 1.5 | 1.4 | 1.3 | 0.9 |
| `balanced`(double)* | 8.7 | 6.4 | 3.3 | 2.9 | 1.7 | 1.7 |
| `binomial(20, 0.5)` | 400.8 | 118.5 | 32.7 | 36.6 | 30.0 | 22.7 |
| `binomial(50, 0.01)` | 71.7 | 22.5 | 7.7 | 8.3 | 7.9 | 6.6 |
| `binomial(100, 0.75)` | 340.5 | 454.5 | 91.7 | 111.5 | 106.3 | 86.4 |
| `cauchy` | 36.1 | 54.4 | 6.1 | 7.1 | 4.7 | 3.9 |
| `chiSquared` | 80.5 | 249.5 | 64.6 | 58.0 | 29.4 | 28.8 |
| `discrete`(int32) | - | 14.0 | 2.9 | 2.6 | 2.4 | 1.7 |
| `discrete`(fp32) | - | 21.9 | 4.3 | 4.0 | 3.6 | 3.0 |
| `discrete`(fp64) | 72.4 | 21.4 | 6.9 | 6.5 | 4.9 | 3.7 |
| `exponential` | 31.0 | 25.3 | 5.5 | 5.3 | 3.3 | 2.9 |
| `extremeValue` | 66.0 | 60.1 | 11.9 | 10.7 | 6.5 | 5.8 |
| `fisherF(1, 1)` | 178.1 | 35.1 | 33.2 | 39.3 | 22.9 | 18.7 |
| `fisherF(5, 5)` | 141.8 | 415.2 | 136.47 | 172.4 | 92.4 | 74.9 |
| `gamma(0.2, 1)` | 207.8 | 211.4 | 54.6 | 51.2 | 26.9 | 27.0 |
| `gamma(5, 3)` | 80.9 | 60.0 | 14.3 | 13.3 | 11.4 | 8.0 |
| `gamma(10.5, 1)` | 81.1 | 248.6 | 63.3 | 58.5 | 29.2 | 28.4 |
| `geometric` | 43.0 | 22.4 | 6.7 | 7.4 | 5.8 |  |
| `lognormal` | 66.3 | 55.4 | 12.8 | 11.8 | 6.2 | 6.2 |
| `negativeBinomial(10, 0.5)` | 312.0 | 301.4 | 82.9 | 100.6 | 95.3 | 77.9 |
| `negativeBinomial(20, 0.25)` | 483.4 | 575.9 | 125.0 | 158.2 | 148.4 | 119.5 |
| `normal(0, 1)` | 38.1 | 28.5 | 6.8 | 6.2 | 3.8 | 3.7 |
| `normal(2, 3)` | 37.6 | 29.0 | 7.3 | 6.6 | 4.0 | 3.9 |
| `poisson(1)` | 31.8 | 25.2 | 9.8 | 10.8 | 9.7 | 8.2 |
| `poisson(16)` | 231.8 | 274.1 | 66.2 | 80.7 | 74.4 | 64.2 |
| `randBits` | 5.2 | 5.4 | 1.4 | 1.3 | 1.1 | 1.0 |
| `studentT(1)` | 122.7 | 120.1 | 15.3 | 19.2 | 12.6 | 9.4 |
| `studentT(20)` | 102.2 | 111.1 | 15.4 | 19.2 | 12.2 | 9.4 |
| `uniformInt(0~63)` | 22.4 | 4.7 | 1.7 | 1.6 | 1.4 | 1.1 |
| `uniformInt(0~100k)` | 21.8 | 10.1 | 6.2 | 6.7 | 6.6 | 5.4 |
| `uniformReal` | 12.9 | 5.7 | 1.4 | 1.2 | 1.4 | 0.7 |
| `weibull` | 41.0 | 35.8 | 17.7 | 15.5 | 8.5 | 8.5 |

* Since there is no equivalent class to `balanced` in C++11 std, we used Eigen::DenseBase::Random instead.

|  | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|---:|
| Mersenne Twister(int32) | 4.7 | 5.6 | 4.0 | 3.7 | 3.5 | 3.6 |
| Mersenne Twister(int64) | 5.4 | 5.3 | 4.0 | 3.9 | 3.4 | 2.6 |

|  | Python 3.6 + scipy 1.5.2 + numpy 1.19.2 | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|---:|
| `Dirichlet(4)` | 6.47 | 6.60 | 2.39 | 2.49 | 1.34 | 1.67 |
| `Dirichlet(100)` | 75.95 | 189.97 | 66.60 | 72.11 | 38.86 | 34.98 |
| `InvWishart(4)` | 140.18 | 7.62 | 4.21 | 4.54 | 3.58 | 3.39 |
| `InvWishart(50)` | 1510.47 | 1737.4 | 697.39 | 733.69 | 604.59 | 554.006 |
| `Multinomial(4, t=20)` | 3.32 | 4.12 | 0.95 | 1.06 | 1.00 | 1.03 |
| `Multinomial(4, t=1000)` | 3.51 | 192.51 | 35.99 | 39.58 | 27.84 | 35.45 |
| `Multinomial(100, t=20)` | 69.19 | 4.80 | 2.00 | 2.20 | 2.28 | 2.09 |
| `Multinomial(100, t=1000)` | 139.74 | 179.43 | 49.48 | 56.19 | 40.78 | 43.18 |
| `MvNormal(4)` | 2.32 | 0.96 | 0.36 | 0.37 | 0.25 | 0.30 |
| `MvNormal(100)` | 49.09 | 57.18 | 17.17 | 18.51 | 10.82 | 11.03 |
| `Wishart(4)` | 71.19 | 5.28 | 2.70 | 2.93 | 2.04 | 1.94 |
| `Wishart(50)` | 1185.26 | 1360.49 | 492.91 | 517.44 | 359.03 | 324.60 |


### Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz (macOS 10.15, clang-1103)

|  | C++ std (or Eigen) | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) |
|---|---:|---:|---:|---:|---:|
| `balanced`* | 6.5 | 7.3 | 1.1 | 1.4 | 1.1 |
| `balanced`(double)* | 6.6 | 7.5 | 2.6 | 3.3 | 2.4 |
| `binomial(20, 0.5)` | 38.8 | 164.9 | 27.7 | 29.3 | 24.9 |
| `binomial(50, 0.01)` | 21.9 | 27.6 | 6.6 | 7.0 | 6.3 |
| `binomial(100, 0.75)` | 52.2 | 421.9 | 93.6 | 94.8 | 89.1 |
| `cauchy` | 36.0 | 30.4 | 5.6 | 5.8 | 4.0 |
| `chiSquared` | 84.4 | 152.2 | 44.1 | 48.7 | 26.2 |
| `discrete`(int32) | - | 12.4 | 2.1 | 2.6 | 2.2 |
| `discrete`(fp32) | - | 23.2 | 3.4 | 3.7 | 3.4 |
| `discrete`(fp64) | 48.6 | 22.9 | 4.2 | 5.0 | 4.6 |
| `exponential` | 22.0 | 18.0 | 4.1 | 4.9 | 3.2 |
| `extremeValue` | 36.2 | 32.0 | 8.7 | 9.5 | 5.1 |
| `fisherF(1, 1)` | 158.2 | 73.1 | 32.3 | 32.1 | 18.1 |
| `fisherF(5, 5)` | 177.3 | 310.1 | 127.0 | 121.8 | 74.3 |
| `gamma(0.2, 1)` | 69.8 | 80.4 | 28.5 | 33.8 | 19.2 |
| `gamma(5, 3)` | 83.9 | 53.3 | 10.6 | 12.4 | 8.6 |
| `gamma(10.5, 1)` | 83.2 | 150.4 | 43.3 | 48.4 | 26.2 |
| `geometric` | 39.6 | 19.0 | 4.3 | 4.4 | 4.1 |
| `lognormal` | 43.8 | 40.7 | 9.0 | 10.8 | 5.7 |
| `negativeBinomial(10, 0.5)` | 217.4 | 274.8 | 71.6 | 73.7 | 68.2 |
| `negativeBinomial(20, 0.25)` | 192.9 | 464.9 | 112.0 | 111.5 | 105.7 |
| `normal(0, 1)` | 32.6 | 28.6 | 5.5 | 6.5 | 3.8 |
| `normal(2, 3)` | 32.9 | 30.5 | 5.7 | 6.7 | 3.9 |
| `poisson(1)` | 37.9 | 31.0 | 7.5 | 7.8 | 7.1 |
| `poisson(16)` | 92.4 | 243.3 | 55.6 | 57.7 | 53.7 |
| `randBits` | 6.5 | 6.5 | 1.1 | 1.3 | 1.1 |
| `studentT(1)` | 115.0 | 54.1 | 15.5 | 15.7 | 8.3 |
| `studentT(20)` | 121.2 | 53.8 | 15.8 | 16.0 | 8.2 |
| `uniformInt(0~63)` | 20.2 | 9.8 | 1.8 | 1.8 | 1.6 |
| `uniformInt(0~100k)` | 25.7 | 16.1 | 8.1 | 8.5 | 7.2 |
| `uniformReal` | 12.7 | 7.0 | 1.0 | 1.2 | 1.1 |
| `weibull` | 23.1 | 19.2 | 11.6 | 13.6 | 7.6 |

* Since there is no equivalent class to `balanced` in C++11 std, we used Eigen::DenseBase::Random instead.

|  | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) |
|---|---:|---:|---:|---:|---:|
| Mersenne Twister(int32) | 6.2 | 6.4 | 1.7 | 2.0 | 1.8 |
| Mersenne Twister(int64) | 6.4 | 6.3 | 2.5 | 3.1 | 2.4 |


|  | Python 3.6 + scipy 1.5.2 + numpy 1.19.2 | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) |
|---|---:|---:|---:|---:|---:|
| `Dirichlet(4)` | 3.54 | 3.29 | 1.25 | 1.25 | 0.83 |
| `Dirichlet(100)` | 57.63 | 145.32 | 49.71 | 49.50 | 29.13 |
| `InvWishart(4)` | 210.92 | 7.53 | 3.72 | 3.66 | 3.10 |
| `InvWishart(50)` | 1980.73 | 1446.40 | 560.40 | 559.73 | 457.07 |
| `Multinomial(4, t=20)` | 2.60 | 5.22 | 1.48 | 1.50 | 1.42 |
| `Multinomial(4, t=1000)` | 3.90 | 208.75 | 29.19 | 29.50 | 27.70 |
| `Multinomial(100, t=20)` | 47.71 | 7.09 | 3.71 | 3.63 | 3.60 |
| `Multinomial(100, t=1000)` | 128.69 | 215.19 | 44.48 | 44.63 | 43.76 |
| `MvNormal(4)` | 2.04 | 1.05 | 0.35 | 0.34 | 0.19 |
| `MvNormal(100)` | 48.69 | 47.10 | 16.25 | 16.12 | 11.41 |
| `Wishart(4)` | 81.11 | 13.24 | 9.87 | 9.81 | 5.90 |
| `Wishart(50)` | 1419.02 | 1087.40 | 448.06 | 442.97 | 328.20 |


### Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz (Windows Server 2019, MSVC2019)

|  | C++ std (or Eigen) | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|
| `balanced`* | 20.7 | 7.2 | 3.3 | 4.0 | 2.2 |
| `balanced`(double)* | 21.9 | 8.8 | 6.7 | 4.3 | 4.3 |
| `binomial(20, 0.5)` | 718.3 | 141.0 | 38.1 | 30.2 | 32.7 |
| `binomial(50, 0.01)` | 61.5 | 21.4 | 7.5 | 6.5 | 8.0 |
| `binomial(100, 0.75)` | 495.9 | 1042.5 | 100.6 | 95.2 | 93.0 |
| `cauchy` | 71.6 | 30.0 | 6.8 | 6.4 | 3.0 |
| `chiSquared` | 243.0 | 147.3 | 63.5 | 34.1 | 24.0 |
| `discrete`(int32) | - | 12.4 | 3.5 | 2.7 | 2.2 |
| `discrete`(fp32) | - | 19.2 | 5.1 | 3.6 | 3.7 |
| `discrete`(fp64) | 83.9 | 19.0 | 6.7 | 7.4 | 4.6 |
| `exponential` | 58.7 | 16.0 | 6.8 | 6.4 | 3.0 |
| `extremeValue` | 64.6 | 27.7 | 13.5 | 9.8 | 5.5 |
| `fisherF(1, 1)` | 178.7 | 75.2 | 35.3 | 28.4 | 17.5 |
| `fisherF(5, 5)` | 491.0 | 298.4 | 125.8 | 87.4 | 60.5 |
| `gamma(0.2, 1)` | 211.7 | 69.3 | 43.7 | 24.7 | 18.7 |
| `gamma(5, 3)` | 272.5 | 42.3 | 17.6 | 17.2 | 8.5 |
| `gamma(10.5, 1)` | 237.8 | 146.2 | 63.7 | 33.8 | 23.5 |
| `geometric` | 49.3 | 17.0 | 7.0 | 5.8 | 5.4 |
| `lognormal` | 169.8 | 37.6 | 12.7 | 7.2 | 5.0 |
| `negativeBinomial(10, 0.5)` | 752.7 | 462.3 | 87.0 | 83.0 | 81.6 |
| `negativeBinomial(20, 0.25)` | 611.4 | 855.3 | 123.7 | 125.3 | 116.6 |
| `normal(0, 1)` | 78.4 | 21.1 | 6.9 | 4.6 | 2.9 |
| `normal(2, 3)` | 77.2 | 22.3 | 6.8 | 4.8 | 3.1 |
| `poisson(1)` | 77.4 | 28.9 | 10.0 | 8.1 | 10.1 |
| `poisson(16)` | 312.9 | 485.5 | 63.6 | 61.5 | 60.5 |
| `randBits` | 6.0 | 6.2 | 3.1 | 2.7 | 2.7 |
| `studentT(1)` | 175.8 | 53.9 | 17.3 | 12.5 | 7.7 |
| `studentT(20)` | 173.2 | 55.5 | 17.9 | 12.7 | 7.6 |
| `uniformInt(0~63)` | 39.1 | 5.2 | 2.0 | 1.4 | 1.6 |
| `uniformInt(0~100k)` | 38.5 | 12.3 | 7.6 | 6.0 | 7.7 |
| `uniformReal` | 53.4 | 5.7 | 1.9 | 2.3 | 1.0 |
| `weibull` | 75.1 | 44.3 | 18.5 | 14.3 | 7.9 |

* Since there is no equivalent class to `balanced` in C++11 std, we used Eigen::DenseBase::Random instead.

|  | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|
| Mersenne Twister(int32) | 6.5 | 6.4 | 5.6 | 5.1 | 4.5 |
| Mersenne Twister(int64) | 6.6 | 6.5 | 6.9 | 5.9 | 5.1 |


|  | Python 3.6 + scipy 1.5.2 + numpy 1.19.2 | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|
| `Dirichlet(4)` | 4.27 | 3.20 | 2.31 | 1.43 | 1.25 |
| `Dirichlet(100)` | 69.61 | 150.33 | 67.01 | 47.34 | 32.47 |
| `InvWishart(4)` | 482.87 | 14.52 | 8.88 | 13.17 | 11.28 |
| `InvWishart(50)` | 2222.72 | 2211.66 | 902.34 | 775.36 | 610.60 |
| `Multinomial(4, t=20)` | 2.99 | 5.41 | 1.99 | 1.92 | 1.78 |
| `Multinomial(4, t=1000)` | 4.23 | 235.84 | 49.73 | 42.41 | 40.76 |
| `Multinomial(100, t=20)` | 58.20 | 9.12 | 5.84 | 6.02 | 5.98 |
| `Multinomial(100, t=1000)` | 130.54 | 234.40 | 72.99 | 66.36 | 55.28 |
| `MvNormal(4)` | 2.25 | 1.89 | 0.35 | 0.32 | 0.25 |
| `MvNormal(100)` | 57.71 | 68.80 | 24.40 | 18.28 | 13.05 |
| `Wishart(4)` | 70.18 | 16.25 | 4.49 | 3.97 | 3.07 |
| `Wishart(50)` | 1471.29 | 1641.73 | 628.58 | 485.68 | 349.81 |


### AMD Ryzen 7 3700x CPU @ 3.60GHz (Windows 10, MSVC2017)

|  | C++ std (or Eigen) | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|
| `balanced`* | 20.8 | 1.9 | 2.0 | 1.4 |
| `balanced`(double)* | 21.7 | 4.1 | 2.7 | 3.0 |
| `binomial(20, 0.5)` | 416.0 | 27.7 | 28.9 | 29.1 |
| `binomial(50, 0.01)` | 37.8 | 6.3 | 6.0 | 6.6 |
| `binomial(100, 0.75)` | 309.1 | 72.4 | 66.0 | 67.0 |
| `cauchy` | 42.2 | 4.8 | 5.1 | 2.7 |
| `chiSquared` | 153.8 | 33.5 | 21.2 | 17.0 |
| `discrete`(int32) | - | 2.4 | 2.3 | 2.5 |
| `discrete`(fp32) | - | 2.6 | 2.3 | 3.5 |
| `discrete`(fp64) | 55.8 | 5.1 | 4.7 | 4.3 |
| `exponential` | 33.4 | 6.4 | 2.8 | 2.2 |
| `extremeValue` | 39.4 | 7.8 | 4.6 | 4.0 |
| `fisherF(1, 1)` | 103.9 | 25.3 | 14.9 | 11.7 |
| `fisherF(5, 5)` | 295.7 | 85.5 | 58.3 | 44.8 |
| `gamma(0.2, 1)` | 128.8 | 31.9 | 18.3 | 15.8 |
| `gamma(5, 3)` | 156.1 | 9.7 | 8.0 | 5.0 |
| `gamma(10.5, 1)` | 148.5 | 33.1 | 21.1 | 17.2 |
| `geometric` | 27.1 | 6.6 | 4.3 | 4.1 |
| `lognormal` | 104.0 | 6.6 | 4.7 | 3.5 |
| `negativeBinomial(10, 0.5)` | 462.1 | 60.0 | 56.4 | 58.6 |
| `negativeBinomial(20, 0.25)` | 357.6 | 84.5 | 80.6 | 78.4 |
| `normal(0, 1)` | 48.8 | 4.2 | 3.7 | 2.3 |
| `normal(2, 3)` | 48.8 | 4.5 | 3.8 | 2.4 |
| `poisson(1)` | 46.4 | 7.9 | 7.4 | 8.2 |
| `poisson(16)` | 192.4 | 43.2 | 40.4 | 40.9 |
| `randBits` | 4.2 | 1.7 | 1.5 | 1.8 |
| `studentT(1)` | 107.0 | 12.3 | 6.8 | 5.7 |
| `studentT(20)` | 107.1 | 12.3 | 6.8 | 5.8 |
| `uniformInt(0~63)` | 31.2 | 1.1 | 1.0 | 1.2 |
| `uniformInt(0~100k)` | 27.7 | 5.6 | 5.6 | 5.4 |
| `uniformReal` | 30.7 | 1.1 | 1.0 | 0.6 |
| `weibull` | 46.5 | 10.6 | 6.4 | 5.2 |

* Since there is no equivalent class to `balanced` in C++11 std, we used Eigen::DenseBase::Random instead.

|  | C++ std | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|
| Mersenne Twister(int32) | 5.0 | 3.4 | 3.4 | 3.3 |
| Mersenne Twister(int64) | 5.1 | 3.9 | 3.9 | 3.3 |

### ARM64 NEON (Cortex-A73)
Currently, Support for ARM64 NEON is experimental and the result may be sub-optimal.
Also keep in mind that NEON does not support vectorization of double type. 
So if you use double type generators, they would fallback into scalar computations.

![Perf_no_vect](/doxygen/images/perf_neon_v0.3.90.png)

The following charts are about multivariate distributions.
![Perf_no_vect](/doxygen/images/perf_mv_part1_neon_v0.3.90.png)
![Perf_no_vect](/doxygen/images/perf_mv_part2_neon_v0.3.90.png)

Cases filled with orange are generators that are slower than reference functions.

## Accuracy
Since vectorized mathematical functions may have a loss of precision, I measured how well the generated random number fits its actual distribution.
32768 samples were generated and Earth Mover's Distance between samples and its actual distribution was calculated for each distribution.
Following table shows the average distance (and stdev.) of results performed 50 times for different seeds.

|  | C++ std | EigenRand |
|---|---:|---:|
| `balanced`* | .0034(.0015) | .0034(.0015) |
| `chiSquared(7)` | .0260(.0091) | .0242(.0079) |
| `exponential(1)` | .0065(.0025) | .0072(.0022) |
| `extremeValue(1, 1)` | .0097(.0029) | .0088(.0025) |
| `gamma(0.2, 1)` | .0380(.0021) | .0377(.0025) |
| `gamma(1, 1)` | .0070(.0020) | .0065(.0023) |
| `gamma(5, 1)` | .0169(.0065) | .0170(.0051) |
| `lognormal(0, 1)` | .0072(.0029) | .0067(.0022) |
| `normal(0, 1)` | .0070(.0024) | .0073(.0020) |
| `uniformReal` | .0018(.0008) | .0017(.0007) |
| `weibull(2, 1)` | .0032(.0013) | .0031(.0010) |

(* Result of `balanced` were from Eigen::Random, not C++ std)

The smaller value means that the sample result fits its distribution better.
The results of EigenRand and C++ std appear to be equivalent within the margin of error.


## License
MIT License

## History

### 0.4.1 (2022-08-13)
* Fixed a bug where double-type generation with std::mt19937 fails compilation.
* Fixed a bug where `UniformIntGen` in scalar mode generates numbers in the wrong range.

### 0.4.0 alpha (2021-09-28)
* Now EigenRand supports ARM & ARM64 NEON architecture experimentally. Please report issues about ARM & ARM64 NEON.
* Now EigenRand has compatibility to `Eigen 3.4.0`.

### 0.3.5 (2021-07-16)
* Now `UniformRealGen` generates accurate double values.
* Fixed a bug where non-vectorized double-type `NormalGen` would get stuck in an infinite loop.
* New overloading functions `balanced` and `balancedLike` which generate values over `[a, b]` were added.

### 0.3.4 (2021-04-25)
* Now Eigen 3.3.4 - 3.3.6 versions are additionally supported.

### 0.3.3 (2021-03-30)
* A compilation failure with some RNGs in `double` type was fixed.
* An internal function name `plgamma` conflict with one of `SpecialFunctionsPacketMath.h` was fixed.

### 0.3.2 (2021-03-26)
* A default constructor for `DiscreteGen` was added.

### 0.3.1 (2020-11-15)
* Compiling errors in the environment `EIGEN_COMP_MINGW && __GXX_ABI_VERSION < 1004` was fixed.

### 0.3.0 (2020-10-17)
* Potential cache conflict in generator was solved.
* Generator classes were added for efficient reusability.
* Multivariate distributions including `Multinomial`, `Dirichlet`, `MvNormal`, `Wishart`, `InvWishart` were added.

### 0.2.2 (2020-08-02)
* Now `ParallelRandomEngineAdaptor` and `MersenneTwister` use aligned array on heap.

### 0.2.1 (2020-07-11)
* A new template class `ParallelRandomEngineAdaptor` yielding the same random sequence regardless of SIMD ISA was added.

### 0.2.0 (2020-07-04)
* New distributions including `cauchy`, `studentT`, `fisherF`, `uniformInt`, `binomial`, `negativeBinomial`, `poisson` and `geometric` were added.
* A new member function `uniform_real` for `PacketRandomEngine` was added.

### 0.1.0 (2020-06-27)
* The first version of `EigenRand`
