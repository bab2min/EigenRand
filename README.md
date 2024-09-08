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
* Currently supports only x86, x86-64(up to AVX2), and ARM64 NEON architecture.

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

Alternatively cmake preset with cmake 3.21 or later can be used to compile EigenRand which also integrates nicely in VSCode
```console
cmake --preset default
cmake --build --preset default
ctest --preset default
```

## Documentation

https://bab2min.github.io/eigenrand/

## Functions

### Random distributions for real types

| Function | Generator | Scalar Type | VoP | Description | Equivalent to |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `Eigen::Rand::balanced` | `Eigen::Rand::BalancedGen` | float, double | Yes | generates real values in the [-1, 1] range | `Eigen::DenseBase<Ty>::Random` for floating point types |
| `Eigen::Rand::beta` | `Eigen::Rand::BetaGen` | float, double | | generates real values on a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) |  |
| `Eigen::Rand::cauchy` | `Eigen::Rand::CauchyGen` | float, double | Yes | generates real values on the [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution). | `std::cauchy_distribution` |
| `Eigen::Rand::chiSquared` | `Eigen::Rand::ChiSquaredGen` | float, double | | generates real values on a [chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution). | `std::chi_squared_distribution` |
| `Eigen::Rand::exponential` | `Eigen::Rand::ExponentialGen` | float, double | Yes | generates real values on an [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution). | `std::exponential_distribution` |
| `Eigen::Rand::extremeValue` | `Eigen::Rand::ExtremeValueGen` | float, double | Yes | generates real values on an [extreme value distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution). | `std::extreme_value_distribution` |
| `Eigen::Rand::fisherF` | `Eigen::Rand::FisherFGen` | float, double | | generates real values on the [Fisher's F distribution](https://en.wikipedia.org/wiki/F_distribution). | `std::fisher_f_distribution` |
| `Eigen::Rand::gamma` | `Eigen::Rand::GammaGen` | float, double | | generates real values on a [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution). | `std::gamma_distribution` |
| `Eigen::Rand::lognormal` | `Eigen::Rand::LognormalGen` | float, double | Yes | generates real values on a [lognormal distribution](https://en.wikipedia.org/wiki/Lognormal_distribution). | `std::lognormal_distribution` |
| `Eigen::Rand::normal` | `Eigen::Rand::StdNormalGen`, `Eigen::Rand::NormalGen` | float, double | Yes | generates real values on a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution). | `std::normal_distribution` |
| `Eigen::Rand::studentT` | `Eigen::Rand::StudentTGen` | float, double | Yes | generates real values on the [Student's t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution). | `std::student_t_distribution` |
| `Eigen::Rand::uniformReal` | `Eigen::Rand::UniformRealGen` | float, double | Yes | generates real values in the `[0, 1)` range. | `std::generate_canonical` |
| `Eigen::Rand::weibull` | `Eigen::Rand::WeibullGen` | float, double | Yes | generates real values on the [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution). | `std::weibull_distribution` |

* VoP indicates 'Vectorization over Parameters'.

### Random distributions for integer types

| Function | Generator | Scalar Type | VoP | Description | Equivalent to |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `Eigen::Rand::binomial` | `Eigen::Rand::BinomialGen` | int | Yes | generates integers on a [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). | `std::binomial_distribution` |
| `Eigen::Rand::discrete` | `Eigen::Rand::DiscreteGen` | int | | generates random integers on a discrete distribution. | `std::discrete_distribution` |
| `Eigen::Rand::geometric` | `Eigen::Rand::GeometricGen` | int | | generates integers on a [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution). | `std::geometric_distribution` |
| `Eigen::Rand::negativeBinomial` | `Eigen::Rand::NegativeBinomialGen` | int | | generates integers on a [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution). | `std::negative_binomial_distribution` |
| `Eigen::Rand::poisson` | `Eigen::Rand::PoissonGen` | int | | generates integers on the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution). | `std::poisson_distribution` |
| `Eigen::Rand::randBits` | `Eigen::Rand::RandbitsGen` | int | | generates integers with random bits. | `Eigen::DenseBase<Ty>::Random` for integer types |
| `Eigen::Rand::uniformInt` | `Eigen::Rand::UniformIntGen` | int | | generates integers in the `[min, max]` range. | `std::uniform_int_distribution` |

* VoP indicates 'Vectorization over Parameters'.

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
The following charts show the relative speed-up of EigenRand compared to references(equivalent functions of C++ std or Eigen for univariate distributions and Scipy for multivariate distributions).

* Since there is no equivalent class to `balanced` in C++11 std, we used Eigen::DenseBase::Random instead.
* Cases filled with orange are generators that are slower than reference functions.

### Windows 2019, MSVC 19.29.30147, Intel(R) Xeon(R) Platinum 8171M CPU, AVX2, Eigen 3.4.0
![Perf_AVX2_Win](/doxygen/images/perf_avx2_win.png)
![Perf_AVX2_Win_Mv1](/doxygen/images/perf_avx2_win_mv1.png)
![Perf_AVX2_Win_Mv1](/doxygen/images/perf_avx2_win_mv2.png)

### Ubuntu 18.04, gcc 7.5.0, Intel(R) Xeon(R) Platinum 8370C CPU, AVX2, Eigen 3.4.0
![Perf_AVX2_Ubu](/doxygen/images/perf_avx2_ubu.png)
![Perf_AVX2_Ubu_Mv1](/doxygen/images/perf_avx2_ubu_mv1.png)
![Perf_AVX2_Ubu_Mv1](/doxygen/images/perf_avx2_ubu_mv2.png)

### macOS Monterey 12.2.1, clang 13.1.6, Apple M1 Pro, NEON, Eigen 3.4.0
![Perf_NEON_mac](/doxygen/images/perf_neon_mac.png)
![Perf_NEON_mac_Mv1](/doxygen/images/perf_neon_mac_mv1.png)
![Perf_NEON_mac_Mv1](/doxygen/images/perf_neon_mac_mv2.png)

You can see the detailed numerical values used to plot the above charts on the [Action](https://github.com/bab2min/EigenRand/actions/workflows/release.yml) page.

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

### 0.5.1 (2024-09-08)
* Add AVX512 support
* Add `EIGENRAND_BUILD_BENCHMARK` cmake option

### 0.5.0 (2023-01-31)
* Improved the performance of `MultinomialGen`.
* Implemented vectorization over parameters to some distributions.
* Optimized the performance of `double`-type generators on NEON architecture.

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
