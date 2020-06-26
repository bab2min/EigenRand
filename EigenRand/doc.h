/**
 * @mainpage EigenRand : The Fastest C++11-compatible random distribution generator for Eigen
 * 
 * https://github.com/bab2min/EigenRand
 * 
 * EigenRand is a header-only library for [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), providing vectorized random number engines and vectorized random distribution generators. 
 * Since the classic Random functions of Eigen relies on an old C function `rand()`, 
 * there is no way to control random numbers and no guarantee for quality of generated numbers. 
 * In addition, Eigen's Random is slow because rand() is hard to vectorize.
 * 
 * EigenRand provides a variety of random distribution functions similar to C++11 standard's random functions, 
 * which can be vectorized and easily integrated into Eigen's expressions of Matrix and Array.
 * 
 * You can get 5~10 times speed by just replacing old Eigen's Random 
 * or unvectorizable c++11 random number generators with EigenRand.
 * 
 * @page getting_started Getting Started
 * 
 * @page list_of_distribution List of Random Distribution
 * 
 * 
 @section list_of_distribution_1 Random Distributions for Real types

| Function | Scalar Type | Description | Equivalent to |
|:---:|:---:|:---:|:---:|
| `Eigen::Rand::balanced` | float, double | generates real values in the [-1, 1] range | `Eigen::DenseBase<Ty>::Random` for floating point types |
| `Eigen::Rand::chiSquared` | float, double | generates real values on a [chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution). | `std::chi_squared_distribution` |
| `Eigen::Rand::exponential` | float, double | generates real values on an [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution). | `std::exponential_distribution` |
| `Eigen::Rand::extremeValue` | float, double | generates real values on an [extreme value distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution). | `std::extreme_value_distribution` |
| `Eigen::Rand::gamma` | float, double | generates real values on a [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution). | `std::gamma_distribution` |
| `Eigen::Rand::lognormal` | float, double | generates real values on a [lognormal distribution](https://en.wikipedia.org/wiki/Lognormal_distribution). | `std::lognormal_distribution` |
| `Eigen::Rand::normal` | float, double | generates real values on a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution). | `std::normal_distribution` |
| `Eigen::Rand::uniformReal` | float, double | generates real values in the [-1, 0) range. | `std::generate_canonical` |
| `Eigen::Rand::weibull` | float, double | generates real values on a [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution). | `std::weibull_distribution` |

 @section list_of_distribution_2 Random Distributions for Integer Types

| Function | Scalar Type | Description | Equivalent to |
|:---:|:---:|:---:|:---:|
| `Eigen::Rand::randBits` | int | generates integers with random bits. | `Eigen::DenseBase<Ty>::Random` for integer types |
| `Eigen::Rand::discrete` | int | generates random integers on a discrete distribution. | `std::discrete_distribution` |
| `Eigen::Rand::uniformInt` | int | generates integers in the [min, max] range. | `std::uniform_int_distribution` |

 @section list_of_distribution_3 Random Number Engines

|  | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::Vmt19937_64` | a vectorized version of Mersenne Twister algorithm. It generates two 64bit random integers simultaneously with SSE2 and four integers with AVX2. | `std::mt19937_64` |

 * 
 * @page performance Performance
 * 
 * The following result is a measure of the time in seconds it takes to generate 1M random numbers. It shows the average of 20 times.

 @section performance_1 Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz (Ubuntu 16.04, gcc7.5)

|  | Eigen | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `balanced` | 9.0 | - | 5.9 | 1.5 | 1.4 | 1.3 | 0.9 |
| `balanced`(double) | 8.7 | - | 6.4 | 3.3 | 2.9 | 1.7 | 1.7 |
| `chiSquared` | - | 80.5 | 249.5 | 64.6 | 58.0 | 29.4 | 28.8 |
| `discrete`(int32) | - | - | 14.0 | 2.9 | 2.6 | 2.4 | 1.7 |
| `discrete`(fp32) | - | - | 21.9 | 4.3 | 4.0 | 3.6 | 3.0 |
| `discrete`(fp64) | - | 72.4 | 21.4 | 6.9 | 6.5 | 4.9 | 3.7 |
| `exponential` | - | 31.0 | 25.3 | 5.5 | 5.3 | 3.3 | 2.9 |
| `extremeValue` | - | 66.0 | 60.1 | 11.9 | 10.7 | 6.5 | 5.8 |
| `gamma(0.2, 1)` | - | 207.8 | 211.4 | 54.6 | 51.2 | 26.9 | 27.0 |
| `gamma(5, 3)` | - | 80.9 | 60.0 | 14.3 | 13.3 | 11.4 | 8.0 |
| `gamma(10.5, 1)` | - | 81.1 | 248.6 | 63.3 | 58.5 | 29.2 | 28.4 |
| `lognormal` | - | 66.3 | 55.4 | 12.8 | 11.8 | 6.2 | 6.2 |
| `normal(0, 1)` | - | 38.1 | 28.5 | 6.8 | 6.2 | 3.8 | 3.7 |
| `normal(2, 3)` | - | 37.6 | 29.0 | 7.3 | 6.6 | 4.0 | 3.9 |
| `randBits` | - | 5.2 | 5.4 | 1.4 | 1.3 | 1.1 | 1.0 |
| `uniformReal` | - | 12.9 | 5.7 | 1.4 | 1.2 | 1.4 | 0.7 |
| `weibull` | - | 41.0 | 35.8 | 17.7 | 15.5 | 8.5 | 8.5 |

|  | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|---:|
| Mersenne Twister(int32) | 4.7 | 5.6 | 4.0 | 3.7 | 3.5 | 3.6 |
| Mersenne Twister(int64) | 5.4 | 5.3 | 4.0 | 3.9 | 3.4 | 2.6 |

 @section performance_2 AMD Ryzen 7 3700x CPU @ 3.60GHz (Windows 10, MSVC2017)

|  | Eigen | C++ std | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|
| `balanced` | 20.8 | - | 1.9 | 2.0 | 1.4 |
| `balanced`(double) | 21.7 | - | 4.1 | 2.7 | 3.0 |
| `chiSquared` | - | 153.8 | 33.5 | 21.2 | 17.0 |
| `discrete`(int32) | - | - | 2.4 | 2.3 | 2.5 |
| `discrete`(fp32) | - | - | 2.6 | 2.3 | 3.5 |
| `discrete`(fp64) | - | 55.8 | 5.1 | 4.7 | 4.3 |
| `exponential` | - | 33.4 | 6.4 | 2.8 | 2.2 |
| `extremeValue` | - | 39.4 | 7.8 | 4.6 | 4.0 |
| `gamma(0.2, 1)` | - | 128.8 | 31.9 | 18.3 | 15.8 |
| `gamma(5, 3)` | - | 156.1 | 9.7 | 8.0 | 5.0 |
| `gamma(10.5, 1)` | - | 148.5 | 33.1 | 21.1 | 17.2 |
| `lognormal` | - | 104.0 | 6.6 | 4.7 | 3.5 |
| `normal(0, 1)` | - | 48.8 | 4.2 | 3.7 | 2.3 |
| `normal(2, 3)` | - | 48.8 | 4.5 | 3.8 | 2.4 |
| `randBits` | - | 4.2 | 1.7 | 1.5 | 1.8 |
| `uniformReal` | - | 30.7 | 1.1 | 1.0 | 0.6 |
| `weibull` | - | 46.5 | 10.6 | 6.4 | 5.2 |

|  | C++ std | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|
| Mersenne Twister(int32) | 5.0 | 3.4 | 3.4 | 3.3 |
| Mersenne Twister(int64) | 5.1 | 3.9 | 3.9 | 3.3 |

 * 
 */