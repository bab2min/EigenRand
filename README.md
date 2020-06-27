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
* Easily integrate with Eigen's expressions
* Currently supports only x86 and x86-64 architecture

## Requirement

* Eigen 3.3.7
* C++11-compatible compilers

## Documentation

https://bab2min.github.io/eigenrand/

## Functions

### Random distributions for real types

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

### Random distributions for integer types

| Function | Scalar Type | Description | Equivalent to |
|:---:|:---:|:---:|:---:|
| `Eigen::Rand::randBits` | int | generates integers with random bits. | `Eigen::DenseBase<Ty>::Random` for integer types |
| `Eigen::Rand::discrete` | int | generates random integers on a discrete distribution. | `std::discrete_distribution` |
| `Eigen::Rand::uniformInt` | int | generates integers in the [min, max] range. | `std::uniform_int_distribution` |

### Random number engines

|  | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::Vmt19937_64` | a vectorized version of Mersenne Twister algorithm. It generates two 64bit random integers simultaneously with SSE2 and four integers with AVX2. | `std::mt19937_64` |

## Performance

The following result is a measure of the time in seconds it takes to generate 1M random numbers. 
It shows the average of 20 times.

### Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz (Ubuntu 16.04, gcc7.5)

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

### Intel(R) Xeon(R) CPU E5-1650 v2 @ 3.50GHz (macOS 10.15, gcc8.4)

|  | Eigen | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 6.5 | - | 7.3 | 1.1 | 1.4 | 1.1 |
| `balanced`(double) | 6.6 | - | 7.5 | 2.6 | 3.3 | 2.4 |
| `chiSquared` | - | 84.4 | 152.2 | 44.1 | 48.7 | 26.2 |
| `discrete`(int32) | - | - | 12.4 | 2.1 | 2.6 | 2.2 |
| `discrete`(fp32) | - | - | 23.2 | 3.4 | 3.7 | 3.4 |
| `discrete`(fp64) | - | 48.6 | 22.9 | 4.2 | 5.0 | 4.6 |
| `exponential` | - | 22.0 | 18.0 | 4.1 | 4.9 | 3.2 |
| `extremeValue` | - | 36.2 | 32.0 | 8.7 | 9.5 | 5.1 |
| `gamma(0.2, 1)` | - | 69.8 | 80.4 | 28.5 | 33.8 | 19.2 |
| `gamma(5, 3)` | - | 83.9 | 53.3 | 10.6 | 12.4 | 8.6 |
| `gamma(10.5, 1)` | - | 83.2 | 150.4 | 43.3 | 48.4 | 26.2 |
| `lognormal` | - | 43.8 | 40.7 | 9.0 | 10.8 | 5.7 |
| `normal(0, 1)` | - | 32.6 | 28.6 | 5.5 | 6.5 | 3.8 |
| `normal(2, 3)` | - | 32.9 | 30.5 | 5.7 | 6.7 | 3.9 |
| `randBits` | - | 6.5 | 6.5 | 1.1 | 1.3 | 1.1 |
| `uniformReal` | - | 12.7 | 7.0 | 1.0 | 1.2 | 1.1 |
| `weibull` | - | 23.1 | 19.2 | 11.6 | 13.6 | 7.6 |

|  | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (SSSE3) | EigenRand (AVX) |
|---|---:|---:|---:|---:|---:|
| Mersenne Twister(int32) | 6.2 | 6.4 | 1.7 | 2.0 | 1.8 |
| Mersenne Twister(int64) | 6.4 | 6.3 | 2.5 | 3.1 | 2.4 |

### Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz (Windows Server 2019, MSVC2019)

|  | Eigen | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|---:|
| `balanced` | 20.7 | - | 7.2 | 3.3 | 4.0 | 2.2 |
| `balanced`(double) | 21.9 | - | 8.8 | 6.7 | 4.3 | 4.3 |
| `chiSquared` | - | 243.0 | 147.3 | 63.5 | 34.1 | 24.0 |
| `discrete`(int32) | - | - | 12.4 | 3.5 | 2.7 | 2.2 |
| `discrete`(fp32) | - | - | 19.2 | 5.1 | 3.6 | 3.7 |
| `discrete`(fp64) | - | 83.9 | 19.0 | 6.7 | 7.4 | 4.6 |
| `exponential` | - | 58.7 | 16.0 | 6.8 | 6.4 | 3.0 |
| `extremeValue` | - | 64.6 | 27.7 | 13.5 | 9.8 | 5.5 |
| `gamma(0.2, 1)` | - | 211.7 | 69.3 | 43.7 | 24.7 | 18.7 |
| `gamma(5, 3)` | - | 272.5 | 42.3 | 17.6 | 17.2 | 8.5 |
| `gamma(10.5, 1)` | - | 237.8 | 146.2 | 63.7 | 33.8 | 23.5 |
| `lognormal` | - | 169.8 | 37.6 | 12.7 | 7.2 | 5.0 |
| `normal(0, 1)` | - | 78.4 | 21.1 | 6.9 | 4.6 | 2.9 |
| `normal(2, 3)` | - | 77.2 | 22.3 | 6.8 | 4.8 | 3.1 |
| `randBits` | - | 6.0 | 6.2 | 3.1 | 2.7 | 2.7 |
| `uniformReal` | - | 53.4 | 5.7 | 1.9 | 2.3 | 1.0 |
| `weibull` | - | 75.1 | 44.3 | 18.5 | 14.3 | 7.9 |

|  | C++ std | EigenRand (No Vect.) | EigenRand (SSE2) | EigenRand (AVX) | EigenRand (AVX2) |
|---|---:|---:|---:|---:|---:|
| Mersenne Twister(int32) | 6.5 | 6.4 | 5.6 | 5.1 | 4.5 |
| Mersenne Twister(int64) | 6.6 | 6.5 | 6.9 | 5.9 | 5.1 |

### AMD Ryzen 7 3700x CPU @ 3.60GHz (Windows 10, MSVC2017)

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
