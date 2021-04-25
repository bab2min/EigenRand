/**
 @mainpage EigenRand : The Fastest C++11-compatible random distribution generator for Eigen
 
 EigenRand is a header-only library for [Eigen](http://eigen.tuxfamily.org/index.php?title=Main_Page), providing vectorized random number engines and vectorized random distribution generators. 
 Since the classic Random functions of Eigen relies on an old C function `rand()`, 
 there is no way to control random numbers and no guarantee for quality of generated numbers. 
 In addition, Eigen's Random is slow because rand() is hard to vectorize.
 
 EigenRand provides a variety of random distribution functions similar to C++11 standard's random functions, 
 which can be vectorized and easily integrated into Eigen's expressions of Matrix and Array.
 
 You can get 5~10 times speed by just replacing old Eigen's Random 
 or unvectorizable c++11 random number generators with EigenRand.
 
 EigenRand currently supports only x86-64 architecture (SSE, AVX, AVX2).

 EigenRand is distributed under the MIT License.

 If you want to contribute or report bugs, please visit Github repository https://github.com/bab2min/EigenRand.
 

 - @link getting_started Getting Started @endlink
 - @link list_of_supported_distribution List of Supported Random Distribution @endlink
 - @link performance Performance @endlink

 @page getting_started Getting Started

 @section getting_started_1 Installation

 You can install EigenRand by just downloading the source codes from [the repository](https://github.com/bab2min/EigenRand/releases). 
 Since EigenRand is a header-only library like Eigen, none of binaries needs to be installed. 
 All you need is [Eigen 3.3.4](http://eigen.tuxfamily.org/index.php?title=Main_Page) or later and C++11 compiler.

 @section getting_started_2 Simple Random Matrix Generators
 @code
 #include <iostream>
 #include <Eigen/Dense>
 #include <EigenRand/EigenRand>

 using namespace Eigen;

 int main()
 {
   // Initialize random number generator with seed=42 for following codes.
   // Or you can use C++11 RNG such as std::mt19937 or std::ranlux48.
   Rand::Vmt19937_64 urng{ 42 };

   // this will generate 4x4 real matrix with range [-1, 1]
   MatrixXf mat = Rand::balanced<MatrixXf>(4, 4, urng);
   std::cout << mat << std::endl;

   // this will generate 10x10 real 2d array on the normal distribution
   ArrayXXf arr = Rand::normal<ArrayXXf>(10, 10, urng);
   std::cout << arr << std::endl;

   return 0;
 }
 @endcode

 @section getting_started_3 Random Matrix Functions with suffix '-Like'
 Basically, in order to call each random distribution function of EigenRand, template parameters must be passed following the dense matrix or array type to be created.
 But, if you have an instance of Eigen::Matrix or Eigen::Array already, you can use -Like function to generate a random matrix or array with the same type and shape.
 @code
 #include <iostream>
 #include <Eigen/Dense>
 #include <EigenRand/EigenRand>

 using namespace Eigen;

 int main()
 {
   Rand::Vmt19937_64 urng{ 42 };

   MatrixXf mat{ 10, 10 };
   // this will generate a random matrix in MatrixXf type with the shape (10, 10)
   // note: it doesn't change mat at all.
   Rand::balancedLike(mat, urng);

   // if you want to assign a random matrix into itself, use assignment operator.
   mat = Rand::balancedLike(mat, urng);
   std::cout << mat << std::endl;
   return 0;
 }
 @endcode

 Every random distribution function has its corresponding -Like function.

 @section getting_started_4 Efficient Reusable Generator
 In the example above, functions, such as `Eigen::Rand::balancedLike`, `Eigen::Rand::normal` and so on, creates a generator internally each time to be called.
 If you want to generate random matrices from the same distribution, consider using Generator classes as following:

 @code
 #include <iostream>
 #include <Eigen/Dense>
 #include <EigenRand/EigenRand>

 using namespace Eigen;

 int main()
 {
   Rand::Vmt19937_64 urng{ 42 };
   // constructs generator for normal distribution with mean=1.0, stdev=2.0
   Rand::NormalGen<float> norm_gen{ 1.0, 2.0 };
   
   // Generator classes have a template function `generate`.
   // 10 by 10 random matrix will be assigned to `mat`.
   MatrixXf mat = norm_gen.template generate<MatrixXf>(10, 10, urng);
   std::cout << mat << std::endl;

   // Generator classes also have `generateLike`. 
   mat = norm_gen.generateLike(mat, urng);
   std::cout << mat << std::endl;
   return 0;
 }
 @endcode

 @section getting_started_5 Drawing samples from Multivariate Distribution
 EigenRand provides generators for some multivariate distributions.

 @code
 #include <iostream>
 #include <Eigen/Dense>
 #include <EigenRand/EigenRand>

 using namespace Eigen;

 int main()
 {
   Rand::Vmt19937_64 urng{ 42 };

   Vector4f mean{ 0, 1, 2, 3 };
   Matrix4f cov;
   cov << 1, 1, 0, 0,
          1, 2, 0, 0,
          0, 0, 3, 1,
          0, 0, 1, 2;
   {
     // constructs MvNormalGen with Scalar=float, Dim=4
     Rand::MvNormalGen<float, 4> gen1{ mean, cov };

     // or you can use `make-` helper function. It can deduce the type of generator to be created.
     auto gen2 = Rand::makeMvNormalGen(mean, cov);
     
     // generates one sample ( shape (4, 1) )
     Vector4f sample = gen1.generate(urng);

     // generates 10 samples ( shape (4, 10) )
     Matrix<float, 4, -1> samples = gen1.generate(urng, 10);
     // or you can just use `MatrixXf` type
   }

   {
     // construct MvWishartGen with Scalar=float, Dim=4, df=4
     auto gen3 = Rand::makeWishartGen(4, cov);

     // generates one sample ( shape (4, 4) )
     Matrix4f sample = gen3.generate(urng);

     // generates 10 samples ( shape (4, 40) )
     Matrix<float, 4, -1> samples = gen3.generate(urng, 10);
     // or you can just use `MatrixXf` type
   }
   return 0;
 }
 @endcode

 * @page list_of_supported_distribution List of Supported Random Distribution
 * 
 * 
 @section list_of_supported_distribution_1 Random Distributions for Real types

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
| `Eigen::Rand::uniformReal` | `Eigen::Rand::UniformRealGen` | float, double | generates real values in the `[-1, 0)` range. | `std::generate_canonical` |
| `Eigen::Rand::weibull` | `Eigen::Rand::WeibullGen` | float, double | generates real values on the [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution). | `std::weibull_distribution` |

 @section list_of_supported_distribution_2 Random Distributions for Integer Types

| Function | Generator | Scalar Type | Description | Equivalent to |
|:---:|:---:|:---:|:---:|:---:|
| `Eigen::Rand::binomial` | `Eigen::Rand::BinomialGen` | int | generates integers on a [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). | `std::binomial_distribution` |
| `Eigen::Rand::discrete` | `Eigen::Rand::DiscreteGen` | int | generates random integers on a discrete distribution. | `std::discrete_distribution` |
| `Eigen::Rand::geometric` | `Eigen::Rand::GeometricGen` | int | generates integers on a [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution). | `std::geometric_distribution` |
| `Eigen::Rand::negativeBinomial` | `Eigen::Rand::NegativeBinomialGen` | int | generates integers on a [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution). | `std::negative_binomial_distribution` |
| `Eigen::Rand::poisson` | `Eigen::Rand::PoissonGen` | int | generates integers on the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution). | `std::poisson_distribution` |
| `Eigen::Rand::randBits` | `Eigen::Rand::RandbitsGen` | int | generates integers with random bits. | `Eigen::DenseBase<Ty>::Random` for integer types |
| `Eigen::Rand::uniformInt` | `Eigen::Rand::UniformIntGen` | int | generates integers in the `[min, max]` range. | `std::uniform_int_distribution` |

 @section list_of_distribution_3 Multivariate Random Distributions
| Generator | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::MultinomialGen` | generates real vectors on a [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution) | [scipy.stats.multinomial in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html#scipy.stats.multinomial) |
| `Eigen::Rand::DirichletGen` | generates real vectors on a [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) | [scipy.stats.dirichlet in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html#scipy.stats.dirichlet) |
| `Eigen::Rand::MvNormalGen` | generates real vectors on a [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | [scipy.stats.multivariate_normal in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal) |
| `Eigen::Rand::WishartGen` | generates real matrices on a [Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution) | [scipy.stats.wishart in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html#scipy.stats.wishart) |
| `Eigen::Rand::InvWishartGen` | generates real matrices on a [inverse Wishart distribution](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution) | [scipy.stats.invwishart in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html#scipy.stats.invwishart) |

 @section list_of_distribution_4 Random Number Engines

|  | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::Vmt19937_64` | a vectorized version of Mersenne Twister algorithm. It generates two 64bit random integers simultaneously with SSE2 and four integers with AVX2. | `std::mt19937_64` |

 * 
 * @page performance Performance
 * The following charts show the relative speed-up of EigenRand compared to Reference(C++ std or Eigen functions). Detailed results are below the charts.

 \image html perf_no_vect.png

 \image html perf_sse2.png

 \image html perf_avx.png

 \image html perf_avx2.png

 \image html perf_mv_part1.png

 \image html perf_mv_part2.png

 * The following result is a measure of the time in seconds it takes to generate 1M random numbers. It shows the average of 20 times.

 @section performance_1 Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz (Ubuntu 16.04, gcc7.5)

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

 @section performance_2 AMD Ryzen 7 3700x CPU @ 3.60GHz (Windows 10, MSVC2017)

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

 * 
 */
