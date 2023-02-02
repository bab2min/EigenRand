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
 
 EigenRand currently supports only x86-64 architecture (SSE, AVX, AVX2) and ARM64 NEON.

 EigenRand is distributed under the MIT License.

 If you want to contribute or report bugs, please visit Github repository https://github.com/bab2min/EigenRand.
 

 - @link getting_started Getting Started @endlink
 - @link list_of_supported_distribution List of Supported Random Distribution @endlink
 - @link performance Performance @endlink

 @page getting_started Getting Started

 @section getting_started_1 Installation

 You can install EigenRand by just downloading the source codes from [the repository](https://github.com/bab2min/EigenRand/releases). 
 Since EigenRand is a header-only library like Eigen, none of binaries needs to be installed. 
 All you need is [Eigen 3.3.4 ~ 3.4.0](http://eigen.tuxfamily.org/index.php?title=Main_Page) and C++11 compiler.

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
   Rand::P8_mt19937_64 urng{ 42 };

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
   Rand::P8_mt19937_64 urng{ 42 };

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

 @section getting_started_4 Vectorization over Parameters
 EigenRand's random number generators typically accept scalar parameters. 
 However, certain generators can generate random numbers efficiently for an array of parameters in an element-wise manner.
 You can see the full list of distributions which support the vectorization over parameters at @link list_of_supported_distribution @endlink.
 
 @code
 #include <iostream>
 #include <Eigen/Dense>
 #include <EigenRand/EigenRand>

 using namespace Eigen;

 int main()
 {
   Rand::P8_mt19937_64 urng{ 42 };

   ArrayXf a{ 10 }, b{ 10 }, c{ 10 };
	 a << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	 b << 10, 12, 14, 16, 18, 20, 22, 24, 26, 28;
   
   // You can use two array parameters.
   // The shape of two parameters should be equal in this case.
   c = Rand::uniformReal(urng, a, b);
   std::cout << c << std::endl;
   // c[0] is generated in the range [a[0], b[0]), 
   // c[1] is generated in the range [a[1], b[1]) ...
   
   // Or you can provide one parameter as a scalar
   // In this case, a scalar parameter is broadcast to the shape of the array parameter.
   c = Rand::uniformReal(urng, -5, b);
   std::cout << c << std::endl;
   // c[0] is generated in the range [-5, b[0]), 
   // c[1] is generated in the range [-5, b[1]) ...

   c = Rand::uniformReal(urng, a, 11);
   std::cout << c << std::endl;
   // c[0] is generated in the range [a[0], 11), 
   // c[1] is generated in the range [a[1], 11) ...
   return 0;
 }
 @endcode



 @section getting_started_5 Efficient Reusable Generator
 In the example above, functions, such as `Eigen::Rand::balancedLike`, `Eigen::Rand::normal` and so on, creates a generator internally each time to be called.
 If you want to generate random matrices from the same distribution, consider using Generator classes as following:

 @code
 #include <iostream>
 #include <Eigen/Dense>
 #include <EigenRand/EigenRand>

 using namespace Eigen;

 int main()
 {
   Rand::P8_mt19937_64 urng{ 42 };
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

 @section getting_started_6 Drawing samples from Multivariate Distribution
 EigenRand provides generators for some multivariate distributions.

 @code
 #include <iostream>
 #include <Eigen/Dense>
 #include <EigenRand/EigenRand>

 using namespace Eigen;

 int main()
 {
   Rand::P8_mt19937_64 urng{ 42 };

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

| Function | Generator | Scalar Type | VoP | Description | Equivalent to |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `Eigen::Rand::balanced` | `Eigen::Rand::BalancedGen` | float, double | yes | generates real values in the [-1, 1] range | `Eigen::DenseBase<Ty>::Random` for floating point types |
| `Eigen::Rand::beta` | `Eigen::Rand::BetaGen` | float, double | | generates real values on a [beta distribution](https://en.wikipedia.org/wiki/Beta_distribution) |  |
| `Eigen::Rand::cauchy` | `Eigen::Rand::CauchyGen` | float, double | yes | generates real values on the [Cauchy distribution](https://en.wikipedia.org/wiki/Cauchy_distribution). | `std::cauchy_distribution` |
| `Eigen::Rand::chiSquared` | `Eigen::Rand::ChiSquaredGen` | float, double | | generates real values on a [chi-squared distribution](https://en.wikipedia.org/wiki/Chi-squared_distribution). | `std::chi_squared_distribution` |
| `Eigen::Rand::exponential` | `Eigen::Rand::ExponentialGen` | float, double | yes | generates real values on an [exponential distribution](https://en.wikipedia.org/wiki/Exponential_distribution). | `std::exponential_distribution` |
| `Eigen::Rand::extremeValue` | `Eigen::Rand::ExtremeValueGen` | float, double | yes | generates real values on an [extreme value distribution](https://en.wikipedia.org/wiki/Generalized_extreme_value_distribution). | `std::extreme_value_distribution` |
| `Eigen::Rand::fisherF` | `Eigen::Rand::FisherFGen` | float, double | | generates real values on the [Fisher's F distribution](https://en.wikipedia.org/wiki/F_distribution). | `std::fisher_f_distribution` |
| `Eigen::Rand::gamma` | `Eigen::Rand::GammaGen` | float, double | | generates real values on a [gamma distribution](https://en.wikipedia.org/wiki/Gamma_distribution). | `std::gamma_distribution` |
| `Eigen::Rand::lognormal` | `Eigen::Rand::LognormalGen` | float, double | yes | generates real values on a [lognormal distribution](https://en.wikipedia.org/wiki/Lognormal_distribution). | `std::lognormal_distribution` |
| `Eigen::Rand::normal` | `Eigen::Rand::StdNormalGen`, `Eigen::Rand::NormalGen` | float, double | yes | generates real values on a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution). | `std::normal_distribution` |
| `Eigen::Rand::studentT` | `Eigen::Rand::StudentTGen` | float, double | yes | generates real values on the [Student's t distribution](https://en.wikipedia.org/wiki/Student%27s_t-distribution). | `std::student_t_distribution` |
| `Eigen::Rand::uniformReal` | `Eigen::Rand::StdUniformRealGen`, `Eigen::Rand::UniformRealGen` | float, double | yes | generates real values in the `[0, 1)` range. | `std::generate_canonical` |
| `Eigen::Rand::weibull` | `Eigen::Rand::WeibullGen` | float, double | yes | generates real values on the [Weibull distribution](https://en.wikipedia.org/wiki/Weibull_distribution). | `std::weibull_distribution` |

* VoP indicates 'Vectorization over Parameters'. 

 @section list_of_supported_distribution_2 Random Distributions for Integer Types

| Function | Generator | Scalar Type | VoP | Description | Equivalent to |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `Eigen::Rand::bernoulli` | `Eigen::Rand::BernoulliGen` | int | yes | generates 0 or 1 on a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution). | `std::bernoulli_distribution` |
| `Eigen::Rand::binomial` | `Eigen::Rand::BinomialGen` | int | yes | generates integers on a [binomial distribution](https://en.wikipedia.org/wiki/Binomial_distribution). | `std::binomial_distribution` |
| `Eigen::Rand::discrete` | `Eigen::Rand::DiscreteGen` | int | | generates random integers on a discrete distribution. | `std::discrete_distribution` |
| `Eigen::Rand::geometric` | `Eigen::Rand::GeometricGen` | int | | generates integers on a [geometric distribution](https://en.wikipedia.org/wiki/Geometric_distribution). | `std::geometric_distribution` |
| `Eigen::Rand::negativeBinomial` | `Eigen::Rand::NegativeBinomialGen` | int | | generates integers on a [negative binomial distribution](https://en.wikipedia.org/wiki/Negative_binomial_distribution). | `std::negative_binomial_distribution` |
| `Eigen::Rand::poisson` | `Eigen::Rand::PoissonGen` | int | | generates integers on the [Poisson distribution](https://en.wikipedia.org/wiki/Poisson_distribution). | `std::poisson_distribution` |
| `Eigen::Rand::randBits` | `Eigen::Rand::RandbitsGen` | int | | generates integers with random bits. | `Eigen::DenseBase<Ty>::Random` for integer types |
| `Eigen::Rand::uniformInt` | `Eigen::Rand::UniformIntGen` | int | | generates integers in the `[min, max]` range. | `std::uniform_int_distribution` |

* VoP indicates 'Vectorization over Parameters'. 

 @section list_of_distribution_3 Multivariate Random Distributions
| Generator | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::MultinomialGen` | generates integer vectors on a [multinomial distribution](https://en.wikipedia.org/wiki/Multinomial_distribution) | [scipy.stats.multinomial in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multinomial.html#scipy.stats.multinomial) |
| `Eigen::Rand::DirichletGen` | generates real vectors on a [Dirichlet distribution](https://en.wikipedia.org/wiki/Dirichlet_distribution) | [scipy.stats.dirichlet in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.dirichlet.html#scipy.stats.dirichlet) |
| `Eigen::Rand::MvNormalGen` | generates real vectors on a [multivariate normal distribution](https://en.wikipedia.org/wiki/Multivariate_normal_distribution) | [scipy.stats.multivariate_normal in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.multivariate_normal.html#scipy.stats.multivariate_normal) |
| `Eigen::Rand::WishartGen` | generates real matrices on a [Wishart distribution](https://en.wikipedia.org/wiki/Wishart_distribution) | [scipy.stats.wishart in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wishart.html#scipy.stats.wishart) |
| `Eigen::Rand::InvWishartGen` | generates real matrices on a [inverse Wishart distribution](https://en.wikipedia.org/wiki/Inverse-Wishart_distribution) | [scipy.stats.invwishart in Python](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invwishart.html#scipy.stats.invwishart) |

 @section list_of_distribution_4 Random Number Engines

|  | Description | Equivalent to |
|:---:|:---:|:---:|
| `Eigen::Rand::Vmt19937_64` | a vectorized version of Mersenne Twister algorithm. It generates two 64bit random integers simultaneously with SSE2 and four integers with AVX2. | `std::mt19937_64` |
| `Eigen::Rand::P8_mt19937_64` | a vectorized version of Mersenne Twister algorithm. Since it generates eight 64bit random integers simultaneously, the random values are the same regardless of architecture. | |

 * 
 * @page performance Performance
 * The following charts show the relative speed-up of EigenRand compared to references(equivalent functions of C++ std or Eigen for univariate distributions and Scipy for multivariate distributions).

Since there is no equivalent class to `balanced` in C++11 std, we used Eigen::DenseBase::Random instead.

Cases filled with orange are generators that are slower than reference functions.

 @section performance_1 Windows 2019, MSVC 19.29.30147, Intel(R) Xeon(R) Platinum 8171M CPU, AVX2, Eigen 3.4.0
 
 \image html perf_avx2_win.png width=80%
 
 \image html perf_avx2_win_mv1.png width=80%
 
 \image html perf_avx2_win_mv2.png width=80%

 @section performance_2 Ubuntu 18.04, gcc 7.5.0, Intel(R) Xeon(R) Platinum 8370C CPU, AVX2, Eigen 3.4.0
 
 \image html perf_avx2_ubu.png width=80%
 
 \image html perf_avx2_ubu_mv1.png width=80%
 
 \image html perf_avx2_ubu_mv2.png width=80%

 @section performance_3 macOS Monterey 12.2.1, clang 13.1.6, Apple M1 Pro, NEON, Eigen 3.4.0
 
 \image html perf_neon_mac.png width=80%
 
 \image html perf_neon_mac_mv1.png width=80%
 
 \image html perf_neon_mac_mv2.png width=80%

 You can see the detailed numerical values used to plot the above charts on the <a href="https://github.com/bab2min/EigenRand/actions/workflows/release.yml" target="_blank">Action Results of GitHub repository</a>.

 * 
 */
