#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

template<class Scalar>
Eigen::Matrix<Scalar, -1, 1> calcMean(const Eigen::Matrix<Scalar, -1, -1>& samples)
{
    return samples.rowwise().mean();
}

template<class Scalar>
Eigen::Matrix<Scalar, -1, -1> calcCov(const Eigen::Matrix<Scalar, -1, -1>& samples)
{
    Eigen::Matrix<Scalar, -1, -1> t = samples.colwise() - samples.rowwise().mean();
    return t * t.transpose() / (samples.cols() - 1);
}

template<class Scalar>
Eigen::Matrix<Scalar, -1, -1> calcMatMean(const Eigen::Matrix<Scalar, -1, -1>& samples)
{
    Eigen::Map<const Eigen::Matrix<Scalar, -1, -1>> reshaped{ samples.data(), samples.rows() * samples.rows(), samples.cols() / samples.rows() };
    Eigen::Matrix<Scalar, -1, -1> ret{ samples.rows(), samples.rows() };
    Eigen::Map<Eigen::Matrix<Scalar, -1, 1>> reshapedRet{ ret.data(), samples.rows() * samples.rows() };
    reshapedRet = reshaped.rowwise().mean();
    return ret;
}

template<class Scalar>
Eigen::Matrix<Scalar, -1, -1> calcMatVar(const Eigen::Matrix<Scalar, -1, -1>& samples)
{
    Eigen::Map<const Eigen::Array<Scalar, -1, -1>> reshaped{ samples.data(), samples.rows() * samples.rows(), samples.cols() / samples.rows() };
    Eigen::Matrix<Scalar, -1, -1> ret{ samples.rows(), samples.rows() };
    Eigen::Map<Eigen::Matrix<Scalar, -1, 1>> reshapedRet{ ret.data(), samples.rows() * samples.rows() };
    reshapedRet = reshaped.pow(2).rowwise().mean() - reshaped.rowwise().mean().pow(2);
    return ret;
}

template<class Ty>
auto relErrors(Ty real, Ty target, double minValue) -> decltype((real - target).abs() / real.abs().max(minValue))
{
    return (real - target).abs() / real.abs().max(minValue);
}

template<class Ty>
bool isSimilarWithErrors(Ty real, Ty target, double relError, double minValue)
{
    return (relErrors(real.array(), target.array(), minValue) <= relError).all();
}

#define EXPECT_SIMILAR_MATRIX(a, b, m) do\
{\
    bool t = isSimilarWithErrors(a, b, 0.1, m);\
    GTEST_TEST_BOOLEAN_(t, #a " is not similar with " #b, false, true, GTEST_NONFATAL_FAILURE_);\
    if (!t)\
    {\
        std::cout << #a "\n" << a << std::endl;\
        std::cout << #b "\n" << b << std::endl;\
        std::cout << "re: \n" << relErrors(a.array(), b.array(), m) << std::endl;\
    }\
} while(0);

template<class Scalar>
Eigen::Matrix<Scalar, -1, 1> marginalTruncNormalMean(
    const Eigen::Matrix<Scalar, -1, 1>& mu,
    const Eigen::Matrix<Scalar, -1, -1>& cov,
    const Eigen::Matrix<Scalar, -1, 1>& lower,
    const Eigen::Matrix<Scalar, -1, 1>& upper)
{
    const Eigen::Index dim = mu.rows();
    Eigen::Matrix<Scalar, -1, 1> result(dim);
    const Scalar inv_sqrt_2pi = Scalar(1.0) / std::sqrt(Scalar(2.0) * Scalar(3.14159265358979323846));
    const Scalar inv_sqrt2 = Scalar(1.0 / 1.4142135623730951);
    for (Eigen::Index i = 0; i < dim; ++i)
    {
        Scalar sigma = std::sqrt(cov(i, i));
        Scalar alpha = (lower(i) - mu(i)) / sigma;
        Scalar beta = (upper(i) - mu(i)) / sigma;
        Scalar phi_alpha = std::exp(Scalar(-0.5) * alpha * alpha) * inv_sqrt_2pi;
        Scalar phi_beta = std::exp(Scalar(-0.5) * beta * beta) * inv_sqrt_2pi;
        Scalar Phi_alpha = Scalar(0.5) * std::erfc(-alpha * inv_sqrt2);
        Scalar Phi_beta = Scalar(0.5) * std::erfc(-beta * inv_sqrt2);
        result(i) = mu(i) + sigma * (phi_alpha - phi_beta) / (Phi_beta - Phi_alpha);
    }
    return result;
}

static constexpr size_t numSamples = 5000;

template <class T>
class MvDistTest : public testing::Test
{
};

using ETypes = testing::Types<float, double>;

TYPED_TEST_CASE(MvDistTest, ETypes);

TYPED_TEST(MvDistTest, normal)
{
    std::cout << "SIMD arch: " << Eigen::SimdInstructionSetsInUse() << std::endl;
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    Eigen::Matrix<TypeParam, -1, 1> mean(9);
    mean << -1, 0, 1, 2, 3, 2, 1, 0, -1;

    Eigen::Matrix<TypeParam, -1, -1> cov(9, 9);
    cov.setZero();
    cov(0, 1) = 0.5;
    cov(2, 7) = 0.5;
    cov(3, 1) = 0.7;
    cov.diagonal() << 1, 1.2, 1.4, 1.6, 1.8, 1.6, 1.4, 1.2, 1;
    cov = cov * cov.transpose();

    auto gen = Eigen::Rand::makeMvNormalGen(mean, cov);
    auto samples = gen.generate(rng, numSamples).eval();
    
    auto mean2 = calcMean(samples);
    auto cov2 = calcCov(samples);

    EXPECT_SIMILAR_MATRIX(mean, mean2, 2.5);
    EXPECT_SIMILAR_MATRIX(cov, cov2, 2.5);
}

TYPED_TEST(MvDistTest, wishart)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    int df = 12;
    Eigen::Matrix<TypeParam, -1, -1> scale(9, 9);
    scale.setZero();
    scale(0, 1) = 0.5;
    scale(2, 7) = 0.5;
    scale(3, 1) = 0.7;
    scale.diagonal() << 1, 1.2, 1.4, 1.6, 1.8, 1.6, 1.4, 1.2, 1;
    scale = scale * scale.transpose();

    auto gen = Eigen::Rand::makeWishartGen(df, scale);
    auto samples = gen.generate(rng, numSamples).eval();

    auto mean = (df * scale).eval();
    auto var = (scale.diagonal() * scale.diagonal().transpose()).eval();
    var.array() += scale.array() * scale.array();
    var *= df;

    auto mean2 = calcMatMean(samples);
    auto var2 = calcMatVar(samples);

    EXPECT_SIMILAR_MATRIX(mean, mean2, 5);
    EXPECT_SIMILAR_MATRIX(var, var2, 5);
}

TYPED_TEST(MvDistTest, invWishart)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    float df = 15;
    int p = 9;
    Eigen::Matrix<TypeParam, -1, -1> scale(p, p);
    scale.setZero();
    scale(0, 1) = 0.5;
    scale(2, 7) = 0.5;
    scale(3, 1) = 0.7;
    scale.diagonal() << 1, 1.2, 1.4, 1.6, 1.8, 1.6, 1.4, 1.2, 1;
    scale = scale * scale.transpose();

    auto gen = Eigen::Rand::makeInvWishartGen(df, scale);
    auto samples = gen.generate(rng, numSamples).eval();

    auto mean = (scale / (df - p - 1)).eval();
    auto var = (scale.diagonal() * scale.diagonal().transpose() * (df - p - 1)).eval();
    var.array() += (scale.array() * scale.array()) * (df - p + 1);
    var /= (df - p) * (df - p - 1) * (df - p - 1) * (df - p - 3);

    auto mean2 = calcMatMean(samples);
    auto var2 = calcMatVar(samples);

    EXPECT_SIMILAR_MATRIX(mean, mean2, 5);
    EXPECT_SIMILAR_MATRIX(var, var2, 5);
}

TEST(MvDistTest, multinomial)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    int n = 10;
    Eigen::Matrix<float, -1, 1> weight(9);
    weight << 1, 2, 3, 4, 5, 6, 7, 8, 9;
    weight /= weight.sum();
    auto gen = Eigen::Rand::makeMultinomialGen(n, weight);
    auto samples = gen.generate(rng, numSamples).array().template cast<float>().matrix().eval();

    auto mean = (weight * n).eval();
    auto cov = (weight * weight.transpose() * -n).eval();
    cov.diagonal() = weight.array() * (1 - weight.array()) * n;

    auto mean2 = calcMean(samples);
    auto cov2 = calcCov(samples);

    EXPECT_SIMILAR_MATRIX(mean, mean2, 2.5);
    EXPECT_SIMILAR_MATRIX(cov, cov2, 2.5);
}

TYPED_TEST(MvDistTest, dirichlet)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    Eigen::Matrix<TypeParam, -1, 1> weight(9);
    weight << .1, .2, .3, .4, .5, .6, .7, .8, .9;
    auto gen = Eigen::Rand::makeDirichletGen(weight);
    auto samples = gen.generate(rng, numSamples).eval();

    auto mean = (weight / weight.sum()).eval();
    auto cov = (-mean * mean.transpose()).eval();
    cov.diagonal() += mean;
    cov /= weight.sum() + 1;

    auto mean2 = calcMean(samples);
    auto cov2 = calcCov(samples);

    EXPECT_SIMILAR_MATRIX(mean, mean2, 2.5);
    EXPECT_SIMILAR_MATRIX(cov, cov2, 2.5);
}

TYPED_TEST(MvDistTest, truncMvNormal)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    Eigen::Matrix<TypeParam, -1, 1> mean(3);
    mean << 1, 2, 3;

    Eigen::Matrix<TypeParam, -1, -1> cov(3, 3);
    cov << 1.0, 0.3, 0.1,
           0.3, 1.2, 0.2,
           0.1, 0.2, 0.8;

    Eigen::Matrix<TypeParam, -1, 1> lower(3), upper(3);
    lower << -1, 0, 1;
    upper <<  5, 5, 6;

    auto gen = Eigen::Rand::makeTruncMvNormalGen(mean, cov, Eigen::Rand::support(lower, upper));
    auto samples = gen.generate(rng, numSamples).eval();

    // All samples must be within bounds
    for (Eigen::Index k = 0; k < samples.cols(); ++k)
    {
        for (Eigen::Index i = 0; i < 3; ++i)
        {
            EXPECT_GE(samples(i, k), lower(i)) << "dim=" << i << " sample=" << k;
            EXPECT_LE(samples(i, k), upper(i)) << "dim=" << i << " sample=" << k;
        }
    }

    auto mean2 = calcMean(samples);
    auto expected_mean = marginalTruncNormalMean(mean, cov, lower, upper);
    EXPECT_SIMILAR_MATRIX(expected_mean, mean2, 0.5);
}

TYPED_TEST(MvDistTest, truncMvNormalTight)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    Eigen::Matrix<TypeParam, -1, 1> mean(3);
    mean << 0, 0, 0;

    Eigen::Matrix<TypeParam, -1, -1> cov(3, 3);
    cov << 1.0, 0.5, 0.0,
           0.5, 1.0, 0.5,
           0.0, 0.5, 1.0;

    Eigen::Matrix<TypeParam, -1, 1> lower(3), upper(3);
    lower << 0.5, 0.5, 0.5;
    upper << 1.5, 1.5, 1.5;

    auto gen = Eigen::Rand::makeTruncMvNormalGen(mean, cov, Eigen::Rand::support(lower, upper), 50);
    auto samples = gen.generate(rng, numSamples).eval();

    // All samples must be within bounds
    for (Eigen::Index k = 0; k < samples.cols(); ++k)
    {
        for (Eigen::Index i = 0; i < 3; ++i)
        {
            EXPECT_GE(samples(i, k), lower(i));
            EXPECT_LE(samples(i, k), upper(i));
        }
    }

    // Not all samples should be identical
    auto col0 = samples.col(0);
    bool all_same = true;
    for (Eigen::Index k = 1; k < samples.cols(); ++k)
    {
        if (!col0.isApprox(samples.col(k), (TypeParam)1e-6))
        {
            all_same = false;
            break;
        }
    }
    EXPECT_FALSE(all_same);

    auto mean2 = calcMean(samples);
    auto expected_mean = marginalTruncNormalMean(mean, cov, lower, upper);
    EXPECT_SIMILAR_MATRIX(expected_mean, mean2, 1.0);
}

TYPED_TEST(MvDistTest, truncMvNormalOneSided)
{
    Eigen::Matrix<TypeParam, -1, 1> mean(3);
    mean << 0, 0, 0;

    Eigen::Matrix<TypeParam, -1, -1> cov(3, 3);
    cov << 1.0, 0.3, 0.0,
           0.3, 1.0, 0.3,
           0.0, 0.3, 1.0;

    Eigen::Matrix<TypeParam, -1, 1> lower(3), upper(3);
    // Lower-bounded only: x >= 0 for all dimensions
    lower << 0, 0, 0;
    upper << (TypeParam)100, (TypeParam)100, (TypeParam)100;

    // Compute reference mean from large sample (marginal approximation is
    // inaccurate here because positive correlations + lower truncation shift
    // the multivariate mean beyond what univariate formulas predict)
    Eigen::Rand::P8_mt19937_64 rng_ref{ 54321 };
    auto gen_ref = Eigen::Rand::makeTruncMvNormalGen(mean, cov, Eigen::Rand::support(lower, upper), 10 * mean.rows());
    auto expected_mean = calcMean(gen_ref.generate(rng_ref, 50000).eval());

    // Sanity: reference mean should exceed marginal truncated normal mean
    // (correlations push all means higher when truncated from below)
    auto marginal_mean = marginalTruncNormalMean(mean, cov, lower, upper);
    for (Eigen::Index i = 0; i < 3; ++i)
    {
        EXPECT_GT(expected_mean(i), marginal_mean(i));
    }

    // Verify symmetry: E[X_0] ≈ E[X_2] (by covariance symmetry)
    EXPECT_NEAR((double)expected_mean(0), (double)expected_mean(2), 0.02);

    // Generate test samples and compare against reference
    Eigen::Rand::P8_mt19937_64 rng{ 42 };
    auto gen = Eigen::Rand::makeTruncMvNormalGen(mean, cov, Eigen::Rand::support(lower, upper));
    auto samples = gen.generate(rng, numSamples).eval();

    for (Eigen::Index k = 0; k < samples.cols(); ++k)
    {
        for (Eigen::Index i = 0; i < 3; ++i)
        {
            EXPECT_GE(samples(i, k), lower(i));
        }
    }

    auto mean2 = calcMean(samples);
    EXPECT_SIMILAR_MATRIX(expected_mean, mean2, 0.5);
}

TYPED_TEST(MvDistTest, truncMvNormalHighDim)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    Eigen::Matrix<TypeParam, -1, 1> mean(9);
    mean << -1, 0, 1, 2, 3, 2, 1, 0, -1;

    Eigen::Matrix<TypeParam, -1, -1> cov(9, 9);
    cov.setZero();
    cov(0, 1) = 0.5;
    cov(2, 7) = 0.5;
    cov(3, 1) = 0.7;
    cov.diagonal() << 1, 1.2, 1.4, 1.6, 1.8, 1.6, 1.4, 1.2, 1;
    cov = cov * cov.transpose();

    Eigen::Matrix<TypeParam, -1, 1> lower(9), upper(9);
    lower << -4, -3, -2, -1, 0, -1, -2, -3, -4;
    upper <<  4,  5,  6,  7, 8,  7,  6,  5,  4;

    auto gen = Eigen::Rand::makeTruncMvNormalGen(mean, cov, Eigen::Rand::support(lower, upper));
    auto samples = gen.generate(rng, numSamples).eval();

    // All samples must be within bounds
    for (Eigen::Index k = 0; k < samples.cols(); ++k)
    {
        for (Eigen::Index i = 0; i < 9; ++i)
        {
            EXPECT_GE(samples(i, k), lower(i)) << "dim=" << i << " sample=" << k;
            EXPECT_LE(samples(i, k), upper(i)) << "dim=" << i << " sample=" << k;
        }
    }

    auto mean2 = calcMean(samples);
    auto expected_mean = marginalTruncNormalMean(mean, cov, lower, upper);
    EXPECT_SIMILAR_MATRIX(expected_mean, mean2, 2.5);
}

TYPED_TEST(MvDistTest, truncMvNormalFromLt)
{
    Eigen::Rand::P8_mt19937_64 rng{ 42 };

    Eigen::Matrix<TypeParam, -1, 1> mean(3);
    mean << 1, 2, 3;

    Eigen::Matrix<TypeParam, -1, -1> cov(3, 3);
    cov << 1.0, 0.3, 0.1,
           0.3, 1.2, 0.2,
           0.1, 0.2, 0.8;

    Eigen::Matrix<TypeParam, -1, 1> lower(3), upper(3);
    lower << -1, 0, 1;
    upper <<  5, 5, 6;

    // Get Cholesky factor
    Eigen::LLT<Eigen::Matrix<TypeParam, -1, -1>> llt(cov);
    auto lt = llt.matrixL().toDenseMatrix().eval();

    auto gen = Eigen::Rand::makeTruncMvNormalGenFromLt(mean, lt, Eigen::Rand::support(lower, upper));
    auto samples = gen.generate(rng, numSamples).eval();

    // All samples must be within bounds
    for (Eigen::Index k = 0; k < samples.cols(); ++k)
    {
        for (Eigen::Index i = 0; i < 3; ++i)
        {
            EXPECT_GE(samples(i, k), lower(i)) << "dim=" << i << " sample=" << k;
            EXPECT_LE(samples(i, k), upper(i)) << "dim=" << i << " sample=" << k;
        }
    }

    // Compare mean with covariance constructor using a different seed
    Eigen::Rand::P8_mt19937_64 rng2{ 42 };
    auto gen2 = Eigen::Rand::makeTruncMvNormalGen(mean, cov, Eigen::Rand::support(lower, upper));
    auto samples2 = gen2.generate(rng2, numSamples).eval();

    auto mean_lt = calcMean(samples);
    auto mean_cov = calcMean(samples2);

    auto expected_mean = marginalTruncNormalMean(mean, cov, lower, upper);
    EXPECT_SIMILAR_MATRIX(expected_mean, mean_lt, 0.5);
    EXPECT_SIMILAR_MATRIX(expected_mean, mean_cov, 0.5);
}
