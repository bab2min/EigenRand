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
