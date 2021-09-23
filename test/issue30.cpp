#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

using TMatrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
using TVector = Eigen::VectorXd;
Eigen::Rand::P8_mt19937_64 gen_eigen{ 42 };

void cdist(const TMatrix& X1, const TMatrix& X2, TMatrix& D, bool squared = false) 
{
    // |X1(i) - X2(j)^T|^2 = |X1(i)|^2 + |X2(j)|^2 - (2 * X1(i)^T X2(j))
    D = ((-2.0 * (X1 * X2.transpose())).colwise()
        + X1.rowwise().squaredNorm()).rowwise()
        + X2.rowwise().squaredNorm().transpose();
    if (!squared) { D = (D.array().sqrt()).matrix(); }
}

const TMatrix Matern52(const TMatrix& X1, const TMatrix& X2, TVector& length_scale) 
{
    TMatrix R(X1.rows(), X2.rows());
    const TMatrix X1sc = X1.array().rowwise() / length_scale.transpose().array();
    const TMatrix X2sc = X2.array().rowwise() / length_scale.transpose().array();
    cdist(X1sc, X2sc, R, false);
    R *= sqrt(5);
    return ((1 + R.array() + square(R.array()) / 3) * (exp(-R.array()))).matrix();
}

void xiong1d(const Eigen::Ref<const TMatrix>& X, Eigen::Ref<TMatrix> Y) 
{
    Y = -0.5 * (sin((40 * pow((X.array() - 0.85), 4)).array()) * cos((2.5 * (X.array() - 0.95)).array()) + 0.5 * (X.array() - 0.9) + 1);
}

TMatrix sample_mvn(const TVector& mean, const TMatrix& K) 
{
    auto sampler = Eigen::Rand::makeMvNormalGen(mean, K);
    return sampler.generate(gen_eigen);
}

TEST(Issue, 30)
{
    std::cout << "SIMD arch: " << Eigen::SimdInstructionSetsInUse() << std::endl;

    TVector x_range = TVector::LinSpaced(10, 0, 1);
    TMatrix X = Eigen::Map<TVector>(x_range.data(), 10);
    TMatrix Y(X.rows(), 1);
    xiong1d(X, Y);
    TVector length_scale = TVector::Ones(1);
    TMatrix K = Matern52(X, X, length_scale);
    std::cout << "K =\n" << K << std::endl;
    TVector mean = TVector::Zero(X.rows());
    TMatrix mvn_samples = sample_mvn(mean, K);
    std::cout << "MVN_SAMPLE =\n" << mvn_samples << std::endl;
    EXPECT_TRUE((mvn_samples.array().abs() <= 2).all());
}
