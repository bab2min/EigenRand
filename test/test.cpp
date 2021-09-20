#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>

template <class T>
class ContinuousDistTest : public testing::Test
{
};

using ETypes = testing::Types<float, double>;

TYPED_TEST_CASE(ContinuousDistTest, ETypes);

TYPED_TEST(ContinuousDistTest, balanced)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::Vmt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::balanced<Matrix>(8, 8, gen);
	mat = Eigen::Rand::balanced<Matrix>(3, 3, gen);
	mat = Eigen::Rand::balanced<Matrix>(5, 5, gen);
	mat = Eigen::Rand::balanced<Matrix>(5, 3, gen);
	mat = Eigen::Rand::balanced<Matrix>(1, 3, gen);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, balanced2)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::Vmt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::balanced<Matrix>(8, 8, gen, 0.5, 2);
	mat = Eigen::Rand::balanced<Matrix>(3, 3, gen, 0.5, 2);
	mat = Eigen::Rand::balanced<Matrix>(5, 5, gen, 0.5, 2);
	mat = Eigen::Rand::balanced<Matrix>(5, 3, gen, 0.5, 2);
	mat = Eigen::Rand::balanced<Matrix>(1, 3, gen, 0.5, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, uniformReal)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::Vmt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::uniformReal<Matrix>(8, 8, gen);
	mat = Eigen::Rand::uniformReal<Matrix>(3, 3, gen);
	mat = Eigen::Rand::uniformReal<Matrix>(5, 5, gen);
	mat = Eigen::Rand::uniformReal<Matrix>(5, 3, gen);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, stdNormal)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::Vmt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::normal<Matrix>(8, 8, gen);
	mat = Eigen::Rand::normal<Matrix>(3, 3, gen);
	mat = Eigen::Rand::normal<Matrix>(5, 5, gen);
	mat = Eigen::Rand::normal<Matrix>(5, 3, gen);
	mat = Eigen::Rand::normal<Matrix>(1, 3, gen);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, normal)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::Vmt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::normal<Matrix>(8, 8, gen, 1, 2);
	mat = Eigen::Rand::normal<Matrix>(3, 3, gen, 1, 2);
	mat = Eigen::Rand::normal<Matrix>(5, 5, gen, 1, 2);
	mat = Eigen::Rand::normal<Matrix>(5, 3, gen, 1, 2);
	mat = Eigen::Rand::normal<Matrix>(1, 3, gen, 1, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, exponential)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::Vmt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::exponential<Matrix>(8, 8, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(3, 3, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(5, 5, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(5, 3, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(1, 3, gen, 2);
	std::cout << mat << std::endl;
}

TEST(Issue, 29)
{
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> CMatrix;
	using Eigen::Rand::Vmt19937_64;
	using Eigen::Rand::uniformReal;

	Vmt19937_64 gen_eigen;
	CMatrix X = uniformReal<CMatrix>(5, 3, gen_eigen);
}
