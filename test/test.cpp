#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <EigenRand/EigenRand>
#include <cmath>

#ifdef EIGEN_VECTORIZE_NEON

TEST(MorePacketMath, log)
{
	Eigen::ArrayXf x(8), y(8);
	x << -1, 0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0;
	y.writePacket<0>(0, Eigen::internal::plog(x.packet<0>(0)));
	y.writePacket<0>(4, Eigen::internal::plog(x.packet<0>(4)));
	std::cout << y.transpose() << std::endl;
	for (int i = 0; i < x.size(); ++i)
	{
		if (std::isnan(std::log(x[i])))
		{
			EXPECT_TRUE(std::isnan(y[i]));
		}
		else
		{
			EXPECT_FLOAT_EQ(std::log(x[i]), y[i]);
		}
	}
}

TEST(MorePacketMath, sin)
{
	Eigen::ArrayXf x(8), y(8);
	x << -1, 0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0;
	y.writePacket<0>(0, Eigen::internal::psin(x.packet<0>(0)));
	y.writePacket<0>(4, Eigen::internal::psin(x.packet<0>(4)));
	std::cout << y.transpose() << std::endl;
	for (int i = 0; i < x.size(); ++i)
	{
		if (std::isnan(std::sin(x[i])))
		{
			EXPECT_TRUE(std::isnan(y[i]));
		}
		else
		{
			EXPECT_FLOAT_EQ(std::sin(x[i]), y[i]);
		}
	}
}

TEST(MorePacketMath, sincos)
{
	Eigen::ArrayXf x(8), y(8), z(8);
	x << -1, 0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0;
	Eigen::internal::Packet4f s, c;
	Eigen::internal::psincos(x.packet<0>(0), s, c);
	y.writePacket<0>(0, s);
	z.writePacket<0>(0, c);
	Eigen::internal::psincos(x.packet<0>(4), s, c);
	y.writePacket<0>(4, s);
	z.writePacket<0>(4, c);
	std::cout << y.transpose() << std::endl;
	std::cout << z.transpose() << std::endl;
	for (int i = 0; i < x.size(); ++i)
	{
		if (std::isnan(std::sin(x[i])))
		{
			EXPECT_TRUE(std::isnan(y[i]));
		}
		else
		{
			EXPECT_FLOAT_EQ(std::sin(x[i]), y[i]);
		}

		if (std::isnan(std::cos(x[i])))
		{
			EXPECT_TRUE(std::isnan(z[i]));
		}
		else
		{
			EXPECT_FLOAT_EQ(std::cos(x[i]), z[i]);
		}
	}
}

TEST(MorePacketMath, sqrt)
{
	Eigen::ArrayXf x(8), y(8);
	x << -1, 0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0;
	y.writePacket<0>(0, Eigen::internal::psqrt(x.packet<0>(0)));
	y.writePacket<0>(4, Eigen::internal::psqrt(x.packet<0>(4)));
	std::cout << y.transpose() << std::endl;
	for (int i = 0; i < x.size(); ++i)
	{
		if (std::isnan(std::sqrt(x[i])))
		{
			EXPECT_TRUE(std::isnan(y[i]));
		}
		else
		{
			EXPECT_FLOAT_EQ(std::sqrt(x[i]), y[i]);
		}
	}
}

#endif

template <class T>
class ContinuousDistTest : public testing::Test
{
};

using ETypes = testing::Types<float, double>;

TYPED_TEST_CASE(ContinuousDistTest, ETypes);

TYPED_TEST(ContinuousDistTest, balanced)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
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
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
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
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
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
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
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
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
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
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::exponential<Matrix>(8, 8, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(3, 3, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(5, 5, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(5, 3, gen, 2);
	mat = Eigen::Rand::exponential<Matrix>(1, 3, gen, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, lognormal)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::lognormal<Matrix>(8, 8, gen, 1, 2);
	mat = Eigen::Rand::lognormal<Matrix>(3, 3, gen, 1, 2);
	mat = Eigen::Rand::lognormal<Matrix>(5, 5, gen, 1, 2);
	mat = Eigen::Rand::lognormal<Matrix>(5, 3, gen, 1, 2);
	mat = Eigen::Rand::lognormal<Matrix>(1, 3, gen, 1, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, beta)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::beta<Matrix>(8, 8, gen, 1, 2);
	mat = Eigen::Rand::beta<Matrix>(3, 3, gen, 1, 2);
	mat = Eigen::Rand::beta<Matrix>(5, 5, gen, 1, 2);
	mat = Eigen::Rand::beta<Matrix>(5, 3, gen, 1, 2);
	mat = Eigen::Rand::beta<Matrix>(1, 3, gen, 1, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, cauchy)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::cauchy<Matrix>(8, 8, gen, 1, 2);
	mat = Eigen::Rand::cauchy<Matrix>(3, 3, gen, 1, 2);
	mat = Eigen::Rand::cauchy<Matrix>(5, 5, gen, 1, 2);
	mat = Eigen::Rand::cauchy<Matrix>(5, 3, gen, 1, 2);
	mat = Eigen::Rand::cauchy<Matrix>(1, 3, gen, 1, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, studentT)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::studentT<Matrix>(8, 8, gen, 5);
	mat = Eigen::Rand::studentT<Matrix>(3, 3, gen, 5);
	mat = Eigen::Rand::studentT<Matrix>(5, 5, gen, 5);
	mat = Eigen::Rand::studentT<Matrix>(5, 3, gen, 5);
	mat = Eigen::Rand::studentT<Matrix>(1, 3, gen, 5);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, gamma)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::gamma<Matrix>(8, 8, gen, 1, 2);
	mat = Eigen::Rand::gamma<Matrix>(3, 3, gen, 1, 2);
	mat = Eigen::Rand::gamma<Matrix>(5, 5, gen, 1, 2);
	mat = Eigen::Rand::gamma<Matrix>(5, 3, gen, 1, 2);
	mat = Eigen::Rand::gamma<Matrix>(1, 3, gen, 1, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, weibull)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::weibull<Matrix>(8, 8, gen, 1, 2);
	mat = Eigen::Rand::weibull<Matrix>(3, 3, gen, 1, 2);
	mat = Eigen::Rand::weibull<Matrix>(5, 5, gen, 1, 2);
	mat = Eigen::Rand::weibull<Matrix>(5, 3, gen, 1, 2);
	mat = Eigen::Rand::weibull<Matrix>(1, 3, gen, 1, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, extremeValueLike)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::extremeValue<Matrix>(8, 8, gen, 1, 2);
	mat = Eigen::Rand::extremeValue<Matrix>(3, 3, gen, 1, 2);
	mat = Eigen::Rand::extremeValue<Matrix>(5, 5, gen, 1, 2);
	mat = Eigen::Rand::extremeValue<Matrix>(5, 3, gen, 1, 2);
	mat = Eigen::Rand::extremeValue<Matrix>(1, 3, gen, 1, 2);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, chiSquared)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::chiSquared<Matrix>(8, 8, gen, 7);
	mat = Eigen::Rand::chiSquared<Matrix>(3, 3, gen, 7);
	mat = Eigen::Rand::chiSquared<Matrix>(5, 5, gen, 7);
	mat = Eigen::Rand::chiSquared<Matrix>(5, 3, gen, 7);
	mat = Eigen::Rand::chiSquared<Matrix>(1, 3, gen, 7);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, fisherF1)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::fisherF<Matrix>(8, 8, gen, 1, 1);
	mat = Eigen::Rand::fisherF<Matrix>(3, 3, gen, 1, 1);
	mat = Eigen::Rand::fisherF<Matrix>(5, 5, gen, 1, 1);
	mat = Eigen::Rand::fisherF<Matrix>(5, 3, gen, 1, 1);
	mat = Eigen::Rand::fisherF<Matrix>(1, 3, gen, 1, 1);
	std::cout << mat << std::endl;
}

TYPED_TEST(ContinuousDistTest, fisherF2)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::fisherF<Matrix>(8, 8, gen, 5, 5);
	mat = Eigen::Rand::fisherF<Matrix>(3, 3, gen, 5, 5);
	mat = Eigen::Rand::fisherF<Matrix>(5, 5, gen, 5, 5);
	mat = Eigen::Rand::fisherF<Matrix>(5, 3, gen, 5, 5);
	mat = Eigen::Rand::fisherF<Matrix>(1, 3, gen, 5, 5);
	std::cout << mat << std::endl;
}

template <class T>
class DiscreteDistTest : public testing::Test
{
};

TYPED_TEST_CASE(DiscreteDistTest, testing::Types<int32_t>);

TYPED_TEST(DiscreteDistTest, uniformInt)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::uniformInt<Matrix>(8, 8, gen, 0, 9);
	mat = Eigen::Rand::uniformInt<Matrix>(3, 3, gen, 0, 9);
	mat = Eigen::Rand::uniformInt<Matrix>(5, 5, gen, 0, 9);
	mat = Eigen::Rand::uniformInt<Matrix>(5, 3, gen, 0, 9);
	mat = Eigen::Rand::uniformInt<Matrix>(1, 3, gen, 0, 9);
	std::cout << mat << std::endl;
}

TYPED_TEST(DiscreteDistTest, discrete)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::discrete<Matrix>(8, 8, gen, { 1, 2, 3, 4, 5, 6 });
	mat = Eigen::Rand::discrete<Matrix>(3, 3, gen, { 1, 2, 3, 4, 5, 6 });
	mat = Eigen::Rand::discrete<Matrix>(5, 5, gen, { 1, 2, 3, 4, 5, 6 });
	mat = Eigen::Rand::discrete<Matrix>(5, 3, gen, { 1, 2, 3, 4, 5, 6 });
	mat = Eigen::Rand::discrete<Matrix>(1, 3, gen, { 1, 2, 3, 4, 5, 6 });
	std::cout << mat << std::endl;
}

TYPED_TEST(DiscreteDistTest, poisson)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::poisson<Matrix>(8, 8, gen, 1);
	mat = Eigen::Rand::poisson<Matrix>(3, 3, gen, 1);
	mat = Eigen::Rand::poisson<Matrix>(5, 5, gen, 1);
	mat = Eigen::Rand::poisson<Matrix>(5, 3, gen, 1);
	mat = Eigen::Rand::poisson<Matrix>(1, 3, gen, 1);
	std::cout << mat << std::endl;
}

TYPED_TEST(DiscreteDistTest, binomial)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::binomial<Matrix>(8, 8, gen, 10, 0.5);
	mat = Eigen::Rand::binomial<Matrix>(3, 3, gen, 10, 0.5);
	mat = Eigen::Rand::binomial<Matrix>(5, 5, gen, 10, 0.5);
	mat = Eigen::Rand::binomial<Matrix>(5, 3, gen, 10, 0.5);
	mat = Eigen::Rand::binomial<Matrix>(1, 3, gen, 10, 0.5);
	std::cout << mat << std::endl;
}

TYPED_TEST(DiscreteDistTest, negativeBinomial)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::negativeBinomial<Matrix>(8, 8, gen, 10, 0.5);
	mat = Eigen::Rand::negativeBinomial<Matrix>(3, 3, gen, 10, 0.5);
	mat = Eigen::Rand::negativeBinomial<Matrix>(5, 5, gen, 10, 0.5);
	mat = Eigen::Rand::negativeBinomial<Matrix>(5, 3, gen, 10, 0.5);
	mat = Eigen::Rand::negativeBinomial<Matrix>(1, 3, gen, 10, 0.5);
	std::cout << mat << std::endl;
}

TYPED_TEST(DiscreteDistTest, geometric)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::geometric<Matrix>(8, 8, gen, 0.25);
	mat = Eigen::Rand::geometric<Matrix>(3, 3, gen, 0.25);
	mat = Eigen::Rand::geometric<Matrix>(5, 5, gen, 0.25);
	mat = Eigen::Rand::geometric<Matrix>(5, 3, gen, 0.25);
	mat = Eigen::Rand::geometric<Matrix>(1, 3, gen, 0.25);
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

TEST(Issue, 39)
{
	{
		Eigen::Matrix<double, 4, 1> mu1;
		mu1.setZero();
		Eigen::Matrix<double, 4, 4> cov1;
		cov1.setIdentity();
		Eigen::Matrix<double, 4, -1> samples;
		Eigen::Rand::MvNormalGen<double, 4> gen_init{ mu1, cov1 };
		std::random_device rd;
		std::mt19937 genn(rd());
		samples = gen_init.generate(genn, 10);
	}

	{
		Eigen::Matrix<double, 3, 1> mu1;
		mu1.setZero();
		Eigen::Matrix<double, 3, 3> cov1;
		cov1.setIdentity();
		Eigen::Matrix<double, 3, -1> samples;
		Eigen::Rand::MvNormalGen<double, 3> gen_init{ mu1, cov1 };
		std::random_device rd;
		std::mt19937 genn(rd());
		samples = gen_init.generate(genn, 10);
	}
}

TEST(Issue, 42)
{
	Eigen::Rand::P8_mt19937_64 generator{ 42 };

	Eigen::VectorXi p;
	p.resize(2);

	for (int i = 0; i < 100; i++) 
	{
		p = Eigen::Rand::uniformIntLike(p, generator, -1, 1);
		EXPECT_GE(p.minCoeff(), -1);
		EXPECT_LE(p.maxCoeff(), 1);
	}
}
