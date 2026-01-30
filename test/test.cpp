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

TYPED_TEST(ContinuousDistTest, balancedV)
{
	using Array = Eigen::Array<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Array a{ 10, 1 }, b{ 10, 1 }, c{ 10, 1 };
	a << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	b << 10, 12, 14, 16, 18, 20, 22, 24, 26, 28;

	c = Eigen::Rand::balanced(gen, a, b);
	EXPECT_TRUE((a <= c).all() && (c <= b).all());
	c = Eigen::Rand::balanced(gen, a, 11);
	EXPECT_TRUE((a <= c).all() && (c <= 11).all());
	c = Eigen::Rand::balanced(gen, 5, b);
	EXPECT_TRUE(((TypeParam)5 <= c).all() && (c <= b).all());
	std::cout << c << std::endl;
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

TYPED_TEST(ContinuousDistTest, uniformRealV)
{
	using Array = Eigen::Array<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Array a{ 10, 1 }, b{ 10, 1 }, c{ 10, 1 };
	a << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	b << 10, 12, 14, 16, 18, 20, 22, 24, 26, 28;

	c = Eigen::Rand::uniformReal(gen, a, b);
	EXPECT_TRUE((a <= c).all() && (c < b).all());
	c = Eigen::Rand::uniformReal(gen, a, 11);
	EXPECT_TRUE((a <= c).all() && (c < 11).all());
	c = Eigen::Rand::uniformReal(gen, 5, b);
	EXPECT_TRUE(((TypeParam)5 <= c).all() && (c < b).all());
	std::cout << c << std::endl;
}

TYPED_TEST(ContinuousDistTest, bernoulli)
{
	using Matrix = Eigen::Matrix<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Matrix mat;

	mat = Eigen::Rand::bernoulli<Matrix>(8, 8, gen);
	mat = Eigen::Rand::bernoulli<Matrix>(3, 3, gen);
	mat = Eigen::Rand::bernoulli<Matrix>(5, 5, gen);
	mat = Eigen::Rand::bernoulli<Matrix>(5, 3, gen);
	std::cout << mat << std::endl;
}

TEST(ContinuousDistTest, bernoulliV)
{
	using Array = Eigen::Array<float, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Array a{ 10, 1 }, c{ 10, 1 };
	a << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.;

	c = Eigen::Rand::bernoulli(gen, a);
	std::cout << c << std::endl;
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

TYPED_TEST(ContinuousDistTest, normalV)
{
	using Array = Eigen::Array<TypeParam, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Array a{ 10, 1 }, b{ 10, 1 }, c{ 10, 1 };
	a << 1, 2, 3, 4, 5, 6, 7, 8, 9, 10;
	b << 10, 12, 14, 16, 18, 20, 22, 24, 26, 28;

	c = Eigen::Rand::normal(gen, a, b);
	c = Eigen::Rand::normal(gen, a, 1);
	c = Eigen::Rand::normal(gen, 0, b);
	std::cout << c << std::endl;
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

TEST(DiscreteDistTest, binomialV)
{
	using Array = Eigen::Array<int32_t, -1, 1>;
	using FArray = Eigen::Array<float, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 42 };
	Array a{ 10 };
	FArray b{ 10 };
	
	a.setLinSpaced(5, 50);
	b.setLinSpaced(0.1, 1.0);

	auto c = Eigen::Rand::binomial(gen, a.replicate(1, 100).eval(), 0.5).eval();
	c = Eigen::Rand::binomial(gen, 50, b.replicate(1, 100).eval()).eval();
	c = Eigen::Rand::binomial(gen, a.replicate(1, 100).eval(), b.replicate(1, 100).eval()).eval();
	std::cout << c.leftCols(10) << std::endl;
	auto fc = c.template cast<float>().eval();
	auto mean = fc.rowwise().mean().eval();
	auto stdev = (fc.square().rowwise().mean() - mean.square()).sqrt().eval();
	for (int i = 0; i < mean.size(); ++i)
	{
		std::cout << mean[i] << " (" << stdev[i] << ")" << std::endl;
	}
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

class UniformIntRangeTest : public testing::Test
{
protected:
	Eigen::Rand::P8_mt19937_64 gen{ 12345 };

	// Helper to test a range and verify all values are within bounds
	template<typename Scalar>
	void testRange(Scalar minVal, Scalar maxVal, int iterations = 10000)
	{
		using Matrix = Eigen::Matrix<Scalar, -1, 1>;
		Matrix vec(100);

		for (int i = 0; i < iterations; ++i)
		{
			vec = Eigen::Rand::uniformInt<Matrix>(100, 1, gen, minVal, maxVal);
			EXPECT_GE(vec.minCoeff(), minVal)
				<< "Value below min for range [" << minVal << ", " << maxVal << "]";
			EXPECT_LE(vec.maxCoeff(), maxVal)
				<< "Value above max for range [" << minVal << ", " << maxVal << "]";
		}
	}

	// Helper to check distribution uniformity using chi-square test
	template<typename Scalar>
	void testUniformity(Scalar minVal, Scalar maxVal, int sampleSize = 100000)
	{
		using Matrix = Eigen::Matrix<Scalar, -1, 1>;
		Matrix vec(sampleSize);
		vec = Eigen::Rand::uniformInt<Matrix>(sampleSize, 1, gen, minVal, maxVal);

		int rangeSize = static_cast<int>(maxVal - minVal + 1);
		double expected = static_cast<double>(sampleSize) / rangeSize;

		std::map<Scalar, int> counts;
		for (int i = 0; i < sampleSize; ++i)
		{
			counts[vec(i)]++;
		}

		// Check all values in range appear
		EXPECT_EQ(counts.size(), static_cast<size_t>(rangeSize))
			<< "Not all values in range [" << minVal << ", " << maxVal << "] appeared";

		// Chi-square test for uniformity
		double chiSquare = 0.0;
		for (auto& kv : counts)
		{
			double diff = kv.second - expected;
			chiSquare += (diff * diff) / expected;
		}

		// Critical value for df=rangeSize-1, alpha=0.001 is approximately 3*df for large df
		// Using a generous threshold to avoid flaky tests
		double criticalValue = 3.0 * (rangeSize - 1) + 30;
		EXPECT_LT(chiSquare, criticalValue)
			<< "Distribution not uniform for range [" << minVal << ", " << maxVal
			<< "], chi-square=" << chiSquare;
	}
};

// Power-of-2 ranges (these should work with simple masking)
TEST_F(UniformIntRangeTest, PowerOf2_Range2)
{
	testRange<int>(0, 1);    // 2 values
	testUniformity<int>(0, 1);
}

TEST_F(UniformIntRangeTest, PowerOf2_Range4)
{
	testRange<int>(0, 3);    // 4 values
	testUniformity<int>(0, 3);
}

TEST_F(UniformIntRangeTest, PowerOf2_Range8)
{
	testRange<int>(0, 7);    // 8 values
	testUniformity<int>(0, 7);
}

TEST_F(UniformIntRangeTest, PowerOf2_Range16)
{
	testRange<int>(0, 15);   // 16 values
	testUniformity<int>(0, 15);
}

TEST_F(UniformIntRangeTest, PowerOf2_Range256)
{
	testRange<int>(0, 255);  // 256 values
	testUniformity<int>(0, 255);
}

// Non-power-of-2 ranges (these require rejection sampling - the bug we fixed)
TEST_F(UniformIntRangeTest, NonPowerOf2_Range3)
{
	testRange<int>(0, 2);    // 3 values
	testUniformity<int>(0, 2);
}

TEST_F(UniformIntRangeTest, NonPowerOf2_Range5)
{
	testRange<int>(0, 4);    // 5 values
	testUniformity<int>(0, 4);
}

TEST_F(UniformIntRangeTest, NonPowerOf2_Range6)
{
	testRange<int>(0, 5);    // 6 values
	testUniformity<int>(0, 5);
}

TEST_F(UniformIntRangeTest, NonPowerOf2_Range7)
{
	testRange<int>(0, 6);    // 7 values
	testUniformity<int>(0, 6);
}

TEST_F(UniformIntRangeTest, NonPowerOf2_Range10)
{
	testRange<int>(0, 9);    // 10 values
	testUniformity<int>(0, 9);
}

TEST_F(UniformIntRangeTest, NonPowerOf2_Range100)
{
	testRange<int>(0, 99);   // 100 values
	testUniformity<int>(0, 99);
}

TEST_F(UniformIntRangeTest, NonPowerOf2_Range101)
{
	testRange<int>(0, 100);  // 101 values (prime-ish)
	testUniformity<int>(0, 100);
}

TEST_F(UniformIntRangeTest, NonPowerOf2_Range1000)
{
	testRange<int>(0, 999);  // 1000 values
	// Skip uniformity test for large ranges (too slow)
}

// Negative ranges
TEST_F(UniformIntRangeTest, NegativeRange_Symmetric)
{
	testRange<int>(-50, 50); // 101 values centered at 0
	testUniformity<int>(-50, 50);
}

TEST_F(UniformIntRangeTest, NegativeRange_AllNegative)
{
	testRange<int>(-100, -1); // 100 negative values
	testUniformity<int>(-100, -1);
}

TEST_F(UniformIntRangeTest, NegativeRange_Issue42)
{
	// This is the Issue #42 test case
	testRange<int>(-1, 1);   // 3 values: -1, 0, 1
	testUniformity<int>(-1, 1);
}

// Single value range (edge case)
TEST_F(UniformIntRangeTest, SingleValue)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;
	Matrix vec(100);
	vec = Eigen::Rand::uniformInt<Matrix>(100, 1, gen, 42, 42);
	EXPECT_TRUE((vec.array() == 42).all());
}

// Large range with 32-bit integers
TEST_F(UniformIntRangeTest, LargeRange)
{
	using Matrix = Eigen::Matrix<int32_t, -1, 1>;
	Matrix vec(1000);

	int32_t minVal = -1000000;
	int32_t maxVal = 1000000;

	for (int iter = 0; iter < 100; ++iter)
	{
		vec = Eigen::Rand::uniformInt<Matrix>(1000, 1, gen, minVal, maxVal);
		EXPECT_GE(vec.minCoeff(), minVal);
		EXPECT_LE(vec.maxCoeff(), maxVal);
	}
}

// ============================================================================
// DiscreteGen Tests (Categorical Distribution)
// ============================================================================

class DiscreteGenTest : public testing::Test
{
protected:
	Eigen::Rand::P8_mt19937_64 gen{ 54321 };
};

// Test with small category count (uses CDF method, < 16 categories)
TEST_F(DiscreteGenTest, SmallCategories_CDF)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;

	// 5 categories with equal weights
	std::vector<double> weights = { 1.0, 1.0, 1.0, 1.0, 1.0 };
	Matrix vec(10000);
	vec = Eigen::Rand::discrete<Matrix>(10000, 1, gen, weights.begin(), weights.end());

	// All values should be in [0, 4]
	EXPECT_GE(vec.minCoeff(), 0);
	EXPECT_LE(vec.maxCoeff(), 4);

	// Count occurrences
	std::map<int, int> counts;
	for (int i = 0; i < 10000; ++i)
	{
		counts[vec(i)]++;
	}

	// Each should be roughly 2000 (within reasonable bounds)
	for (int i = 0; i < 5; ++i)
	{
		EXPECT_GT(counts[i], 1500) << "Category " << i << " underrepresented";
		EXPECT_LT(counts[i], 2500) << "Category " << i << " overrepresented";
	}
}

// Test with large category count (uses Alias method, >= 16 categories)
TEST_F(DiscreteGenTest, LargeCategories_Alias)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;

	// 20 categories with equal weights
	std::vector<double> weights(20, 1.0);
	Matrix vec(20000);
	vec = Eigen::Rand::discrete<Matrix>(20000, 1, gen, weights.begin(), weights.end());

	// All values should be in [0, 19]
	EXPECT_GE(vec.minCoeff(), 0);
	EXPECT_LE(vec.maxCoeff(), 19);

	// Count occurrences
	std::map<int, int> counts;
	for (int i = 0; i < 20000; ++i)
	{
		counts[vec(i)]++;
	}

	// Each should be roughly 1000 (within reasonable bounds)
	for (int i = 0; i < 20; ++i)
	{
		EXPECT_GT(counts[i], 500) << "Category " << i << " underrepresented";
		EXPECT_LT(counts[i], 1500) << "Category " << i << " overrepresented";
	}
}

// Test with unequal weights
TEST_F(DiscreteGenTest, UnequalWeights)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;

	// Category 0 has 10x the weight of others
	std::vector<double> weights = { 10.0, 1.0, 1.0, 1.0 };
	Matrix vec(13000);
	vec = Eigen::Rand::discrete<Matrix>(13000, 1, gen, weights.begin(), weights.end());

	std::map<int, int> counts;
	for (int i = 0; i < 13000; ++i)
	{
		counts[vec(i)]++;
	}

	// Category 0 should be ~10000, others ~1000 each
	EXPECT_GT(counts[0], 8000) << "Category 0 should dominate";
	EXPECT_LT(counts[0], 11500);

	for (int i = 1; i < 4; ++i)
	{
		EXPECT_GT(counts[i], 500) << "Category " << i << " should have ~1000";
		EXPECT_LT(counts[i], 2000);
	}
}

// Test with very skewed weights (one category dominates)
TEST_F(DiscreteGenTest, HighlySkewedWeights)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;

	// Category 2 has 100x the weight of others
	std::vector<double> weights = { 1.0, 1.0, 100.0, 1.0, 1.0 };
	Matrix vec(10400);
	vec = Eigen::Rand::discrete<Matrix>(10400, 1, gen, weights.begin(), weights.end());

	std::map<int, int> counts;
	for (int i = 0; i < 10400; ++i)
	{
		counts[vec(i)]++;
	}

	// Category 2 should be ~10000 (100/104 * 10400)
	EXPECT_GT(counts[2], 9000) << "Category 2 should dominate";
}

// Test with float precision
TEST_F(DiscreteGenTest, FloatPrecision)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;

	Eigen::Rand::DiscreteGen<int, float> dgen({ 1.0f, 2.0f, 3.0f, 4.0f });
	Matrix vec(10000);
	vec = dgen.generate<Matrix>(10000, 1, gen);

	EXPECT_GE(vec.minCoeff(), 0);
	EXPECT_LE(vec.maxCoeff(), 3);
}

// Test with double precision
TEST_F(DiscreteGenTest, DoublePrecision)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;

	Eigen::Rand::DiscreteGen<int, double> dgen({ 1.0, 2.0, 3.0, 4.0 });
	Matrix vec(10000);
	vec = dgen.generate<Matrix>(10000, 1, gen);

	EXPECT_GE(vec.minCoeff(), 0);
	EXPECT_LE(vec.maxCoeff(), 3);
}

// Test boundary: exactly 16 categories (transition point)
TEST_F(DiscreteGenTest, BoundaryCategories16)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;

	std::vector<double> weights(16, 1.0);
	Matrix vec(16000);
	vec = Eigen::Rand::discrete<Matrix>(16000, 1, gen, weights.begin(), weights.end());

	EXPECT_GE(vec.minCoeff(), 0);
	EXPECT_LE(vec.maxCoeff(), 15);

	// Verify all categories appear
	std::set<int> seen;
	for (int i = 0; i < 16000; ++i)
	{
		seen.insert(vec(i));
	}
	EXPECT_EQ(seen.size(), 16u);
}

// ============================================================================
// Additional uniformInt Tests
// ============================================================================

TEST(UniformIntAdditional, MultipleSizes)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 99999 };

	// Test with different vector sizes to exercise different code paths
	std::vector<int> sizes = { 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100 };

	for (int size : sizes)
	{
		Matrix vec(size);
		vec = Eigen::Rand::uniformInt<Matrix>(size, 1, gen, 0, 99);
		EXPECT_GE(vec.minCoeff(), 0) << "Size " << size << " below min";
		EXPECT_LE(vec.maxCoeff(), 99) << "Size " << size << " above max";
	}
}

// ============================================================================
// Matrix Shape Tests
// ============================================================================

TEST(MatrixShapeTest, UniformIntVariousShapes)
{
	using Matrix = Eigen::Matrix<int, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 11111 };

	// Test various matrix shapes
	std::vector<std::pair<int, int>> shapes = {
		{1, 1}, {1, 10}, {10, 1}, {3, 3}, {5, 7}, {7, 5},
		{16, 16}, {17, 17}, {100, 1}, {1, 100}, {32, 32}
	};

	for (auto& shape : shapes)
	{
		Matrix mat = Eigen::Rand::uniformInt<Matrix>(shape.first, shape.second, gen, 0, 99);
		EXPECT_EQ(mat.rows(), shape.first);
		EXPECT_EQ(mat.cols(), shape.second);
		EXPECT_GE(mat.minCoeff(), 0);
		EXPECT_LE(mat.maxCoeff(), 99);
	}
}

TEST(MatrixShapeTest, DiscreteVariousShapes)
{
	using Matrix = Eigen::Matrix<int, -1, -1>;
	Eigen::Rand::P8_mt19937_64 gen{ 22222 };

	std::vector<std::pair<int, int>> shapes = {
		{1, 1}, {1, 10}, {10, 1}, {3, 3}, {5, 7}, {7, 5},
		{16, 16}, {17, 17}
	};

	for (auto& shape : shapes)
	{
		Matrix mat = Eigen::Rand::discrete<Matrix>(shape.first, shape.second, gen, { 1, 2, 3, 4, 5 });
		EXPECT_EQ(mat.rows(), shape.first);
		EXPECT_EQ(mat.cols(), shape.second);
		EXPECT_GE(mat.minCoeff(), 0);
		EXPECT_LE(mat.maxCoeff(), 4);
	}
}

// ============================================================================
// Stress Tests for PacketFilter (via uniformInt)
// ============================================================================

TEST(StressTest, UniformIntManyIterations)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 77777 };

	// This stress test exercises the compress_append function heavily
	// Non-power-of-2 range requires rejection sampling
	Matrix vec(1000);

	int outOfRangeCount = 0;
	for (int iter = 0; iter < 10000; ++iter)
	{
		vec = Eigen::Rand::uniformInt<Matrix>(1000, 1, gen, 0, 99);
		if (vec.minCoeff() < 0 || vec.maxCoeff() > 99)
		{
			outOfRangeCount++;
		}
	}

	EXPECT_EQ(outOfRangeCount, 0) << "Out of range values detected in stress test";
}

TEST(StressTest, UniformIntSmallRange)
{
	using Matrix = Eigen::Matrix<int, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 88888 };

	// Very small range with high rejection rate
	Matrix vec(10000);

	for (int iter = 0; iter < 100; ++iter)
	{
		vec = Eigen::Rand::uniformInt<Matrix>(10000, 1, gen, 0, 2);
		EXPECT_GE(vec.minCoeff(), 0);
		EXPECT_LE(vec.maxCoeff(), 2);
	}
}

// ============================================================================
// Different Scalar Types
// ============================================================================

TEST(ScalarTypeTest, UniformInt_int32)
{
	using Matrix = Eigen::Matrix<int32_t, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 33333 };

	Matrix vec(1000);
	vec = Eigen::Rand::uniformInt<Matrix>(1000, 1, gen, -1000, 1000);
	EXPECT_GE(vec.minCoeff(), -1000);
	EXPECT_LE(vec.maxCoeff(), 1000);
}

TEST(ScalarTypeTest, UniformInt_LargeRange)
{
	// Test with large 32-bit range
	using Matrix = Eigen::Matrix<int32_t, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 44444 };

	Matrix vec(1000);
	vec = Eigen::Rand::uniformInt<Matrix>(1000, 1, gen, -100000000, 100000000);
	EXPECT_GE(vec.minCoeff(), -100000000);
	EXPECT_LE(vec.maxCoeff(), 100000000);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(EdgeCaseTest, UniformIntMaxRange)
{
	using Matrix = Eigen::Matrix<int32_t, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 55555 };

	// Large but not maximum range
	Matrix vec(100);
	vec = Eigen::Rand::uniformInt<Matrix>(100, 1, gen, -1000000, 1000000);
	EXPECT_GE(vec.minCoeff(), -1000000);
	EXPECT_LE(vec.maxCoeff(), 1000000);
}

TEST(EdgeCaseTest, DiscreteEmptyWeightHandling)
{
	// Test behavior with weights that sum to a small value
	using Matrix = Eigen::Matrix<int, -1, 1>;
	Eigen::Rand::P8_mt19937_64 gen{ 66666 };

	// Very small but non-zero weights
	std::vector<double> weights = { 0.001, 0.001, 0.001 };
	Matrix vec(1000);
	vec = Eigen::Rand::discrete<Matrix>(1000, 1, gen, weights.begin(), weights.end());

	EXPECT_GE(vec.minCoeff(), 0);
	EXPECT_LE(vec.maxCoeff(), 2);
}
