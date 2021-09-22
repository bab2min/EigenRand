#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <chrono>
#include <random>

#include <EigenRand/EigenRand>

class BenchmarkHelper
{
	std::map<std::string, double>& timing;
	std::map<std::string, std::pair<double, double> >& mean_var;
public:

	template<typename _EigenTy>
	class ScopeMeasure
	{
		BenchmarkHelper& bh;
		std::string name;
		_EigenTy& results;
		std::chrono::high_resolution_clock::time_point start;
	public:
		ScopeMeasure(BenchmarkHelper& _bh, 
			const std::string& _name,
			_EigenTy& _results) :
			bh{ _bh }, name{ _name }, results{ _results },
			start{ std::chrono::high_resolution_clock::now() }
		{
		}

		~ScopeMeasure()
		{
			if (!name.empty())
			{
				bh.timing[name] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

				double mean = results.template cast<double>().sum() / results.size();
				double sqmean = results.template cast<double>().array().square().sum() / results.size();
				bh.mean_var[name] = std::make_pair(mean, sqmean - mean * mean);
			}
		}
	};

	BenchmarkHelper(std::map<std::string, double>& _timing,
		std::map<std::string, std::pair<double, double>>& _mean_var)
		: timing{ _timing }, mean_var{ _mean_var }
	{
	}

	template<typename _EigenTy>
	ScopeMeasure<_EigenTy> measure(const std::string& name, _EigenTy& results)
	{
		return ScopeMeasure<_EigenTy>(*this, name, results);
	}
};

template<typename Rng>
std::map<std::string, double> test_eigenrand(size_t size, const std::string& suffix, std::map<std::string, std::pair<double, double> >& results)
{
	std::map<std::string, double> ret;

	BenchmarkHelper bh{ ret, results };

	Rng urng;

	Eigen::ArrayXXi xi{ size, size };
	Eigen::ArrayXXf x{ size, size };
	Eigen::ArrayXXd xd{ size, size };

	{
		auto scope = bh.measure("randBits/int" + suffix, xi);
		xi = Eigen::Rand::randBitsLike(xi, urng);
	}

	{
		auto scope = bh.measure("randBits/int/gen" + suffix, xi);
		Eigen::Rand::RandbitsGen<int32_t> gen;
		xi = gen.generateLike(xi, urng);
	}

	{
		auto scope = bh.measure("uniformInt/int(0~6)" + suffix, xi);
		xi = Eigen::Rand::uniformIntLike(xi, urng, 0, 6);
	}

	{
		auto scope = bh.measure("uniformInt/int(0~6)/gen" + suffix, xi);
		Eigen::Rand::UniformIntGen<int32_t> gen{ 0, 6 };
		xi = gen.generateLike(xi, urng);
	}

	{
		auto scope = bh.measure("uniformInt/int(0~63)" + suffix, xi);
		xi = Eigen::Rand::uniformIntLike(xi, urng, 0, 63);
	}

	{
		auto scope = bh.measure("uniformInt/int(5~79)" + suffix, xi);
		xi = Eigen::Rand::uniformIntLike(xi, urng, 5, 79);
	}

	{
		auto scope = bh.measure("uniformInt/int(0~100k)" + suffix, xi);
		xi = Eigen::Rand::uniformIntLike(xi, urng, 0, 100000);
	}

	{
		auto scope = bh.measure("discreteF/int(s=5)" + suffix, xi);
		xi = Eigen::Rand::discreteFLike(xi, urng, { 1, 2, 3, 4, 5 });
	}

	{
		auto scope = bh.measure("discreteF/int(s=5)/gen" + suffix, xi);
		Eigen::Rand::DiscreteGen<int32_t> gen{ 1, 2, 3, 4, 5 };
		xi = gen.generateLike(xi, urng);
	}

	{
		auto scope = bh.measure("discreteF/int(s=8)" + suffix, xi);
		xi = Eigen::Rand::discreteFLike(xi, urng, { 8, 7, 6, 5, 4, 3, 2, 1 });
	}

	{
		std::vector<double> ws(50);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discreteF/int(s=50)" + suffix, xi);
		xi = Eigen::Rand::discreteFLike(xi, urng, ws.begin(), ws.end());
	}

	{
		std::vector<double> ws(250);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discreteF/int(s=250)" + suffix, xi);
		xi = Eigen::Rand::discreteFLike(xi, urng, ws.begin(), ws.end());
	}

	{
		auto scope = bh.measure("discreteD/int(s=5)" + suffix, xi);
		xi = Eigen::Rand::discreteDLike(xi, urng, { 1, 2, 3, 4, 5 });
	}

	{
		auto scope = bh.measure("discreteD/int(s=8)" + suffix, xi);
		xi = Eigen::Rand::discreteDLike(xi, urng, { 8, 7, 6, 5, 4, 3, 2, 1 });
	}

	{
		std::vector<double> ws(50);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discreteD/int(s=50)" + suffix, xi);
		xi = Eigen::Rand::discreteDLike(xi, urng, ws.begin(), ws.end());
	}

	{
		std::vector<double> ws(250);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discreteD/int(s=250)" + suffix, xi);
		xi = Eigen::Rand::discreteDLike(xi, urng, ws.begin(), ws.end());
	}

	{
		auto scope = bh.measure("discrete/int(s=5)" + suffix, xi);
		xi = Eigen::Rand::discreteLike(xi, urng, { 1, 2, 3, 4, 5 });
	}

	{
		auto scope = bh.measure("discrete/int(s=8)" + suffix, xi);
		xi = Eigen::Rand::discreteLike(xi, urng, { 8, 7, 6, 5, 4, 3, 2, 1 });
	}

	{
		std::vector<double> ws(50);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discrete/int(s=50)" + suffix, xi);
		xi = Eigen::Rand::discreteLike(xi, urng, ws.begin(), ws.end());
	}

	{
		std::vector<double> ws(250);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discrete/int(s=250)" + suffix, xi);
		xi = Eigen::Rand::discreteLike(xi, urng, ws.begin(), ws.end());
	}

	{
		auto scope = bh.measure("bernoulli/int" + suffix, xi);
		xi = Eigen::Rand::bernoulli(xi, urng, 1. / 3);
	}

	{
		auto scope = bh.measure("bernoulli/int/gen" + suffix, xi);
		Eigen::Rand::BernoulliGen<int32_t> gen{ 1. / 3};
		xi = gen.generateLike(xi, urng);
	}

	{
		auto scope = bh.measure("bernoulli" + suffix, x);
		x = Eigen::Rand::bernoulli(x, urng, 1. / 3);
	}

	{
		auto scope = bh.measure("bernoulli/gen" + suffix, x);
		Eigen::Rand::BernoulliGen<float> gen{ 1. / 3 };
		x = gen.generateLike(x, urng);
	}

	{
		auto scope = bh.measure("bernoulli/double" + suffix, xd);
		xd = Eigen::Rand::bernoulli(xd, urng, 1. / 3);
	}

	{
		auto scope = bh.measure("bernoulli/double/gen" + suffix, x);
		Eigen::Rand::BernoulliGen<double> gen{ 1. / 3 };
		xd = gen.generateLike(xd, urng);
	}
	
	{
		auto scope = bh.measure("balanced" + suffix, x);
		x = Eigen::Rand::balancedLike(x, urng);
	}

	{
		auto scope = bh.measure("balanced/double" + suffix, xd);
		xd = Eigen::Rand::balancedLike(xd, urng);
	}

	{
		auto scope = bh.measure("balanced square" + suffix, x);
		x = Eigen::Rand::balancedLike(x, urng).square();
	}

	{
		auto scope = bh.measure("uniformReal" + suffix, x);
		x = Eigen::Rand::uniformRealLike(x, urng);
	}

	{
		auto scope = bh.measure("uniformReal/double" + suffix, xd);
		xd = Eigen::Rand::uniformRealLike(xd, urng);
	}

	{
		auto scope = bh.measure("uniformReal sqrt" + suffix, x);
		x = Eigen::Rand::uniformRealLike(x, urng).sqrt();
	}

	{
		auto scope = bh.measure("normal(0,1)" + suffix, x);
		x = Eigen::Rand::normalLike(x, urng);
	}

	{
		auto scope = bh.measure("normal(0,1)/double" + suffix, xd);
		xd = Eigen::Rand::normalLike(xd, urng);
	}

	{
		auto scope = bh.measure("normal(0,1) square" + suffix, x);
		x = Eigen::Rand::normalLike(x, urng).square();
	}

	{
		auto scope = bh.measure("normal(2,3)" + suffix, x);
		x = Eigen::Rand::normalLike(x, urng, 2, 3);
	}

	{
		auto scope = bh.measure("normal(2,3)/double" + suffix, xd);
		xd = Eigen::Rand::normalLike(xd, urng, 2, 3);
	}

	{
		auto scope = bh.measure("lognormal(0,1)" + suffix, x);
		x = Eigen::Rand::lognormalLike(x, urng);
	}
	
	{
		auto scope = bh.measure("lognormal(0,1)/double" + suffix, xd);
		xd = Eigen::Rand::lognormalLike(xd, urng);
	}

	{
		auto scope = bh.measure("exponential(1)" + suffix, x);
		x = Eigen::Rand::exponentialLike(x, urng);
	}

	{
		auto scope = bh.measure("exponential(1)/double" + suffix, xd);
		xd = Eigen::Rand::exponentialLike(xd, urng);
	}

	{
		auto scope = bh.measure("gamma(1,2)" + suffix, x);
		x = Eigen::Rand::gammaLike(x, urng, 1, 2);
	}

	{
		auto scope = bh.measure("gamma(1,2)/double" + suffix, xd);
		xd = Eigen::Rand::gammaLike(xd, urng, 1, 2);
	}

	{
		auto scope = bh.measure("gamma(5,3)" + suffix, x);
		x = Eigen::Rand::gammaLike(x, urng, 5, 3);
	}

	{
		auto scope = bh.measure("gamma(5,3)/gen" + suffix, x);
		Eigen::Rand::GammaGen<float> gen{ 5, 3 };
		x = gen.generateLike(x, urng);
	}

	{
		auto scope = bh.measure("gamma(5,3)/double" + suffix, xd);
		xd = Eigen::Rand::gammaLike(xd, urng, 5, 3);
	}

	{
		auto scope = bh.measure("gamma(0.2,1)" + suffix, x);
		x = Eigen::Rand::gammaLike(x, urng, 0.2, 1);
	}

	{
		auto scope = bh.measure("gamma(0.2,1)/double" + suffix, xd);
		xd = Eigen::Rand::gammaLike(xd, urng, 0.2, 1);
	}

	{
		auto scope = bh.measure("gamma(10.5,1)" + suffix, x);
		x = Eigen::Rand::gammaLike(x, urng, 10.5, 1);
	}

	{
		auto scope = bh.measure("gamma(10.5,1)/double" + suffix, xd);
		xd = Eigen::Rand::gammaLike(xd, urng, 10.5, 1);
	}

	{
		auto scope = bh.measure("weibull(2,3)" + suffix, x);
		x = Eigen::Rand::weibullLike(x, urng, 2, 3);
	}

	{
		auto scope = bh.measure("weibull(2,3)/double" + suffix, xd);
		xd = Eigen::Rand::weibullLike(xd, urng, 2, 3);
	}

	{
		auto scope = bh.measure("extremeValue(0,1)" + suffix, x);
		x = Eigen::Rand::extremeValueLike(x, urng, 0, 1);
	}

	{
		auto scope = bh.measure("extremeValue(0,1)/double" + suffix, xd);
		xd = Eigen::Rand::extremeValueLike(xd, urng, 0, 1);
	}

	{
		auto scope = bh.measure("chiSquared(15)" + suffix, x);
		x = Eigen::Rand::chiSquaredLike(x, urng, 15);
	}

	{
		auto scope = bh.measure("chiSquared(15)/double" + suffix, xd);
		xd = Eigen::Rand::chiSquaredLike(xd, urng, 15);
	}

	{
		auto scope = bh.measure("cauchy" + suffix, x);
		x = Eigen::Rand::cauchyLike(x, urng);
	}

	{
		auto scope = bh.measure("cauchy/double" + suffix, xd);
		xd = Eigen::Rand::cauchyLike(xd, urng);
	}

	{
		auto scope = bh.measure("studentT(1)" + suffix, x);
		x = Eigen::Rand::studentTLike(x, urng, 1);
	}

	{
		auto scope = bh.measure("studentT(1)/double" + suffix, xd);
		xd = Eigen::Rand::studentTLike(xd, urng, 1);
	}

	{
		auto scope = bh.measure("studentT(5)" + suffix, x);
		x = Eigen::Rand::studentTLike(x, urng, 5);
	}

	{
		auto scope = bh.measure("studentT(5)/double" + suffix, xd);
		xd = Eigen::Rand::studentTLike(xd, urng, 5);
	}

	{
		auto scope = bh.measure("studentT(20)" + suffix, x);
		x = Eigen::Rand::studentTLike(x, urng, 20);
	}

	{
		auto scope = bh.measure("fisherF(1,1)" + suffix, x);
		x = Eigen::Rand::fisherFLike(x, urng, 1, 1);
	}

	{
		auto scope = bh.measure("fisherF(1,1)/double" + suffix, xd);
		xd = Eigen::Rand::fisherFLike(xd, urng, 1, 1);
	}

	{
		auto scope = bh.measure("fisherF(5,1)" + suffix, x);
		x = Eigen::Rand::fisherFLike(x, urng, 5, 1);
	}

	{
		auto scope = bh.measure("fisherF(1,5)" + suffix, x);
		x = Eigen::Rand::fisherFLike(x, urng, 1, 5);
	}

	{
		auto scope = bh.measure("fisherF(5,5)" + suffix, x);
		x = Eigen::Rand::fisherFLike(x, urng, 5, 5);
	}

	{
		auto scope = bh.measure("fisherF(5,5)/double" + suffix, xd);
		xd = Eigen::Rand::fisherFLike(xd, urng, 5, 5);
	}

	{
		auto scope = bh.measure("poisson(1)" + suffix, xi);
		xi = Eigen::Rand::poissonLike(xi, urng, 1);
	}

	{
		auto scope = bh.measure("poisson(8)" + suffix, xi);
		xi = Eigen::Rand::poissonLike(xi, urng, 8);
	}

	{
		auto scope = bh.measure("poisson(16)" + suffix, xi);
		xi = Eigen::Rand::poissonLike(xi, urng, 16);
	}

	{
		auto scope = bh.measure("binomial(20,0.5)" + suffix, xi);
		xi = Eigen::Rand::binomialLike(xi, urng, 20, 0.5);
	}

	{
		auto scope = bh.measure("binomial(50,0.01)" + suffix, xi);
		xi = Eigen::Rand::binomialLike(xi, urng, 50, 0.01);
	}

	{
		auto scope = bh.measure("binomial(100,0.75)" + suffix, xi);
		xi = Eigen::Rand::binomialLike(xi, urng, 100, 0.75);
	}

	{
		auto scope = bh.measure("geometric(0.5)" + suffix, xi);
		xi = Eigen::Rand::geometricLike(xi, urng, 0.5);
	}

	{
		auto scope = bh.measure("negativeBinomial(10,0.5)" + suffix, xi);
		xi = Eigen::Rand::negativeBinomialLike(xi, urng, 10, 0.5);
	}

	{
		auto scope = bh.measure("negativeBinomial(20,0.25)" + suffix, xi);
		xi = Eigen::Rand::negativeBinomialLike(xi, urng, 20, 0.25);
	}

	{
		auto scope = bh.measure("negativeBinomial(30,0.75)" + suffix, xi);
		xi = Eigen::Rand::negativeBinomialLike(xi, urng, 30, 0.75);
	}
	return ret;
}

std::map<std::string, double> test_nullary(size_t size, const std::string& suffix, std::map<std::string, std::pair<double, double> >& results)
{
	std::map<std::string, double> ret;

	BenchmarkHelper bh{ ret, results };

	std::mt19937_64 urng;

	Eigen::ArrayXXi xi{ size, size };
	Eigen::ArrayXXf x{ size, size };
	Eigen::ArrayXXd xd{ size, size };

	{
		auto scope = bh.measure("randBits/int" + suffix, xi);
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return urng(); });
	}

	{
		auto scope = bh.measure("uniformInt/int(0~6)" + suffix, xi);
		std::uniform_int_distribution<> dist{ 0, 6 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("uniformInt/int(0~63)" + suffix, xi);
		std::uniform_int_distribution<> dist{ 0, 63 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}
	
	{
		auto scope = bh.measure("uniformInt/int(5~79)" + suffix, xi);
		std::uniform_int_distribution<> dist{ 5, 79 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("uniformInt/int(0~100k)" + suffix, xi);
		std::uniform_int_distribution<> dist{ 0, 100000 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("discreteD/int(s=5)" + suffix, xi);
		std::discrete_distribution<> dist{ {1, 2, 3, 4, 5} };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("discreteD/int(s=8)" + suffix, xi);
		std::discrete_distribution<> dist{ {8, 7, 6, 5, 4, 3, 2, 1} };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		std::vector<double> ws(50);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discreteD/int(s=50)" + suffix, xi);
		std::discrete_distribution<> dist{ ws.begin(), ws.end() };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		std::vector<double> ws(250);
		std::iota(ws.begin(), ws.end(), 1);
		std::shuffle(ws.begin(), ws.end(), std::mt19937_64{});
		auto scope = bh.measure("discreteD/int(s=250)" + suffix, xi);
		std::discrete_distribution<> dist{ ws.begin(), ws.end() };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}
	
	{
		auto scope = bh.measure("uniformReal" + suffix, x);
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return std::generate_canonical<float, 32>(urng); });
	}

	{
		auto scope = bh.measure("uniformReal sqrt" + suffix, x);
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return std::generate_canonical<float, 32>(urng); }).sqrt();
	}

	{
		auto scope = bh.measure("normal(0,1)" + suffix, x);
		std::normal_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("normal(0,1) square" + suffix, x);
		std::normal_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); }).square();
	}

	{
		auto scope = bh.measure("normal(2,3)" + suffix, x);
		std::normal_distribution<float> dist{ 2, 3 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("lognormal(0,1)" + suffix, x);
		std::lognormal_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("exponential(1)" + suffix, x);
		std::exponential_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gamma(1,2)" + suffix, x);
		std::gamma_distribution<float> dist{ 1, 2 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gamma(0.2,1)" + suffix, x);
		std::gamma_distribution<float> dist{ 0.2, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gamma(5,3)" + suffix, x);
		std::gamma_distribution<float> dist{ 5, 3 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gamma(5,3)/double" + suffix, xd);
		std::gamma_distribution<double> dist{ 5, 3 };
		xd = Eigen::ArrayXXd::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gamma(10.5,1)" + suffix, x);
		std::gamma_distribution<float> dist{ 10.5, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("weibull(2,3)" + suffix, x);
		std::weibull_distribution<float> dist{ 2, 3 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("extremeValue(0,1)" + suffix, x);
		std::extreme_value_distribution<float> dist{ 0, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("chiSquared(15)" + suffix, x);
		std::chi_squared_distribution<float> dist{ 15 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("cauchy" + suffix, x);
		std::cauchy_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("studentT(1)" + suffix, x);
		std::student_t_distribution<float> dist{ 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("studentT(5)" + suffix, x);
		std::student_t_distribution<float> dist{ 5 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("studentT(20)" + suffix, x);
		std::student_t_distribution<float> dist{ 20 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("fisherF(1,1)" + suffix, x);
		std::fisher_f_distribution<float> dist{ 1, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("fisherF(5,1)" + suffix, x);
		std::fisher_f_distribution<float> dist{ 5, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("fisherF(1,5)" + suffix, x);
		std::fisher_f_distribution<float> dist{ 1, 5 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("fisherF(5,5)" + suffix, x);
		std::fisher_f_distribution<float> dist{ 5, 5 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("poisson(1)" + suffix, xi);
		std::poisson_distribution<> dist{ 1 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("poisson(8)" + suffix, xi);
		std::poisson_distribution<> dist{ 8 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("poisson(16)" + suffix, xi);
		std::poisson_distribution<> dist{ 16 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("binomial(20,0.5)" + suffix, xi);
		std::binomial_distribution<> dist{ 20, 0.5 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("binomial(50,0.01)" + suffix, xi);
		std::binomial_distribution<> dist{ 50, 0.01 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("binomial(100,0.75)" + suffix, xi);
		std::binomial_distribution<> dist{ 100, 0.75 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("geometric(0.5)" + suffix, xi);
		std::geometric_distribution<> dist{ 0.5 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("negativeBinomial(10,0.5)" + suffix, xi);
		std::negative_binomial_distribution<> dist{ 10, 0.5 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("negativeBinomial(20,0.25)" + suffix, xi);
		std::negative_binomial_distribution<> dist{ 20, 0.25 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("negativeBinomial(30,0.75)" + suffix, xi);
		std::negative_binomial_distribution<> dist{ 30, 0.75 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return dist(urng); });
	}
	return ret;
}

std::map<std::string, double> test_old(size_t size, const std::string& suffix, std::map<std::string, std::pair<double, double> >& results)
{
	std::map<std::string, double> ret;

	BenchmarkHelper bh{ ret, results };

	Eigen::ArrayXXf x{ size, size };
	Eigen::ArrayXXd xd{ size, size };

	{
		auto scope = bh.measure("balanced" + suffix, x);
		x = Eigen::ArrayXXf::Random(size, size);
	}

	{
		auto scope = bh.measure("balanced/double" + suffix, xd);
		xd = Eigen::ArrayXXd::Random(size, size);
	}

	{
		auto scope = bh.measure("balanced square" + suffix, x);
		x = Eigen::ArrayXXf::Random(size, size).square();
	}
	return ret;
}

template<typename Rng>
std::map<std::string, double> test_rng(Rng&& rng, size_t size, const std::string& suffix, std::map<std::string, std::pair<double, double> >& results)
{
	std::map<std::string, double> ret;
	BenchmarkHelper bh{ ret, results };

	Eigen::Array<uint64_t, -1, -1> x{ size, size };

	{
		auto scope = bh.measure(suffix, x);
		for (size_t i = 0; i < size * size; ++i)
		{
			x.data()[i] = rng();
		}
	}
	return ret;
}

int main(int argc, char** argv)
{
	size_t size = 1000, repeat = 20;

	if (argc > 1) size = std::stoi(argv[1]);
	if (argc > 2) repeat = std::stoi(argv[2]);

	std::cout << "[Benchmark] Generating Random Matrix " << size << "x" << size 
		<< " " << repeat << " times" << std::endl;

	std::map<std::string, double> time, timeSq;
	std::map<std::string, std::pair<double, double> > results;

	for (size_t i = 0; i < repeat; ++i)
	{
		for (auto& p : test_rng(std::mt19937{}, size, "rng\tmt19937", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_rng(Eigen::Rand::makeUniversalRng<uint32_t>(Eigen::Rand::Vmt19937_64{}), size, "rng\tvmt19937_64/32", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_rng(Eigen::Rand::P8_mt19937_64_32{}, size, "rng\tP8_mt19937_64/32", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_rng(std::mt19937_64{}, size, "rng\tmt19937_64", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_rng(Eigen::Rand::makeUniversalRng<uint64_t>(Eigen::Rand::Vmt19937_64{}), size, "rng\tvmt19937_64", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_rng(Eigen::Rand::P8_mt19937_64{}, size, "rng\tP8_mt19937_64", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_old(size, "\t:Old", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_nullary(size, "\t:NullaryExpr", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		for (auto& p : test_eigenrand<std::mt19937_64>(size, "\t:ERand", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

#if defined(EIGEN_VECTORIZE_SSE2) || defined(EIGEN_VECTORIZE_AVX)
		for (auto& p : test_eigenrand<Eigen::Rand::Vmt19937_64>(size, "\t:ERand+vRNG", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}
#endif

	}

	std::cout << "[Average Time] Mean (Stdev)" << std::endl;
	for (auto& p : time)
	{
		double mean = p.second / repeat;
		double var = (timeSq[p.first] / repeat) - mean * mean;
		size_t sp = p.first.find('\t');
		std::cout << std::left << std::setw(28) << p.first.substr(0, sp);
		std::cout << std::setw(14) << p.first.substr(sp + 1);
		std::cout << ": " << mean * 1000 << " (" << std::sqrt(var) * 1000 << ")" << std::endl;
	}

	std::cout << std::endl << "[Statistics] Mean (Stdev)" << std::endl;
	for (auto& p : results)
	{
		size_t sp = p.first.find('\t');
		std::cout << std::left << std::setw(28) << p.first.substr(0, sp);
		std::cout << std::setw(14) << p.first.substr(sp + 1);
		std::cout << ": " << p.second.first << " (" << std::sqrt(p.second.second) << ")" << std::endl;
	}
	std::cout << std::endl;
	return 0;
}
