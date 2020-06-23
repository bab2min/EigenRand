#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
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
				bh.timing[name] = (std::chrono::high_resolution_clock::now() - start).count() / 1e+6;

				double mean = results.template cast<float>().sum() / results.size();
				double sqmean = results.template cast<float>().array().square().sum() / results.size();
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
		auto scope = bh.measure("normalDist(0,1)" + suffix, x);
		x = Eigen::Rand::normalDistLike(x, urng);
	}

	{
		auto scope = bh.measure("normalDist(0,1) square" + suffix, x);
		x = Eigen::Rand::normalDistLike(x, urng).square();
	}

	{
		auto scope = bh.measure("normalDist(2,3)" + suffix, x);
		x = Eigen::Rand::normalDistLike(x, urng, 2, 3);
	}

	{
		auto scope = bh.measure("lognormalDist(0,1)" + suffix, x);
		x = Eigen::Rand::lognormalDistLike(x, urng);
	}

	{
		auto scope = bh.measure("expDist(1)" + suffix, x);
		x = Eigen::Rand::expDistLike(x, urng);
	}

	{
		auto scope = bh.measure("gammaDist(1,2)" + suffix, x);
		x = Eigen::Rand::gammaDistLike(x, urng, 1, 2);
	}

	{
		auto scope = bh.measure("gammaDist(5,3)" + suffix, x);
		x = Eigen::Rand::gammaDistLike(x, urng, 5, 3);
	}

	{
		auto scope = bh.measure("gammaDist(0.2,1)" + suffix, x);
		x = Eigen::Rand::gammaDistLike(x, urng, 0.2, 1);
	}

	{
		auto scope = bh.measure("gammaDist(10.5,1)" + suffix, x);
		x = Eigen::Rand::gammaDistLike(x, urng, 10.5, 1);
	}

	{
		auto scope = bh.measure("weibullDist(2,3)" + suffix, x);
		x = Eigen::Rand::weibullDistLike(x, urng, 2, 3);
	}

	{
		auto scope = bh.measure("extremeValueDist(0,1)" + suffix, x);
		x = Eigen::Rand::extremeValueDistLike(x, urng, 0, 1);
	}

	{
		auto scope = bh.measure("chiSquaredDist(15)" + suffix, x);
		x = Eigen::Rand::chiSquaredDistLike(x, urng, 15);
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
		auto scope = bh.measure("uniformReal" + suffix, x);
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return std::generate_canonical<float, 32>(urng); });
	}

	{
		auto scope = bh.measure("uniformReal sqrt" + suffix, x);
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return std::generate_canonical<float, 32>(urng); }).sqrt();
	}

	{
		auto scope = bh.measure("normalDist(0,1)" + suffix, x);
		std::normal_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("normalDist(0,1) square" + suffix, x);
		std::normal_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); }).square();
	}

	{
		auto scope = bh.measure("normalDist(2,3)" + suffix, x);
		std::normal_distribution<float> dist{ 2, 3 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("lognormalDist(0,1)" + suffix, x);
		std::lognormal_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("expDist(1)" + suffix, x);
		std::exponential_distribution<float> dist;
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gammaDist(1,2)" + suffix, x);
		std::gamma_distribution<float> dist{ 1, 2 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gammaDist(0.2,1)" + suffix, x);
		std::gamma_distribution<float> dist{ 0.2, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gammaDist(5,3)" + suffix, x);
		std::gamma_distribution<float> dist{ 5, 3 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("gammaDist(10.5,1)" + suffix, x);
		std::gamma_distribution<float> dist{ 10.5, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("weibullDist(2,3)" + suffix, x);
		std::weibull_distribution<float> dist{ 2, 3 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("extremeValueDist(0,1)" + suffix, x);
		std::extreme_value_distribution<float> dist{ 0, 1 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
	}

	{
		auto scope = bh.measure("chiSquaredDist(15)" + suffix, x);
		std::chi_squared_distribution<float> dist{ 15 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&]() { return dist(urng); });
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


		for (auto& p : test_eigenrand<Eigen::Rand::vmt19937_64>(size, "\t:ERand+vRNG", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

	}

	std::cout << "[Average Time] Mean (Stdev)" << std::endl;
	for (auto& p : time)
	{
		double mean = p.second / repeat;
		double var = (timeSq[p.first] / repeat) - mean * mean;
		size_t sp = p.first.find('\t');
		std::cout << std::left << std::setw(28) << p.first.substr(0, sp);
		std::cout << std::setw(14) << p.first.substr(sp + 1);
		std::cout << ": " << mean << " (" << std::sqrt(var) << ")" << std::endl;
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
