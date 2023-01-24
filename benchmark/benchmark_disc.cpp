#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <chrono>
#include <random>

#include <EigenRand/EigenRand>
#include "utils.hpp"

static std::initializer_list<size_t> test_sizes = {
	2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	18, 20, 22, 24, 26, 28, 30, 32,
	40, 48, 56, 64,
	80, 96, 112, 128,
	160, 192, 224, 256,
	320, 384, 448, 512,
	640, 768, 896, 1024,
};

template<typename Rng>
std::map<std::string, double> test_eigenrand(size_t size, const std::string& suffix, std::map<std::string, std::pair<double, double> >& results)
{
	std::map<std::string, double> ret;

	BenchmarkHelper bh{ ret, results };

	Rng urng;

	Eigen::ArrayXXi xi{ size, size };
	
	/*for (size_t n : test_sizes)
	{
		auto dist = rand_vector<double>(n);
		auto scope = bh.measure("discreteF/int(s=" + format_digit(n, 4) + ")" + suffix, xi);
		xi = Eigen::Rand::discreteFLike(xi, urng, dist.begin(), dist.end());
	}*/

	for (size_t n : test_sizes)
	{
		auto dist = rand_vector<double>(n);
		auto scope = bh.measure("discreteF/int(s=" + format_digit(n, 4) + ")/gen" + suffix, xi);
		Eigen::Rand::DiscreteGen<int32_t> gen{ dist.begin(), dist.end() };
		xi = gen.generateLike(xi, urng);
	}

	/*for (size_t n : test_sizes)
	{
		auto dist = rand_vector<double>(n);
		auto scope = bh.measure("discreteD/int(s=" + format_digit(n, 4) + ")" + suffix, xi);
		xi = Eigen::Rand::discreteDLike(xi, urng, dist.begin(), dist.end());
	}*/

	for (size_t n : test_sizes)
	{
		auto dist = rand_vector<double>(n);
		auto scope = bh.measure("discreteD/int(s=" + format_digit(n, 4) + ")/gen" + suffix, xi);
		Eigen::Rand::DiscreteGen<int32_t, double> gen{ dist.begin(), dist.end() };
		xi = gen.generateLike(xi, urng);
	}

	/*for (size_t n : test_sizes)
	{
		auto dist = rand_vector<double>(n);
		auto scope = bh.measure("discrete/int(s=" + format_digit(n, 4) + ")" + suffix, xi);
		xi = Eigen::Rand::discreteLike(xi, urng, dist.begin(), dist.end());
	}*/

	for (size_t n : test_sizes)
	{
		auto dist = rand_vector<double>(n);
		auto scope = bh.measure("discrete/int(s=" + format_digit(n, 4) + ")/gen" + suffix, xi);
		Eigen::Rand::DiscreteGen<int32_t, int32_t> gen{ dist.begin(), dist.end() };
		xi = gen.generateLike(xi, urng);
	}
	return ret;
}

std::map<std::string, double> test_nullary(size_t size, const std::string& suffix, std::map<std::string, std::pair<double, double> >& results)
{
	std::map<std::string, double> ret;

	BenchmarkHelper bh{ ret, results };

	std::mt19937_64 urng;

	Eigen::ArrayXXi xi{ size, size };

	for (size_t n : test_sizes)
	{
		auto dist = rand_vector<double>(n);
		auto scope = bh.measure("discreteD/int(s=" + format_digit(n, 4) + ")" + suffix, xi);
		std::discrete_distribution<> gen{ dist.begin(), dist.end() };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&]() { return gen(urng); });
	}
	return ret;
}

int main(int argc, char** argv)
{
	size_t size = 1000, repeat = 20;

	if (argc > 1) size = std::stoi(argv[1]);
	if (argc > 2) repeat = std::stoi(argv[2]);

	std::cout << "SIMD arch: " << Eigen::SimdInstructionSetsInUse() << std::endl;
	std::cout << "[Benchmark] Generating Random Matrix " << size << "x" << size 
		<< " " << repeat << " times" << std::endl;

	std::map<std::string, double> time, timeSq;
	std::map<std::string, std::pair<double, double> > results;

	for (size_t i = 0; i < repeat; ++i)
	{
		for (auto& p : test_nullary(size, "\t:NullaryExpr", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

		/*for (auto& p : test_eigenrand<std::mt19937_64>(size, "\t:ERand", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}*/

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

	/*std::cout << std::endl << "[Statistics] Mean (Stdev)" << std::endl;
	for (auto& p : results)
	{
		size_t sp = p.first.find('\t');
		std::cout << std::left << std::setw(28) << p.first.substr(0, sp);
		std::cout << std::setw(14) << p.first.substr(sp + 1);
		std::cout << ": " << p.second.first << " (" << std::sqrt(p.second.second) << ")" << std::endl;
	}
	std::cout << std::endl;*/
	return 0;
}
