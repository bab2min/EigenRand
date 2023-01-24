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


template<typename Rng>
std::map<std::string, double> test_eigenrand(size_t size, const std::string& suffix, std::map<std::string, std::pair<double, double> >& results)
{
	std::map<std::string, double> ret;

	BenchmarkHelper bh{ ret, results };

	Rng urng;

	Eigen::ArrayXXi xi{ size, size };
	Eigen::ArrayXXf x{ size, size }, x2{ size, size };

	{
		auto scope = bh.measure("uniformRealV" + suffix, x);
		std::array<float, 5> a = { 1, 3, 5, 7, 9 };
		std::array<float, 5> b = { 10, 11, 12, 13, 14 };
		for (int i = 0; i < xi.cols(); ++i)
		{
			x.col(i).setConstant(a[i % a.size()]);
			x2.row(i).setConstant(b[i % b.size()]);
		}

		x = Eigen::Rand::uniformReal(urng, x, x2);
	}

	{
		auto scope = bh.measure("binomialV" + suffix, xi);
		std::array<int32_t, 3> n = { 20, 50, 100 };
		std::array<float, 3> p = { 0.5, 0.01, 0.75 };
		for (int i = 0; i < xi.cols(); ++i)
		{
			xi.col(i).setConstant(n[i % n.size()]);
			x.row(i).setConstant(p[i % n.size()]);
		}
		
		xi = Eigen::Rand::binomial(urng, xi, x);
	}

	{
		auto scope = bh.measure("normalV" + suffix, x);
		std::array<float, 5> a = { -2, -1, 0, 1, 2 };
		std::array<float, 5> b = { .1, .2, .3, .4, .5 };
		for (int i = 0; i < xi.cols(); ++i)
		{
			x.col(i).setConstant(a[i % a.size()]);
			x2.row(i).setConstant(b[i % b.size()]);
		}

		x = Eigen::Rand::normal(urng, x, x2);
	}

	{
		auto scope = bh.measure("cauchyV" + suffix, x);
		std::array<float, 5> a = { -2, -1, 0, 1, 2 };
		std::array<float, 5> b = { .1, .2, .3, .4, .5 };
		for (int i = 0; i < xi.cols(); ++i)
		{
			x.col(i).setConstant(a[i % a.size()]);
			x2.row(i).setConstant(b[i % b.size()]);
		}

		x = Eigen::Rand::cauchy(urng, x, x2);
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

	{
		auto scope = bh.measure("uniformRealV" + suffix, x);
		std::array<float, 5> a = { 1, 3, 5, 7, 9 };
		std::array<float, 5> b = { 10, 11, 12, 13, 14 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&](int r, int c)
		{
			std::uniform_real_distribution<float> dist{ a[c % a.size()], b[r % b.size()] };
			return dist(urng);
		});
	}

	{
		auto scope = bh.measure("binomialV" + suffix, xi);
		std::array<int32_t, 3> n = { 20, 50, 100 };
		std::array<float, 3> p = { 0.5, 0.01, 0.75 };
		xi = Eigen::ArrayXXi::NullaryExpr(size, size, [&](int r, int c) 
		{ 
			std::binomial_distribution<> dist{ n[c % n.size()], p[r % p.size()] };
			return dist(urng); 
		});
	}

	{
		auto scope = bh.measure("normalV" + suffix, x);
		std::array<float, 5> a = { -2, -1, 0, 1, 2 };
		std::array<float, 5> b = { .1, .2, .3, .4, .5 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&](int r, int c)
		{
			std::normal_distribution<float> dist{ a[c % a.size()], b[r % b.size()] };
			return dist(urng);
		});
	}

	{
		auto scope = bh.measure("cauchyV" + suffix, x);
		std::array<float, 5> a = { -2, -1, 0, 1, 2 };
		std::array<float, 5> b = { .1, .2, .3, .4, .5 };
		x = Eigen::ArrayXXf::NullaryExpr(size, size, [&](int r, int c)
		{
			std::cauchy_distribution<float> dist{ a[c % a.size()], b[r % b.size()] };
			return dist(urng);
		});
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

		for (auto& p : test_eigenrand<std::mt19937_64>(size, "\t:ERand", results))
		{
			time[p.first] += p.second;
			timeSq[p.first] += p.second * p.second;
		}

#if defined(EIGEN_VECTORIZE_SSE2) || defined(EIGEN_VECTORIZE_AVX) || defined(EIGEN_VECTORIZE_NEON)
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
