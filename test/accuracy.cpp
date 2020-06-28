/*
compare the distribution from sampling result 
with the original probabilistic distribution
by using Earth Mover's Distance(EMD).

Given two probability density function a and b, the EMD is calculated as follows:

  EMD(a, b) := integral_{x=-inf}^{+inf} { | A(x) - B(x) | }
where A and B is cumulative distribution function of a and b. 

*/

#include <map>
#include <string>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <random>

#include <EigenRand/EigenRand>

template<typename ArrTy, typename PdfFn>
double calc_emd_with_pdf(ArrTy&& arr, const std::tuple<PdfFn, double, double>& cdf_range, size_t step)
{
	double ret = 0;
	auto arr_begin = arr.data();
	auto arr_end = arr_begin + arr.size();
	std::sort(arr_begin, arr_end);
	const auto& pdf = std::get<0>(cdf_range);
	const auto& lb = std::get<1>(cdf_range), &rb = std::get<2>(cdf_range);

	Eigen::ArrayXf cdf{ step };
	double acc = 0;
	for (size_t i = 0; i < step; ++i)
	{
		const double x = lb + (rb - lb) * i / step;
		cdf[i] = acc;
		acc += pdf(x);
	}
	cdf /= acc;

	size_t arr_p = 0;
	for (size_t i = 0; i < step; ++i)
	{
		const double x = lb + (rb - lb) * i / step;
		arr_p = std::find_if(arr_begin + arr_p, arr_end, [&](double a) { return a > x; }) - arr_begin;
		double arr_cum = (double)arr_p / arr.size();
		ret += std::abs(cdf[i] - arr_cum);
	}
	return ret * (rb - lb) / step;
}


template<typename ArrTy, typename CdfFn>
double calc_emd_with_cdf(ArrTy&& arr, const std::tuple<CdfFn, double, double>& cdf_range, size_t step)
{
	double ret = 0;
	auto arr_begin = arr.data();
	auto arr_end = arr_begin + arr.size();
	std::sort(arr_begin, arr_end);
	const auto& cdf = std::get<0>(cdf_range);
	const auto& lb = std::get<1>(cdf_range), &rb = std::get<2>(cdf_range);

	size_t arr_p = 0;
	for (size_t i = 0; i < step; ++i)
	{
		const double x = lb + (rb - lb) * i / step;
		arr_p = std::find_if(arr_begin + arr_p, arr_end, [&](double a) { return a > x; }) - arr_begin;
		double arr_cum = (double)arr_p / arr.size();
		ret += std::abs(cdf(x) - arr_cum);
	}
	return ret * (rb - lb) / step;
}

double gamma_dist(double x, double alpha, double beta)
{
	if (x <= 0) return 0;
	return std::exp(-x / beta)
		/ (std::tgamma(alpha) * std::pow(beta, alpha))
		* std::pow(x, alpha - 1);
}

double chisquared_dist(double x, double n)
{
	if (x <= 0) return 0;
	return std::pow(x, n / 2 - 1) * std::exp(-x / 2) 
		/ std::pow(2, n / 2) / std::tgamma(n / 2);
}

double student_t_dist(double x, double n)
{
	return std::pow(1 + x * x /n, -(n + 1) / 2);
}

auto balanced_cdf = std::make_tuple([](double x) { return (x + 1) / 2; }, -1., 1.);
auto ur_cdf = std::make_tuple([](double x) { return x; }, 0., 1.);
auto normal_cdf = std::make_tuple([](double x) { return (1 + std::erf(x / std::sqrt(2))) / 2; }, -8., 8.);
auto lognormal_cdf = std::make_tuple([](double x) { return (1 + std::erf(std::log(x) / std::sqrt(2))) / 2; }, 0., 4.);
auto exp_cdf = std::make_tuple([](double x) { return 1 - std::exp(-x); }, 0., 32.);
auto weibull_cdf = std::make_tuple([](double x) { return 1 - std::exp(-std::pow(x, 2)); }, 0., 32.);
auto extreme_value_cdf = std::make_tuple([](double x) { return std::exp(-std::exp(1 - x)); }, -8., 24.);
auto gamma11_pdf = std::make_tuple([](double x) { return gamma_dist(x, 1, 1); }, 0., 16.);
auto gamma51_pdf = std::make_tuple([](double x) { return gamma_dist(x, 5, 1); }, 0., 16.);
auto gamma21_pdf = std::make_tuple([](double x) { return gamma_dist(x, 0.2, 1); }, 0., 16.);
auto chisquared_pdf = std::make_tuple([](double x) { return chisquared_dist(x, 7); }, 0., 64.);
auto cauchy_cdf = std::make_tuple([](double x) { return std::atan(x) / Eigen::internal::constant::pi + 0.5; }, -16., 16.);
auto student5_pdf = std::make_tuple([](double x) { return student_t_dist(x, 5); }, -8., 8.);
auto student20_pdf = std::make_tuple([](double x) { return student_t_dist(x, 20); }, -8., 8.);

template<typename Rng>
std::map<std::string, double> test_eigenrand(size_t size, size_t step, size_t seed)
{
	std::map<std::string, double> ret;
	Eigen::ArrayXf arr{ size };
	Eigen::ArrayXd arrd{ size };
	Rng urng{ seed };

	arr = Eigen::Rand::balancedLike(arr, urng);
	ret["balanced"] = calc_emd_with_cdf(arr, balanced_cdf, step);

	arr = Eigen::Rand::uniformRealLike(arr, urng);
	ret["uniformReal"] = calc_emd_with_cdf(arr, ur_cdf, step);

	arrd = Eigen::Rand::uniformRealLike(arrd, urng);
	ret["uniformReal/double"] = calc_emd_with_cdf(arrd, ur_cdf, step);

	arr = Eigen::Rand::normalLike(arr, urng);
	ret["normal"] = calc_emd_with_cdf(arr, normal_cdf, step);

	arr = Eigen::Rand::lognormalLike(arr, urng);
	ret["lognormal"] = calc_emd_with_cdf(arr, lognormal_cdf, step);

	arr = Eigen::Rand::gammaLike(arr, urng, 1, 1);
	ret["gamma(1,1)"] = calc_emd_with_pdf(arr, gamma11_pdf, step);

	arr = Eigen::Rand::gammaLike(arr, urng, 5, 1);
	ret["gamma(5,1)"] = calc_emd_with_pdf(arr, gamma51_pdf, step);

	arr = Eigen::Rand::gammaLike(arr, urng, 0.2, 1);
	ret["gamma(0.2,1)"] = calc_emd_with_pdf(arr, gamma21_pdf, step);

	arr = Eigen::Rand::exponentialLike(arr, urng);
	ret["exponential"] = calc_emd_with_cdf(arr, exp_cdf, step);

	arr = Eigen::Rand::weibullLike(arr, urng, 2);
	ret["weibull(2,1)"] = calc_emd_with_cdf(arr, weibull_cdf, step);

	arr = Eigen::Rand::extremeValueLike(arr, urng, 1, 1);
	ret["extremeValue(1,1)"] = calc_emd_with_cdf(arr, extreme_value_cdf, step);

	arr = Eigen::Rand::chiSquaredLike(arr, urng, 7);
	ret["chiSquared(7)"] = calc_emd_with_pdf(arr, chisquared_pdf, step);

	arr = Eigen::Rand::cauchyLike(arr, urng);
	ret["cauchy"] = calc_emd_with_cdf(arr, cauchy_cdf, step);

	arr = Eigen::Rand::studentTLike(arr, urng, 1);
	ret["studentT(1)"] = calc_emd_with_cdf(arr, cauchy_cdf, step);

	arr = Eigen::Rand::studentTLike(arr, urng, 5);
	ret["studentT(5)"] = calc_emd_with_pdf(arr, student5_pdf, step);

	arr = Eigen::Rand::studentTLike(arr, urng, 20);
	ret["studentT(20)"] = calc_emd_with_pdf(arr, student20_pdf, step);
	return ret;
}

std::map<std::string, double> test_old(size_t size, size_t step, size_t seed)
{
	std::map<std::string, double> ret;
	Eigen::ArrayXf arr{ size };

	arr = Eigen::ArrayXf::Random(size);
	ret["balanced"] = calc_emd_with_cdf(arr, balanced_cdf, step);
	return ret;
}


std::map<std::string, double> test_cpp11(size_t size, size_t step, size_t seed)
{
	std::map<std::string, double> ret;
	Eigen::ArrayXf arr{ size };
	Eigen::ArrayXd arrd{ size };
	std::mt19937_64 urng{ seed };

	{
		std::uniform_real_distribution<float> dist;
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["uniformReal"] = calc_emd_with_cdf(arr, ur_cdf, step);
	
	{
		std::uniform_real_distribution<double> dist;
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["uniformReal/double"] = calc_emd_with_cdf(arrd, ur_cdf, step);

	{
		std::normal_distribution<> dist;
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["normal"] = calc_emd_with_cdf(arr, normal_cdf, step);


	{
		std::lognormal_distribution<> dist;
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["lognormal"] = calc_emd_with_cdf(arr, lognormal_cdf, step);


	{
		std::gamma_distribution<> dist{ 1, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(1,1)"] = calc_emd_with_pdf(arr, gamma11_pdf, step);

	{
		std::gamma_distribution<> dist{ 5, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(5,1)"] = calc_emd_with_pdf(arr, gamma51_pdf, step);

	{
		std::gamma_distribution<> dist{ 0.2, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(0.2,1)"] = calc_emd_with_pdf(arr, gamma21_pdf, step);

	{
		std::exponential_distribution<> dist;
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["exponential"] = calc_emd_with_cdf(arr, exp_cdf, step);

	{
		std::weibull_distribution<> dist{ 2, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["weibull(2,1)"] = calc_emd_with_cdf(arr, weibull_cdf, step);

	{
		std::extreme_value_distribution<> dist{ 1, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["extremeValue(1,1)"] = calc_emd_with_cdf(arr, extreme_value_cdf, step);

	{
		std::chi_squared_distribution<> dist{ 7 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["chiSquared(7)"] = calc_emd_with_pdf(arr, chisquared_pdf, step);

	{
		std::cauchy_distribution<> dist{ 0, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["cauchy"] = calc_emd_with_cdf(arr, cauchy_cdf, step);

	{
		std::student_t_distribution<> dist{ 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(1)"] = calc_emd_with_cdf(arr, cauchy_cdf, step);

	{
		std::student_t_distribution<> dist{ 5 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(5)"] = calc_emd_with_pdf(arr, student5_pdf, step);

	{
		std::student_t_distribution<> dist{ 20 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(20)"] = calc_emd_with_pdf(arr, student20_pdf, step);
	return ret;
}


int main(int argc, char** argv)
{
	size_t size = 32768, step = size * 8, repeat = 50;

	if (argc > 1) size = std::stoi(argv[1]);
	if (argc > 2) step = std::stoi(argv[2]);

	std::map<std::string, double> err, errSq;

	for (size_t i = 0; i < repeat; ++i)
	{
		for (auto& p : test_eigenrand<std::mt19937_64>(size, step, 42 * i))
		{
			err[p.first + "\t:EigenRand"] += p.second;
			errSq[p.first + "\t:EigenRand"] += p.second * p.second;
		}
#if defined(EIGEN_VECTORIZE_SSE2) || defined(EIGEN_VECTORIZE_AVX)
		for (auto& p : test_eigenrand<Eigen::Rand::Vmt19937_64>(size, step, 42 * i))
		{
			err[p.first + "\t:ERand+Vrng"] += p.second;
			errSq[p.first + "\t:ERand+Vrng"] += p.second * p.second;
		}
#endif
		for (auto& p : test_old(size, step, 42 * i))
		{
			err[p.first + "\t:Old"] += p.second;
			errSq[p.first + "\t:Old"] += p.second * p.second;
		}

		for (auto& p : test_cpp11(size, step, 42 * i))
		{
			err[p.first + "\t:C++11"] += p.second;
			errSq[p.first + "\t:C++11"] += p.second * p.second;
		}
	}

	std::cout << "[Earth Mover's Distance] Mean (Stdev)" << std::endl;
	for (auto& p : err)
	{
		double mean = p.second / repeat;
		double var = (errSq[p.first] / repeat) - mean * mean;
		size_t sp = p.first.find('\t');
		std::cout << std::left << std::setw(28) << p.first.substr(0, sp);
		std::cout << std::setw(14) << p.first.substr(sp + 1);
		std::cout << ": " << mean << " (" << std::sqrt(var) << ")" << std::endl;
	}
	return 0;
}
