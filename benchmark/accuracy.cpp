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

	Eigen::ArrayXd cdf{ step };
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


template<typename ArrTy, typename PmfFn>
double calc_kldiv(ArrTy&& arr, const std::tuple<PmfFn, int, int>& pmf_range)
{
	double ret = 0;
	const auto& pmf = std::get<0>(pmf_range);
	const auto& b = std::get<1>(pmf_range), e = std::get<2>(pmf_range);
	Eigen::ArrayXd p{ e - b + 1 }, q{ e - b + 1};
	p.setZero();
	for (int i = 0; i < arr.size(); ++i)
	{
		if (arr[i] < b || arr[i] > e) continue;
		p[arr[i] - b] += 1;
	}
	p /= arr.size();
	
	for (int i = b; i <= e; ++i) q[i - b] = pmf(i);
	q /= q.sum();

	return (((p / q) + 1e-12).log() * p).sum();
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

double fisher_f_dist(double x, double m, double n)
{
	if (x <= 0) return 0;
	return std::sqrt((std::pow(m * x, m) * std::pow(n, n)) / std::pow(m * x + n, m + n)) / x;
}

double combination(int n, int c)
{
	if (n == 0 || n == 1 || c == 0 || c == n) return 1;
	c = std::min(c, n - c);
	
	double ret = 1;
	for (int i = 0; i < c; ++i)
	{
		ret *= (n - i);
		ret /= i + 1;
	}
	return ret;
}

double binomial_dist(int x, int n, double p)
{
	return combination(n, x) * std::pow(p, x) * std::pow(1 - p, n - x);
}

double negative_binomial_dist(int x, int n, double p)
{
	return combination(n + x - 1, x) * std::pow(1 - p, x);
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
auto cauchy_cdf = std::make_tuple([](double x) { return std::atan(x) / Eigen::Rand::constant::pi + 0.5; }, -16., 16.);
auto student5_pdf = std::make_tuple([](double x) { return student_t_dist(x, 5); }, -8., 8.);
auto student20_pdf = std::make_tuple([](double x) { return student_t_dist(x, 20); }, -8., 8.);
auto fisher11_cdf = std::make_tuple([](double x) { return 2 * std::atan(std::sqrt(x)) / Eigen::Rand::constant::pi; }, 0., 256.);
auto fisher55_pdf = std::make_tuple([](double x) { return fisher_f_dist(x, 5, 5); }, 0., 16.);

auto uniform10_pmf = std::make_tuple([](int x) { return 1; }, 0, 9);
auto discrete_pmf = std::make_tuple([](int x) { return x + 1; }, 0, 5);
auto poisson1_pmf = std::make_tuple([](int x) { return std::pow(1, x) / std::tgamma(x + 1); }, 0, 127);
auto poisson16_pmf = std::make_tuple([](int x) { return std::pow(16, x) / std::tgamma(x + 1); }, 0, 127);
auto binomial1_pmf = std::make_tuple([](int x) { return binomial_dist(x, 10, 0.5); }, 0, 10);
auto binomial2_pmf = std::make_tuple([](int x) { return binomial_dist(x, 30, 0.75); }, 0, 30);
auto binomial3_pmf = std::make_tuple([](int x) { return binomial_dist(x, 50, 0.25); }, 0, 50);
auto negbinomial1_pmf = std::make_tuple([](int x) { return negative_binomial_dist(x, 10, 0.5); }, 0, 127);
auto negbinomial2_pmf = std::make_tuple([](int x) { return negative_binomial_dist(x, 20, 0.25); }, 0, 127);
auto negbinomial3_pmf = std::make_tuple([](int x) { return negative_binomial_dist(x, 30, 0.75); }, 0, 127);
auto geometric25_pmf = std::make_tuple([](int x) { return std::pow(1 - 0.25, x); }, 0, 127);
auto geometric75_pmf = std::make_tuple([](int x) { return std::pow(1 - 0.75, x); }, 0, 127);

template<typename Rng>
std::map<std::string, double> test_eigenrand_cont(size_t size, size_t step, size_t seed)
{
	std::map<std::string, double> ret;
	Eigen::ArrayXf arr{ size };
	Rng urng{ seed };

	arr = Eigen::Rand::balancedLike(arr, urng);
	ret["balanced"] = calc_emd_with_cdf(arr, balanced_cdf, step);

	arr = Eigen::Rand::uniformRealLike(arr, urng);
	ret["uniformReal"] = calc_emd_with_cdf(arr, ur_cdf, step);

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

	arr = Eigen::Rand::fisherFLike(arr, urng, 1, 1);
	ret["fisherF(1,1)"] = calc_emd_with_cdf(arr, fisher11_cdf, step);

	arr = Eigen::Rand::fisherFLike(arr, urng, 5, 5);
	ret["fisherF(5,5)"] = calc_emd_with_pdf(arr, fisher55_pdf, step);

#ifdef TEST_DOUBLE
	Eigen::ArrayXd arrd{ size };
	arrd = Eigen::Rand::uniformRealLike(arrd, urng);
	ret["uniformReal/double"] = calc_emd_with_cdf(arrd, ur_cdf, step);

	arrd = Eigen::Rand::normalLike(arrd, urng);
	ret["normal/double"] = calc_emd_with_cdf(arrd, normal_cdf, step);

	arrd = Eigen::Rand::lognormalLike(arrd, urng);
	ret["lognormal/double"] = calc_emd_with_cdf(arrd, lognormal_cdf, step);

	arrd = Eigen::Rand::gammaLike(arrd, urng, 1, 1);
	ret["gamma(1,1)/double"] = calc_emd_with_pdf(arrd, gamma11_pdf, step);

	arrd = Eigen::Rand::gammaLike(arrd, urng, 5, 1);
	ret["gamma(5,1)/double"] = calc_emd_with_pdf(arrd, gamma51_pdf, step);

	arrd = Eigen::Rand::gammaLike(arrd, urng, 0.2, 1);
	ret["gamma(0.2,1)/double"] = calc_emd_with_pdf(arrd, gamma21_pdf, step);

	arrd = Eigen::Rand::exponentialLike(arrd, urng);
	ret["exponential/double"] = calc_emd_with_cdf(arrd, exp_cdf, step);

	arrd = Eigen::Rand::weibullLike(arrd, urng, 2);
	ret["weibull(2,1)/double"] = calc_emd_with_cdf(arrd, weibull_cdf, step);

	arrd = Eigen::Rand::extremeValueLike(arrd, urng, 1, 1);
	ret["extremeValue(1,1)/double"] = calc_emd_with_cdf(arrd, extreme_value_cdf, step);

	arrd = Eigen::Rand::chiSquaredLike(arrd, urng, 7);
	ret["chiSquared(7)/double"] = calc_emd_with_pdf(arrd, chisquared_pdf, step);

	arrd = Eigen::Rand::cauchyLike(arrd, urng);
	ret["cauchy/double"] = calc_emd_with_cdf(arrd, cauchy_cdf, step);

	arrd = Eigen::Rand::studentTLike(arrd, urng, 1);
	ret["studentT(1)/double"] = calc_emd_with_cdf(arrd, cauchy_cdf, step);

	arrd = Eigen::Rand::studentTLike(arrd, urng, 5);
	ret["studentT(5)/double"] = calc_emd_with_pdf(arrd, student5_pdf, step);

	arrd = Eigen::Rand::studentTLike(arrd, urng, 20);
	ret["studentT(20)/double"] = calc_emd_with_pdf(arrd, student20_pdf, step);

	arrd = Eigen::Rand::fisherFLike(arrd, urng, 1, 1);
	ret["fisherF(1,1)/double"] = calc_emd_with_cdf(arrd, fisher11_cdf, step);

	arrd = Eigen::Rand::fisherFLike(arrd, urng, 5, 5);
	ret["fisherF(5,5)/double"] = calc_emd_with_pdf(arrd, fisher55_pdf, step);
#endif
	return ret;
}

template<typename Rng>
std::map<std::string, double> test_eigenrand_disc(size_t size, size_t step, size_t seed)
{
	std::map<std::string, double> ret;
	Eigen::ArrayXi arri{ size };
	Rng urng{ seed };

	arri = Eigen::Rand::uniformIntLike(arri, urng, 0, 9);
	ret["uniformInt(0,9)"] = calc_kldiv(arri, uniform10_pmf);

	arri = Eigen::Rand::discreteLike(arri, urng, { 1, 2, 3, 4, 5, 6 });
	ret["discrete(1,2,3,4,5,6)"] = calc_kldiv(arri, discrete_pmf);

	arri = Eigen::Rand::poissonLike(arri, urng, 1);
	ret["poisson(1)"] = calc_kldiv(arri, poisson1_pmf);

	arri = Eigen::Rand::poissonLike(arri, urng, 16);
	ret["poisson(16)"] = calc_kldiv(arri, poisson16_pmf);

	arri = Eigen::Rand::binomialLike(arri, urng, 10, 0.5);
	ret["binomial(10,0.5)"] = calc_kldiv(arri, binomial1_pmf);

	arri = Eigen::Rand::binomialLike(arri, urng, 30, 0.75);
	ret["binomial(30,0.75)"] = calc_kldiv(arri, binomial2_pmf);

	arri = Eigen::Rand::binomialLike(arri, urng, 50, 0.25);
	ret["binomial(50,0.25)"] = calc_kldiv(arri, binomial3_pmf);

	arri = Eigen::Rand::negativeBinomialLike(arri, urng, 10, 0.5);
	ret["negativeBinomial(10,0.5)"] = calc_kldiv(arri, negbinomial1_pmf);

	arri = Eigen::Rand::negativeBinomialLike(arri, urng, 20, 0.25);
	ret["negativeBinomial(20,0.25)"] = calc_kldiv(arri, negbinomial2_pmf);

	arri = Eigen::Rand::negativeBinomialLike(arri, urng, 30, 0.75);
	ret["negativeBinomial(30,0.75)"] = calc_kldiv(arri, negbinomial3_pmf);

	arri = Eigen::Rand::geometricLike(arri, urng, 0.25);
	ret["geometric(0.25)"] = calc_kldiv(arri, geometric25_pmf);

	arri = Eigen::Rand::geometricLike(arri, urng, 0.75);
	ret["geometric(0.75)"] = calc_kldiv(arri, geometric75_pmf);
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

std::map<std::string, double> test_cpp11_cont(size_t size, size_t step, size_t seed)
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
		std::normal_distribution<double> dist;
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["normal/double"] = calc_emd_with_cdf(arrd, normal_cdf, step);

	{
		std::lognormal_distribution<> dist;
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["lognormal"] = calc_emd_with_cdf(arr, lognormal_cdf, step);

	{
		std::lognormal_distribution<double> dist;
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["lognormal/double"] = calc_emd_with_cdf(arrd, lognormal_cdf, step);

	{
		std::gamma_distribution<> dist{ 1, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(1,1)"] = calc_emd_with_pdf(arr, gamma11_pdf, step);

	{
		std::gamma_distribution<double> dist{ 1, 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(1,1)/double"] = calc_emd_with_pdf(arrd, gamma11_pdf, step);

	{
		std::gamma_distribution<> dist{ 5, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(5,1)"] = calc_emd_with_pdf(arr, gamma51_pdf, step);

	{
		std::gamma_distribution<double> dist{ 5, 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(5,1)/double"] = calc_emd_with_pdf(arrd, gamma51_pdf, step);

	{
		std::gamma_distribution<> dist{ 0.2, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(0.2,1)"] = calc_emd_with_pdf(arr, gamma21_pdf, step);

	{
		std::gamma_distribution<double> dist{ 0.2, 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["gamma(0.2,1)/double"] = calc_emd_with_pdf(arrd, gamma21_pdf, step);

	{
		std::exponential_distribution<> dist;
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["exponential"] = calc_emd_with_cdf(arr, exp_cdf, step);

	{
		std::exponential_distribution<double> dist;
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["exponential/double"] = calc_emd_with_cdf(arrd, exp_cdf, step);

	{
		std::weibull_distribution<> dist{ 2, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["weibull(2,1)"] = calc_emd_with_cdf(arr, weibull_cdf, step);

	{
		std::weibull_distribution<double> dist{ 2, 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["weibull(2,1)/double"] = calc_emd_with_cdf(arrd, weibull_cdf, step);

	{
		std::extreme_value_distribution<> dist{ 1, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["extremeValue(1,1)"] = calc_emd_with_cdf(arr, extreme_value_cdf, step);

	{
		std::extreme_value_distribution<double> dist{ 1, 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["extremeValue(1,1)/double"] = calc_emd_with_cdf(arrd, extreme_value_cdf, step);

	{
		std::chi_squared_distribution<> dist{ 7 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["chiSquared(7)"] = calc_emd_with_pdf(arr, chisquared_pdf, step);

	{
		std::chi_squared_distribution<double> dist{ 7 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["chiSquared(7)/double"] = calc_emd_with_pdf(arrd, chisquared_pdf, step);

	{
		std::cauchy_distribution<> dist{ 0, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["cauchy"] = calc_emd_with_cdf(arr, cauchy_cdf, step);

	{
		std::cauchy_distribution<double> dist{ 0, 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["cauchy/double"] = calc_emd_with_cdf(arrd, cauchy_cdf, step);

	{
		std::student_t_distribution<> dist{ 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(1)"] = calc_emd_with_cdf(arr, cauchy_cdf, step);

	{
		std::student_t_distribution<double> dist{ 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(1)/double"] = calc_emd_with_cdf(arrd, cauchy_cdf, step);

	{
		std::student_t_distribution<> dist{ 5 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(5)"] = calc_emd_with_pdf(arr, student5_pdf, step);

	{
		std::student_t_distribution<double> dist{ 5 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(5)/double"] = calc_emd_with_pdf(arrd, student5_pdf, step);

	{
		std::student_t_distribution<> dist{ 20 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(20)"] = calc_emd_with_pdf(arr, student20_pdf, step);

	{
		std::student_t_distribution<double> dist{ 20 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["studentT(20)/double"] = calc_emd_with_pdf(arrd, student20_pdf, step);

	{
		std::fisher_f_distribution<> dist{ 1, 1 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["fisherF(1,1)"] = calc_emd_with_cdf(arr, fisher11_cdf, step);

	{
		std::fisher_f_distribution<double> dist{ 1, 1 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["fisherF(1,1)/double"] = calc_emd_with_cdf(arrd, fisher11_cdf, step);

	{
		std::fisher_f_distribution<> dist{ 5, 5 };
		arr = Eigen::ArrayXf::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["fisherF(5,5)"] = calc_emd_with_pdf(arr, fisher55_pdf, step);

	{
		std::fisher_f_distribution<double> dist{ 5, 5 };
		arrd = Eigen::ArrayXd::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["fisherF(5,5)/double"] = calc_emd_with_pdf(arrd, fisher55_pdf, step);
	return ret;
}

std::map<std::string, double> test_cpp11_disc(size_t size, size_t step, size_t seed)
{
	std::map<std::string, double> ret;
	Eigen::ArrayXi arri{ size };
	std::mt19937_64 urng{ seed };

	{
		std::uniform_int_distribution<> dist{ 0, 9 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["uniformInt(0,9)"] = calc_kldiv(arri, uniform10_pmf);

	{
		std::discrete_distribution<> dist{ 1, 2, 3, 4, 5, 6 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["discrete(1,2,3,4,5,6)"] = calc_kldiv(arri, discrete_pmf);

	{
		std::poisson_distribution<> dist{ 1 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["poisson(1)"] = calc_kldiv(arri, poisson1_pmf);

	{
		std::poisson_distribution<> dist{ 16 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["poisson(16)"] = calc_kldiv(arri, poisson16_pmf);

	{
		std::binomial_distribution<> dist{ 10, 0.5 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["binomial(10,0.5)"] = calc_kldiv(arri, binomial1_pmf);

	{
		std::binomial_distribution<> dist{ 30, 0.75 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["binomial(30,0.75)"] = calc_kldiv(arri, binomial2_pmf);

	{
		std::binomial_distribution<> dist{ 50, 0.25 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["binomial(50,0.25)"] = calc_kldiv(arri, binomial3_pmf);

	{
		std::negative_binomial_distribution<> dist{ 10, 0.5 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["negativeBinomial(10,0.5)"] = calc_kldiv(arri, negbinomial1_pmf);

	{
		std::negative_binomial_distribution<> dist{ 20, 0.25 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["negativeBinomial(20,0.25)"] = calc_kldiv(arri, negbinomial2_pmf);

	{
		std::negative_binomial_distribution<> dist{ 30, 0.75 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["negativeBinomial(30,0.75)"] = calc_kldiv(arri, negbinomial3_pmf);

	{
		std::geometric_distribution<> dist{ 0.25 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["geometric(0.25)"] = calc_kldiv(arri, geometric25_pmf);

	{
		std::geometric_distribution<> dist{ 0.75 };
		arri = Eigen::ArrayXi::NullaryExpr(size, [&]() { return dist(urng); });
	}
	ret["geometric(0.75)"] = calc_kldiv(arri, geometric75_pmf);
	return ret;
}

int main(int argc, char** argv)
{
	size_t size = 32768, step = size * 8, repeat = 50;

	if (argc > 1) size = std::stoi(argv[1]);
	if (argc > 2) step = std::stoi(argv[2]);

	std::map<std::string, double> err, errSq, kl, klSq;

	for (size_t i = 0; i < repeat; ++i)
	{
		std::cout << "Repeat " << i << " ..." << std::endl;
		for (auto& p : test_eigenrand_cont<std::mt19937_64>(size, step, 42 * i))
		{
			err[p.first + "\t:EigenRand"] += p.second;
			errSq[p.first + "\t:EigenRand"] += p.second * p.second;
		}

		for (auto& p : test_eigenrand_disc<std::mt19937_64>(size, step, 42 * i))
		{
			kl[p.first + "\t:EigenRand"] += p.second;
			klSq[p.first + "\t:EigenRand"] += p.second * p.second;
		}

#if defined(EIGEN_VECTORIZE_SSE2) || defined(EIGEN_VECTORIZE_AVX) || defined(EIGEN_VECTORIZE_NEON)
		for (auto& p : test_eigenrand_cont<Eigen::Rand::Vmt19937_64>(size, step, 42 * i))
		{
			err[p.first + "\t:ERand+Vrng"] += p.second;
			errSq[p.first + "\t:ERand+Vrng"] += p.second * p.second;
		}

		for (auto& p : test_eigenrand_disc<Eigen::Rand::Vmt19937_64>(size, step, 42 * i))
		{
			kl[p.first + "\t:ERand+Vrng"] += p.second;
			klSq[p.first + "\t:ERand+Vrng"] += p.second * p.second;
		}
#endif
#ifndef SKIP_REFERENCE
		for (auto& p : test_cpp11_cont(size, step, 42 * i))
		{
			err[p.first + "\t:C++11"] += p.second;
			errSq[p.first + "\t:C++11"] += p.second * p.second;
		}

		for (auto& p : test_cpp11_disc(size, step, 42 * i))
		{
			kl[p.first + "\t:C++11"] += p.second;
			klSq[p.first + "\t:C++11"] += p.second * p.second;
		}

		for (auto& p : test_old(size, step, 42 * i))
		{
			err[p.first + "\t:Old"] += p.second;
			errSq[p.first + "\t:Old"] += p.second * p.second;
		}
#endif
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
	std::cout << std::endl;
	std::cout << "[KL Divergence] Mean (Stdev)" << std::endl;
	for (auto& p : kl)
	{
		double mean = p.second / repeat;
		double var = (klSq[p.first] / repeat) - mean * mean;
		size_t sp = p.first.find('\t');
		std::cout << std::left << std::setw(28) << p.first.substr(0, sp);
		std::cout << std::setw(14) << p.first.substr(sp + 1);
		std::cout << ": " << mean << " (" << std::sqrt(var) << ")" << std::endl;
	}
	std::cout << std::endl;
	return 0;
}
