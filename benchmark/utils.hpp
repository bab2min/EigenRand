#pragma once

#include <map>
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <chrono>

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


class MatrixBenchmarkHelper
{
	std::map<std::string, double>& timing;
	std::map<std::string, std::pair<Eigen::MatrixXd, Eigen::MatrixXd> >& mean_var;
public:

	template<bool _matrix, typename _Ty>
	class ScopeMeasure
	{
		MatrixBenchmarkHelper& bh;
		std::string name;
		Eigen::MatrixBase<_Ty>& results;
		std::chrono::high_resolution_clock::time_point start;
	public:
		ScopeMeasure(MatrixBenchmarkHelper& _bh,
			const std::string& _name,
			Eigen::MatrixBase<_Ty>& _results) :
			bh{ _bh }, name{ _name }, results{ _results },
			start{ std::chrono::high_resolution_clock::now() }
		{
		}

		~ScopeMeasure()
		{
			if (!name.empty())
			{
				bh.timing[name] = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start).count();

				if (_matrix)
				{
					Eigen::Index dim = results.rows(), cnt = results.cols() / dim;
					Eigen::MatrixXd sum(dim, dim), sqsum(dim, dim);
					sum.setZero();
					sqsum.setZero();
					for (Eigen::Index i = 0; i < cnt; ++i)
					{
						auto t = results.middleCols(dim * i, dim).template cast<double>().array();
						sum += t.matrix();
						sqsum += (t * t).matrix();
					}
					sum /= cnt;
					sqsum /= cnt;
					sqsum = sqsum.array() - (sum.array() * sum.array());
					bh.mean_var[name] = std::make_pair(std::move(sum), std::move(sqsum));
				}
				else
				{
					Eigen::MatrixXd zc = results.template cast<double>();
					Eigen::VectorXd mean = zc.rowwise().mean();
					zc.colwise() -= mean;
					Eigen::MatrixXd cov = zc * zc.transpose();
					cov /= results.cols();
					bh.mean_var[name] = std::make_pair(std::move(mean), std::move(cov));
				}
			}
		}
	};

	MatrixBenchmarkHelper(std::map<std::string, double>& _timing,
		std::map<std::string, std::pair<Eigen::MatrixXd, Eigen::MatrixXd>>& _mean_var)
		: timing{ _timing }, mean_var{ _mean_var }
	{
	}

	template<bool _matrix, typename _Ty>
	ScopeMeasure<_matrix, _Ty> measure(const std::string& name, Eigen::MatrixBase<_Ty>& results)
	{
		return ScopeMeasure<_matrix, _Ty>(*this, name, results);
	}
};


template<class Ty>
std::vector<Ty> rand_vector(size_t size)
{
	std::mt19937_64 rng;
	std::vector<Ty> ret(size);
	for (auto& v : ret)
	{
		v = std::generate_canonical<Ty, sizeof(Ty) * 8>(rng);
	}
	return ret;
}

template<class Ty>
std::string format_digit(Ty n, size_t l = 0)
{
	auto ret = std::to_string(n);
	while (ret.size() < l)
	{
		ret.insert(ret.begin(), '0');
	}
	return ret;
}
