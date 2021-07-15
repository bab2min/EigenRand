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
	std::map<std::string, std::pair<Eigen::MatrixXd, Eigen::MatrixXd> >& mean_var;
public:

	template<bool _matrix, typename _Ty>
	class ScopeMeasure
	{
		BenchmarkHelper& bh;
		std::string name;
		Eigen::MatrixBase<_Ty>& results;
		std::chrono::high_resolution_clock::time_point start;
	public:
		ScopeMeasure(BenchmarkHelper& _bh, 
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

	BenchmarkHelper(std::map<std::string, double>& _timing,
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

template<typename Scalar, typename GenTy, typename Rng>
void test_single_and_batch(size_t size, BenchmarkHelper& bh, const std::string& suffix, GenTy&& gen, Rng& urng)
{
	Eigen::Matrix<Scalar, -1, -1> xf;
	xf.resize(gen.dims(), size);
	{
		auto scope = bh.template measure<false>(suffix, xf);
		for (size_t i = 0; i < size; ++i)
		{
			xf.col(i) = gen.generate(urng);
		}
	}

	{
		auto scope = bh.template measure<false>("batch/" + suffix, xf);
		xf = gen.generate(urng, size);
	}
}


template<typename Scalar, typename GenTy, typename Rng>
void test_single_and_batch_matrix(size_t size, BenchmarkHelper& bh, const std::string& suffix, GenTy&& gen, Rng& urng)
{
	Eigen::Matrix<Scalar, -1, -1> xf;
	xf.resize(gen.dims(), size * gen.dims());
	{
		auto scope = bh.template measure<true>(suffix, xf);
		for (size_t i = 0; i < size; ++i)
		{
			xf.middleCols(i * gen.dims(), gen.dims()) = gen.generate(urng);
		}
	}

	{
		auto scope = bh.template measure<true>("batch/" + suffix, xf);
		xf = gen.generate(urng, size);
	}
}

template<typename Rng>
std::map<std::string, double> test_eigenrand(size_t size, const std::string& suffix, std::map<std::string, std::pair<Eigen::MatrixXd, Eigen::MatrixXd> >& results)
{
	std::map<std::string, double> ret;
	BenchmarkHelper bh{ ret, results };
	Rng urng;

	{
		Eigen::Vector4f mean{ 0, 1, 2, 3 };
		Eigen::Matrix4f cov;
		cov << 1, 1, 0, 0,
			1, 2, 0, 0,
			0, 0, 3, 1,
			0, 0, 1, 2;
		test_single_and_batch<float>(size, bh, "MvNormal/4/float" + suffix, Eigen::Rand::makeMvNormalGen(mean, cov), urng);
	}

	{
		Eigen::VectorXf mean = Eigen::VectorXf::LinSpaced(100, 0, 99);
		Eigen::MatrixXf cov(100, 100);
		cov = Eigen::Rand::normalLike(cov, urng) * 0.1f;
		cov = cov * cov.transpose();
		cov += Eigen::MatrixXf::Identity(100, 100) * 4;
		test_single_and_batch<float>(size, bh, "MvNormal/100/float" + suffix, Eigen::Rand::makeMvNormalGen(mean, cov), urng);
	}

	{
		Eigen::Vector4f w{ 1, 2, 3, 4 };
		test_single_and_batch<int32_t>(size, bh, "Multinomial/4/float t=20" + suffix, Eigen::Rand::makeMultinomialGen<int32_t>(20, w), urng);
		test_single_and_batch<int32_t>(size, bh, "Multinomial/4/float t=1000" + suffix, Eigen::Rand::makeMultinomialGen<int32_t>(1000, w), urng);
	}

	{
		Eigen::VectorXf w = Eigen::VectorXf::LinSpaced(100, 0.1, 10.0);
		test_single_and_batch<int32_t>(size, bh, "Multinomial/100/float t=20" + suffix, Eigen::Rand::makeMultinomialGen<int32_t>(20, w), urng);
		test_single_and_batch<int32_t>(size, bh, "Multinomial/100/float t=1000" + suffix, Eigen::Rand::makeMultinomialGen<int32_t>(1000, w), urng);
	}

	{
		Eigen::Vector4f alpha{ 0.1, 0.2, 0.3, 0.4 };
		test_single_and_batch<float>(size, bh, "Dirichlet/4/float" + suffix, Eigen::Rand::makeDirichletGen(alpha), urng);
	}

	{
		Eigen::VectorXf alpha = Eigen::VectorXf::LinSpaced(100, 0.1, 10.0);
		test_single_and_batch<float>(size, bh, "Dirichlet/100/float" + suffix, Eigen::Rand::makeDirichletGen(alpha), urng);
	}

	{
		Eigen::Matrix4f scale;
		scale << 2, 1, 0, 0,
				1, 2, 1, 0,
				0, 1, 2, 1,
				0, 0, 1, 2;
		test_single_and_batch_matrix<float>(size, bh, "Wishart/4/float" + suffix, Eigen::Rand::makeWishartGen(4, scale), urng);
	}

	{
		Eigen::MatrixXf scale(50, 50);
		scale = Eigen::Rand::uniformRealLike(scale, urng) * 0.1f;
		scale = scale * scale.transpose();
		scale += Eigen::MatrixXf::Identity(50, 50);
		test_single_and_batch_matrix<float>(size, bh, "Wishart/50/float" + suffix, Eigen::Rand::makeWishartGen(50, scale), urng);
	}

	{
		Eigen::Matrix4f scale;
		scale << 2, 1, 0, 0,
			1, 2, 1, 0,
			0, 1, 2, 1,
			0, 0, 1, 2;
		test_single_and_batch_matrix<float>(size, bh, "InvWishart/4/float" + suffix, Eigen::Rand::makeInvWishartGen(8, scale), urng);
	}

	{
		Eigen::MatrixXf scale(50, 50);
		scale = Eigen::Rand::uniformRealLike(scale, urng) * 0.1f;
		scale = scale * scale.transpose();
		scale += Eigen::MatrixXf::Identity(50, 50);
		test_single_and_batch_matrix<float>(size, bh, "InvWishart/50/float" + suffix, Eigen::Rand::makeInvWishartGen(54, scale), urng);
	}
	return ret;
}

int main(int argc, char** argv)
{
	size_t size = 10000, repeat = 20;

	if (argc > 1) size = std::stoi(argv[1]);
	if (argc > 2) repeat = std::stoi(argv[2]);

	std::cout << "[Benchmark] Generating Random Matrix " << size << " samples "
		<< repeat << " times" << std::endl;

	std::map<std::string, double> time, timeSq;
	std::map<std::string, std::pair<Eigen::MatrixXd, Eigen::MatrixXd> > results;

	for (size_t i = 0; i < repeat; ++i)
	{
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

	std::cout << std::endl << "[Statistics]" << std::endl;
	Eigen::IOFormat sformat{ 4, 0, ", ", "\n     ", "[", "]" };
	for (auto& p : results)
	{
		if (p.second.first.rows() >= 10) continue;
		size_t sp = p.first.find('\t');
		std::cout << std::left << std::setw(28) << p.first.substr(0, sp);
		std::cout << std::setw(14) << p.first.substr(sp + 1) << std::endl;
		std::cout << "Mean " << p.second.first.transpose().format(sformat) << std::endl;
		std::cout << "Var  " << p.second.second.format(sformat) << std::endl << std::endl;
	}
	std::cout << std::endl;
	return 0;
}
