/**
 * @file Multinomial.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.5.1
 * @date 2024-09-08
 *
 * @copyright Copyright (c) 2020-2024
 *
 */

#ifndef EIGENRAND_MVDISTS_MULTINOMIAL_H
#define EIGENRAND_MVDISTS_MULTINOMIAL_H

namespace Eigen
{
	namespace Rand
	{
		/**
		 * @brief Generator of real vectors on a multinomial distribution
		 * 
		 * @tparam _Scalar 
		 * @tparam Dim number of dimensions, or `Eigen::Dynamic`
		 */
		template<typename _Scalar = int32_t, Index Dim = -1>
		class MultinomialGen : public MvVecGenBase<MultinomialGen<_Scalar, Dim>, _Scalar, Dim>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "`MultinomialGen` needs integral types.");
			_Scalar trials;
			Matrix<double, Dim, 1> probs;
			DiscreteGen<_Scalar, int32_t> discrete;
		public:
			/**
			 * @brief Construct a new multinomial generator
			 * 
			 * @tparam WeightTy 
			 * @param _trials the number of trials
			 * @param _weights the weights of each category, `(Dim, 1)` shape matrix or vector
			 */
			template<typename WeightTy>
			MultinomialGen(_Scalar _trials, const MatrixBase<WeightTy>& _weights)
				: trials{ _trials }, probs{ _weights.template cast<double>() }, discrete(probs.data(), probs.data() + probs.size())
			{
				eigen_assert(_weights.cols() == 1);
				for (Index i = 0; i < probs.size(); ++i)
				{
					eigen_assert(probs[i] >= 0);
				}
				probs /= probs.sum();
			}

			MultinomialGen(const MultinomialGen&) = default;
			MultinomialGen(MultinomialGen&&) = default;

			MultinomialGen& operator=(const MultinomialGen&) = default;
			MultinomialGen& operator=(MultinomialGen&&) = default;

			Index dims() const { return probs.rows(); }

			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng, Index samples)
			{
				const Index dim = probs.size();
				Matrix<_Scalar, Dim, -1> ret(dim, samples);
				if (trials < std::max(30 * (dim - 1), (Index)100))
				{
					for (Index s = 0; s < samples; ++s)
					{
						ret.col(s) = generate(urng);
					}
				}
				else
				{
					Array<_Scalar, 1, -1> r_trials{ samples }, t{ samples };
					Map<const Array<float, -1, 1>> fr_trials{ reinterpret_cast<float*>(r_trials.data()), samples };
					Map<Array<float, -1, 1>> ft{ reinterpret_cast<float*>(t.data()), samples };
					r_trials.setConstant(trials);
					ft = impl::binomial(urng, fr_trials, probs[0]);
					r_trials -= t;
					ret.row(0) = t.matrix();
					double rest_p = 1 - probs[0];
					for (Index i = 1; i < dim - 1; ++i)
					{
						ft = impl::binomial(urng, fr_trials, probs[i] / rest_p);
						r_trials -= t;
						ret.row(i) = t.matrix();
						rest_p -= probs[i];
					}

					ret.row(dim - 1) = r_trials;
				}
				return ret;
			}

			template<typename Urng>
			inline Matrix<_Scalar, Dim, 1> generate(Urng&& urng)
			{
				const Index dim = probs.size();
				Matrix<_Scalar, Dim, 1> ret(dim);
				//if (trials < std::max(30 * (dim - 1), (Index)100))
				{
					ret.setZero();
					auto d = discrete.template generate<Matrix<_Scalar, 16, 1>>(16, 1, urng);
					Matrix<_Scalar, 16, 1> buf;
					Index i;
					for (i = 0; i < (trials & ~15); i += 16)
					{
						buf = d;
						for (Index j = 0; j < 16; ++j)
						{
							ret[buf[j]] += 1;
						}
					}
					buf.middleRows(0, trials - i) = d.middleRows(0, trials - i);
					for (Index j = 0; j < trials - i; ++j)
					{
						ret[buf[j]] += 1;
					}
				}
				/*else
				{
					double rest_p = 1;
					_Scalar t = trials;
					for (Index i = 0; i < dim - 1; ++i)
					{
						ret[i] = binomial<Array<_Scalar, 1, 1>>(1, 1, urng, t, probs[i] / rest_p)(0);
						t -= ret[i];
						rest_p -= probs[i];
					}
					ret[dim - 1] = trials - ret.head(dim - 1).sum();
				}*/
				return ret;
			}
		};

		/**
		 * @brief helper function constructing Eigen::Rand::MultinomialGen
		 * 
		 * @tparam IntTy 
		 * @tparam WeightTy 
		 * @param trials the number of trials
		 * @param probs The weights of each category with shape `(Dim, 1)` of matrix or vector.
		 * The number of entries determines the dimensionality of the distribution
		 * @return an instance of MultinomialGen in the appropriate type
		 */
		template<typename IntTy, typename WeightTy>
		inline auto makeMultinomialGen(IntTy trials, const MatrixBase<WeightTy>& probs)
			-> MultinomialGen<IntTy, MatrixBase<WeightTy>::RowsAtCompileTime>
		{
			return MultinomialGen<IntTy, MatrixBase<WeightTy>::RowsAtCompileTime>{ trials, probs };
		}

		/**
		 * @brief Generator of reals on a Dirichlet distribution
		 * 
		 * @tparam _Scalar 
		 * @tparam Dim number of dimensions, or `Eigen::Dynamic`
		 */
		template<typename _Scalar, Index Dim = -1>
		class DirichletGen : public MvVecGenBase<DirichletGen<_Scalar, Dim>, _Scalar, Dim>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "`DirichletGen` needs floating point types.");

			Matrix<_Scalar, Dim, 1> alpha;
			std::vector<GammaGen<_Scalar>> gammas;
		public:
			/**
			 * @brief Construct a new Dirichlet generator
			 * 
			 * @tparam AlphaTy 
			 * @param _alpha the concentration parameters with shape `(Dim, 1)` matrix or vector
			 */
			template<typename AlphaTy>
			DirichletGen(const MatrixBase<AlphaTy>& _alpha)
				: alpha{ _alpha }
			{
				eigen_assert(_alpha.cols() == 1);
				for (Index i = 0; i < alpha.size(); ++i)
				{
					eigen_assert(alpha[i] > 0);
					gammas.emplace_back(alpha[i]);
				}
			}

			DirichletGen(const DirichletGen&) = default;
			DirichletGen(DirichletGen&&) = default;

			Index dims() const { return alpha.rows(); }

			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng, Index samples)
			{
				const Index dim = alpha.size();
				Matrix<_Scalar, Dim, -1> ret(dim, samples);
				Matrix<_Scalar, -1, 1> tmp(samples);
				for (Index i = 0; i < dim; ++i)
				{
					tmp = gammas[i].generateLike(tmp, urng);
					ret.row(i) = tmp.transpose();
				}
				ret.array().rowwise() /= ret.array().colwise().sum();
				return ret;
			}

			template<typename Urng>
			inline Matrix<_Scalar, Dim, 1> generate(Urng&& urng)
			{
				const Index dim = alpha.size();
				Matrix<_Scalar, Dim, 1> ret(dim);
				for (Index i = 0; i < dim; ++i)
				{
					ret[i] = gammas[i].template generate<Matrix<_Scalar, 1, 1>>(1, 1, urng)(0);
				}
				ret /= ret.sum();
				return ret;
			}
		};

		/**
		 * @brief helper function constructing Eigen::Rand::DirichletGen
		 * 
		 * @tparam AlphaTy 
		 * @param alpha The concentration parameters with shape `(Dim, 1)` of matrix or vector.
		 * The number of entries determines the dimensionality of the distribution.
		 * @return an instance of MultinomialGen in the appropriate type
		 */
		template<typename AlphaTy>
		inline auto makeDirichletGen(const MatrixBase<AlphaTy>& alpha)
			-> DirichletGen<typename MatrixBase<AlphaTy>::Scalar, MatrixBase<AlphaTy>::RowsAtCompileTime>
		{
			return DirichletGen<typename MatrixBase<AlphaTy>::Scalar, MatrixBase<AlphaTy>::RowsAtCompileTime>{ alpha };
		}
	}
}

#endif
