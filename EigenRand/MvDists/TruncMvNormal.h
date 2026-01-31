/**
 * @file TruncMvNormal.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.6.0
 * @date 2026-01-31
 *
 * @copyright Copyright (c) 2020-2026
 *
 */

#ifndef EIGENRAND_MVDISTS_TRUNC_MVNORMAL_H
#define EIGENRAND_MVDISTS_TRUNC_MVNORMAL_H

namespace Eigen
{
	namespace Rand
	{
		/**
		 * @brief Generator of real vectors on a truncated multivariate normal distribution
		 *
		 * @note This class is experimental and its interface may change in future versions.
		 *
		 * Uses component-wise Gibbs sampling with the precision matrix formulation.
		 * Samples from X ~ N(mu, Sigma) truncated to the hyper-rectangle [lower, upper].
		 *
		 * @tparam _Scalar Numeric type
		 * @tparam Dim number of dimensions, or `Eigen::Dynamic`
		 */
		template<typename _Scalar, Index Dim = -1>
		class TruncMvNormalGen : public MvVecGenBase<TruncMvNormalGen<_Scalar, Dim>, _Scalar, Dim>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "`TruncMvNormalGen` needs floating point types.");

			Matrix<_Scalar, Dim, 1> mean;
			Matrix<_Scalar, Dim, Dim> precision;     // H = Sigma^{-1}
			Matrix<_Scalar, Dim, 1> lower_bounds;
			Matrix<_Scalar, Dim, 1> upper_bounds;
			Matrix<_Scalar, Dim, 1> cond_sd;         // precomputed: sqrt(1/H[i,i])
			Matrix<_Scalar, Dim, 1> inv_diag;        // precomputed: 1/H[i,i]
			Matrix<_Scalar, Dim, 1> initial_state;   // clamp(mean, lower, upper)
			Index burn_in;
			StdUniformRealGen<_Scalar> ur;

		public:
			/**
			 * @brief Construct a new truncated multivariate normal generator from lower triangular Cholesky factor
			 *
			 * @tparam MeanTy
			 * @tparam LTTy
			 * @tparam SupportVecTy
			 * @param _mean mean vector of the distribution
			 * @param _lt lower triangular matrix of decomposed covariance
			 * @param _support a Support object specifying the truncation bounds
			 * @param _burn_in number of Gibbs sweeps (0 = auto: 5*dim)
			 */
			template<typename MeanTy, typename LTTy, typename SupportVecTy>
			TruncMvNormalGen(const MatrixBase<MeanTy>& _mean,
				const MatrixBase<LTTy>& _lt,
				const Support<SupportVecTy>& _support,
				Index _burn_in,
				detail::LowerTriangular)
				: mean{ _mean }, lower_bounds{ _support.lower }, upper_bounds{ _support.upper }
			{
				const Index dim = _mean.rows();
				eigen_assert(_mean.cols() == 1);
				eigen_assert(_lt.rows() == dim && _lt.cols() == dim);
				eigen_assert(_support.lower.rows() == dim && _support.lower.cols() == 1);
				eigen_assert(_support.upper.rows() == dim && _support.upper.cols() == 1);

				// Compute precision: H = (L L^T)^{-1} = L^{-T} L^{-1}
				auto id = Matrix<_Scalar, Dim, Dim>::Identity(dim, dim);
				auto inv_lt = _lt.template triangularView<Lower>().solve(id).eval();
				precision = inv_lt.transpose() * inv_lt;

				// Precompute per-dimension values
				cond_sd.resize(dim);
				inv_diag.resize(dim);
				initial_state.resize(dim);
				for (Index i = 0; i < dim; ++i)
				{
					eigen_assert(lower_bounds(i) < upper_bounds(i));
					inv_diag(i) = _Scalar(1) / precision(i, i);
					cond_sd(i) = std::sqrt(inv_diag(i));
					// Clamp mean to bounds for initial state
					initial_state(i) = std::max(lower_bounds(i), std::min(mean(i), upper_bounds(i)));
				}

				burn_in = (_burn_in == 0) ? 5 * dim : _burn_in;
			}

			/**
			 * @brief Construct a new truncated multivariate normal generator from covariance matrix
			 *
			 * @tparam MeanTy
			 * @tparam CovTy
			 * @tparam SupportVecTy
			 * @param _mean mean vector of the distribution
			 * @param _cov covariance matrix (should be positive semi-definite)
			 * @param _support a Support object specifying the truncation bounds
			 * @param _burn_in number of Gibbs sweeps (0 = auto: 5*dim)
			 */
			template<typename MeanTy, typename CovTy, typename SupportVecTy>
			TruncMvNormalGen(const MatrixBase<MeanTy>& _mean,
				const MatrixBase<CovTy>& _cov,
				const Support<SupportVecTy>& _support,
				Index _burn_in = 0,
				detail::FullMatrix = {})
				: TruncMvNormalGen{ _mean, detail::template get_lt<_Scalar, Dim>(_cov), _support, _burn_in, lower_triangular }
			{
			}

			TruncMvNormalGen(const TruncMvNormalGen&) = default;
			TruncMvNormalGen(TruncMvNormalGen&&) = default;

			TruncMvNormalGen& operator=(const TruncMvNormalGen&) = default;
			TruncMvNormalGen& operator=(TruncMvNormalGen&&) = default;

			Index dims() const { return mean.rows(); }

			/**
			 * @brief generates one sample via Gibbs sampling
			 *
			 * @tparam Urng
			 * @param urng c++11-style random number generator
			 * @return a random vector within the truncation bounds
			 */
			template<typename Urng>
			inline Matrix<_Scalar, Dim, 1> generate(Urng&& urng)
			{
				const Index dim = mean.rows();
				Matrix<_Scalar, Dim, 1> x = initial_state;
				Matrix<_Scalar, Dim, 1> diff = x - mean;

				for (Index sweep = 0; sweep < burn_in; ++sweep)
				{
					for (Index i = 0; i < dim; ++i)
					{
						// Conditional mean: mu_i - (1/H_ii) * sum_{j!=i} H_ij * (x_j - mu_j)
						_Scalar h_dot = precision.row(i).dot(diff);
						_Scalar cond_mean = mean(i) - inv_diag(i) * (h_dot - precision(i, i) * diff(i));

						// Sample from TruncNormal(cond_mean, cond_sd[i], lower[i], upper[i])
						_Scalar a_norm = (lower_bounds(i) - cond_mean) / cond_sd(i);
						_Scalar b_norm = (upper_bounds(i) - cond_mean) / cond_sd(i);
						_Scalar phi_a = (_Scalar)0.5 * std::erfc(-a_norm * (_Scalar)(1.0 / 1.4142135623730951));
						_Scalar phi_b = (_Scalar)0.5 * std::erfc(-b_norm * (_Scalar)(1.0 / 1.4142135623730951));

						_Scalar u = phi_a + (phi_b - phi_a) * ur(urng);
						_Scalar z = (_Scalar)(-1.4142135623730951) * detail::scalar_erfinv((_Scalar)1 - (_Scalar)2 * u);
						_Scalar new_xi = z * cond_sd(i) + cond_mean;

						diff(i) = new_xi - mean(i);
						x(i) = new_xi;
					}
				}
				return x;
			}

			/**
			 * @brief generates multiple samples at once via Gibbs sampling
			 *
			 * Each column is an independent Gibbs chain.
			 *
			 * @tparam Urng
			 * @param urng c++11-style random number generator
			 * @param samples the number of samples to be generated
			 * @return a random matrix with shape (dim, samples)
			 */
			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng, Index samples)
			{
				const Index dim = mean.rows();
				// Each column is an independent chain
				Matrix<_Scalar, Dim, -1> x = initial_state.replicate(1, samples);
				Matrix<_Scalar, Dim, -1> diff = x.colwise() - mean;

				for (Index sweep = 0; sweep < burn_in; ++sweep)
				{
					for (Index i = 0; i < dim; ++i)
					{
						// Batched: H.row(i) * diff for all chains at once
						auto h_dots = (precision.row(i) * diff).eval();

						for (Index k = 0; k < samples; ++k)
						{
							_Scalar cond_mean = mean(i) - inv_diag(i)
								* (h_dots(k) - precision(i, i) * diff(i, k));

							// Sample from TruncNormal(cond_mean, cond_sd[i], lower[i], upper[i])
							_Scalar a_norm = (lower_bounds(i) - cond_mean) / cond_sd(i);
							_Scalar b_norm = (upper_bounds(i) - cond_mean) / cond_sd(i);
							_Scalar phi_a = (_Scalar)0.5 * std::erfc(-a_norm * (_Scalar)(1.0 / 1.4142135623730951));
							_Scalar phi_b = (_Scalar)0.5 * std::erfc(-b_norm * (_Scalar)(1.0 / 1.4142135623730951));

							_Scalar u = phi_a + (phi_b - phi_a) * ur(urng);
							_Scalar z = (_Scalar)(-1.4142135623730951) * detail::scalar_erfinv((_Scalar)1 - (_Scalar)2 * u);
							_Scalar new_xi = z * cond_sd(i) + cond_mean;

							diff(i, k) = new_xi - mean(i);
							x(i, k) = new_xi;
						}
					}
				}
				return x;
			}
		};

		/**
		 * @brief helper function constructing Eigen::Rand::TruncMvNormalGen from covariance matrix
		 *
		 * @tparam MeanTy
		 * @tparam CovTy
		 * @tparam SupportVecTy
		 * @param mean mean vector of the distribution
		 * @param cov covariance matrix (should be positive semi-definite)
		 * @param support a Support object specifying the truncation bounds
		 * @param burn_in number of Gibbs sweeps (0 = auto: 5*dim)
		 */
		template<typename MeanTy, typename CovTy, typename SupportVecTy>
		inline auto makeTruncMvNormalGen(
			const MatrixBase<MeanTy>& mean,
			const MatrixBase<CovTy>& cov,
			const Support<SupportVecTy>& support,
			Index burn_in = 0)
			-> TruncMvNormalGen<typename MatrixBase<MeanTy>::Scalar, MatrixBase<MeanTy>::RowsAtCompileTime>
		{
			static_assert(
				std::is_same<typename MatrixBase<MeanTy>::Scalar, typename MatrixBase<CovTy>::Scalar>::value,
				"Derived::Scalar must be the same with `mean` and `cov`'s Scalar."
			);
			static_assert(
				detail::normal_check_dims<MeanTy, CovTy>(),
				"assert: mean.RowsAtCompileTime == cov.RowsAtCompileTime && cov.RowsAtCompileTime == cov.ColsAtCompileTime"
			);
			return { mean, cov, support, burn_in };
		}

		/**
		 * @brief helper function constructing Eigen::Rand::TruncMvNormalGen from Cholesky factor
		 *
		 * @tparam MeanTy
		 * @tparam LTTy
		 * @tparam SupportVecTy
		 * @param mean mean vector of the distribution
		 * @param lt lower triangular matrix of decomposed covariance
		 * @param support a Support object specifying the truncation bounds
		 * @param burn_in number of Gibbs sweeps (0 = auto: 5*dim)
		 */
		template<typename MeanTy, typename LTTy, typename SupportVecTy>
		inline auto makeTruncMvNormalGenFromLt(
			const MatrixBase<MeanTy>& mean,
			const MatrixBase<LTTy>& lt,
			const Support<SupportVecTy>& support,
			Index burn_in = 0)
			-> TruncMvNormalGen<typename MatrixBase<MeanTy>::Scalar, MatrixBase<MeanTy>::RowsAtCompileTime>
		{
			static_assert(
				std::is_same<typename MatrixBase<MeanTy>::Scalar, typename MatrixBase<LTTy>::Scalar>::value,
				"Derived::Scalar must be the same with `mean` and `lt`'s Scalar."
			);
			static_assert(
				detail::normal_check_dims<MeanTy, LTTy>(),
				"assert: mean.RowsAtCompileTime == lt.RowsAtCompileTime && lt.RowsAtCompileTime == lt.ColsAtCompileTime"
			);
			return { mean, lt, support, burn_in, lower_triangular };
		}

	}
}

#endif
