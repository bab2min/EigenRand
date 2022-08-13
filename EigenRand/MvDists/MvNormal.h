/**
 * @file MvNormal.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.4.1
 * @date 2022-08-13
 *
 * @copyright Copyright (c) 2020-2021
 *
 */

#ifndef EIGENRAND_MVDISTS_MVNORMAL_H
#define EIGENRAND_MVDISTS_MVNORMAL_H

namespace Eigen
{
	namespace Rand
	{
		namespace detail
		{
			template<typename _Scalar, Index Dim, typename _Mat>
			Matrix<_Scalar, Dim, Dim> get_lt(const MatrixBase<_Mat>& mat)
			{
				LLT<Matrix<_Scalar, Dim, Dim>> llt(mat);
				if (llt.info() == Eigen::Success)
				{
					return llt.matrixL();
				}
				else
				{
					SelfAdjointEigenSolver<Matrix<_Scalar, Dim, Dim>> solver(mat);
					if (solver.info() != Eigen::Success)
					{
						throw std::runtime_error{ "The matrix cannot be solved!" };
					}
					return solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
				}
			}

			class FullMatrix {};
			class LowerTriangular {};
			class InvLowerTriangular {};
		}

		constexpr detail::FullMatrix full_matrix;
		constexpr detail::LowerTriangular lower_triangular;
		constexpr detail::InvLowerTriangular inv_lower_triangular;
		

		/**
		 * @brief Generator of real vectors on a multivariate normal distribution
		 * 
		 * @tparam _Scalar Numeric type
		 * @tparam Dim number of dimensions, or `Eigen::Dynamic`
		 */
		template<typename _Scalar, Index Dim = -1>
		class MvNormalGen : public MvVecGenBase<MvNormalGen<_Scalar, Dim>, _Scalar, Dim>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "`MvNormalGen` needs floating point types.");

			Matrix<_Scalar, Dim, 1> mean;
			Matrix<_Scalar, Dim, Dim> lt;
			StdNormalGen<_Scalar> stdnorm;

		public:
			/**
			 * @brief Construct a new multivariate normal generator from lower triangular matrix of decomposed covariance
			 * 
			 * @tparam MeanTy 
			 * @tparam LTTy 
			 * @param _mean mean vector of the distribution
			 * @param _lt lower triangular matrix of decomposed covariance
			 */
			template<typename MeanTy, typename LTTy>
			MvNormalGen(const MatrixBase<MeanTy>& _mean, const MatrixBase<LTTy>& _lt, detail::LowerTriangular)
				: mean{ _mean }, lt{ _lt }
			{
				eigen_assert(_mean.cols() == 1 && _mean.rows() == _lt.rows() && _lt.rows() == _lt.cols());
			}

			/**
			 * @brief Construct a new multivariate normal generator from covariance matrix
			 * 
			 * @tparam MeanTy 
			 * @tparam CovTy 
			 * @param _mean mean vector of the distribution
			 * @param _cov covariance matrix (should be positive semi-definite)
			 */
			template<typename MeanTy, typename CovTy>
			MvNormalGen(const MatrixBase<MeanTy>& _mean, const MatrixBase<CovTy>& _cov, detail::FullMatrix = {})
				: MvNormalGen{ _mean, detail::template get_lt<_Scalar, Dim>(_cov), lower_triangular }
			{
			}

			MvNormalGen(const MvNormalGen&) = default;
			MvNormalGen(MvNormalGen&&) = default;

			MvNormalGen& operator=(const MvNormalGen&) = default;
			MvNormalGen& operator=(MvNormalGen&&) = default;

			Index dims() const { return mean.rows(); }

			template<typename Urng>
			inline auto generate(Urng&& urng, Index samples)
				-> decltype((lt * stdnorm.template generate<Matrix<_Scalar, Dim, -1>>(mean.rows(), samples, std::forward<Urng>(urng))).colwise() + mean)
			{
				return (lt * stdnorm.template generate<Matrix<_Scalar, Dim, -1>>(mean.rows(), samples, std::forward<Urng>(urng))).colwise() + mean;
			}

			template<typename Urng>
			inline auto generate(Urng&& urng)
				-> decltype((lt * stdnorm.template generate<Matrix<_Scalar, Dim, 1>>(mean.rows(), 1, std::forward<Urng>(urng))).colwise() + mean)
			{
				return (lt * stdnorm.template generate<Matrix<_Scalar, Dim, 1>>(mean.rows(), 1, std::forward<Urng>(urng))).colwise() + mean;
			}
		};

		/**
		 * @brief helper function constructing Eigen::Rand::MvNormal
		 * 
		 * @tparam MeanTy 
		 * @tparam CovTy 
		 * @param mean mean vector of the distribution
		 * @param cov covariance matrix (should be positive semi-definite)
		 */
		template<typename MeanTy, typename CovTy>
		inline auto makeMvNormalGen(const MatrixBase<MeanTy>& mean, const MatrixBase<CovTy>& cov)
			-> MvNormalGen<typename MatrixBase<MeanTy>::Scalar, MatrixBase<MeanTy>::RowsAtCompileTime>
		{
			static_assert(
				std::is_same<typename MatrixBase<MeanTy>::Scalar, typename MatrixBase<CovTy>::Scalar>::value,
				"Derived::Scalar must be the same with `mean` and `cov`'s Scalar."
			);
			static_assert(
				MatrixBase<MeanTy>::RowsAtCompileTime == MatrixBase<CovTy>::RowsAtCompileTime &&
				MatrixBase<CovTy>::RowsAtCompileTime == MatrixBase<CovTy>::ColsAtCompileTime,
				"assert: mean.RowsAtCompileTime == cov.RowsAtCompileTime && cov.RowsAtCompileTime == cov.ColsAtCompileTime"
			);
			return { mean, cov };
		}

		/**
		 * @brief helper function constructing Eigen::Rand::MvNormal
		 * 
		 * @tparam MeanTy 
		 * @tparam LTTy 
		 * @param mean mean vector of the distribution
		 * @param lt lower triangular matrix of decomposed covariance
		 */
		template<typename MeanTy, typename LTTy>
		inline auto makeMvNormalGenFromLt(const MatrixBase<MeanTy>& mean, const MatrixBase<LTTy>& lt)
			-> MvNormalGen<typename MatrixBase<MeanTy>::Scalar, MatrixBase<MeanTy>::RowsAtCompileTime>
		{
			static_assert(
				std::is_same<typename MatrixBase<MeanTy>::Scalar, typename MatrixBase<LTTy>::Scalar>::value,
				"Derived::Scalar must be the same with `mean` and `lt`'s Scalar."
			);
			static_assert(
				MatrixBase<MeanTy>::RowsAtCompileTime == MatrixBase<LTTy>::RowsAtCompileTime &&
				MatrixBase<LTTy>::RowsAtCompileTime == MatrixBase<LTTy>::ColsAtCompileTime,
				"assert: mean.RowsAtCompileTime == lt.RowsAtCompileTime && lt.RowsAtCompileTime == lt.ColsAtCompileTime"
			);
			return { mean, lt, lower_triangular };
		}

		/**
		 * @brief Generator of real matrices on a Wishart distribution
		 * 
		 * @tparam _Scalar 
		 * @tparam Dim number of dimensions, or `Eigen::Dynamic`
		 */
		template<typename _Scalar, Index Dim>
		class WishartGen : public MvMatGenBase<WishartGen<_Scalar, Dim>, _Scalar, Dim>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "`WishartGen` needs floating point types.");

			Index df;
			Matrix<_Scalar, Dim, Dim> chol;
			StdNormalGen<_Scalar> stdnorm;
			std::vector<ChiSquaredGen<_Scalar>> chisqs;
		public:
			/**
			 * @brief Construct a new Wishart generator from lower triangular matrix of decomposed scale
			 * 
			 * @tparam ScaleTy 
			 * @param _df degrees of freedom
			 * @param _lt lower triangular matrix of decomposed scale
			 */
			template<typename ScaleTy>
			WishartGen(Index _df, const MatrixBase<ScaleTy>& _lt, detail::LowerTriangular)
				: df{ _df }, chol{ _lt }
			{
				eigen_assert(df > chol.rows() - 1);
				eigen_assert(chol.rows() == chol.cols());

				for (Index i = 0; i < chol.rows(); ++i)
				{
					chisqs.emplace_back(df - i);
				}
			}

			/**
			 * @brief Construct a new Wishart generator from scale matrix
			 * 
			 * @tparam ScaleTy 
			 * @param _df degrees of freedom
			 * @param _lt scale matrix (should be positive definitive)
			 */
			template<typename ScaleTy>
			WishartGen(Index _df, const MatrixBase<ScaleTy>& _scale, detail::FullMatrix = {})
				: WishartGen{ _df, detail::template get_lt<_Scalar, Dim>(_scale), lower_triangular }
			{
				eigen_assert(_scale.rows() == _scale.cols());
			}

			WishartGen(const WishartGen&) = default;
			WishartGen(WishartGen&&) = default;

			WishartGen& operator=(const WishartGen&) = default;
			WishartGen& operator=(WishartGen&&) = default;

			Index dims() const { return chol.rows(); }

			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng, Index samples)
			{
				const Index dim = chol.rows();
				const Index normSamples = samples * dim * (dim - 1) / 2;
				using ArrayXs = Array<_Scalar, -1, 1>;
				Matrix<_Scalar, Dim, -1> rand_mat(dim, dim * samples), tmp(dim, dim * samples);
				
				_Scalar* ptr = tmp.data();
				Map<ArrayXs>{ ptr, normSamples } = stdnorm.template generate<ArrayXs>(normSamples, 1, urng);
				for (Index j = 0; j < samples; ++j)
				{
					for (Index i = 0; i < dim - 1; ++i)
					{
						rand_mat.col(i + j * dim).tail(dim - 1 - i) = Map<ArrayXs>{ ptr, dim - 1 - i };
						ptr += dim - 1 - i;
					}
				}
				
				for (Index i = 0; i < dim; ++i)
				{
					_Scalar* ptr = tmp.data();
					Map<ArrayXs>{ ptr, samples } = chisqs[i].template generate<ArrayXs>(samples, 1, urng).sqrt();
					for (Index j = 0; j < samples; ++j)
					{
						rand_mat(i, i + j * dim) = *ptr++;
					}
				}

				for (Index j = 0; j < samples; ++j)
				{
					rand_mat.middleCols(j * dim, dim).template triangularView<StrictlyUpper>().setZero();
				}
				tmp.noalias() = chol * rand_mat;
				
				for (Index j = 0; j < samples; ++j)
				{
					auto t = tmp.middleCols(j * dim, dim);
					rand_mat.middleCols(j * dim, dim).noalias() = t * t.transpose();
				}
				return rand_mat;
			}

			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng)
			{
				const Index dim = chol.rows();
				const Index normSamples = dim * (dim - 1) / 2;
				using ArrayXs = Array<_Scalar, -1, 1>;
				Matrix<_Scalar, Dim, Dim> rand_mat(dim, dim);
				Map<ArrayXs>{ rand_mat.data(), normSamples } = stdnorm.template generate<ArrayXs>(normSamples, 1, urng);

				for (Index i = 0; i < dim / 2; ++i)
				{
					rand_mat.col(dim - 2 - i).tail(i + 1) = rand_mat.col(i).head(i + 1);
				}
				
				for (Index i = 0; i < dim; ++i)
				{
					rand_mat(i, i) = chisqs[i].template generate<Array<_Scalar, 1, 1>>(1, 1, urng).sqrt()(0);
				}
				rand_mat.template triangularView<StrictlyUpper>().setZero();

				auto t = (chol * rand_mat).eval();
				return (t * t.transpose()).eval();
			}
		};

		/**
		 * @brief helper function constructing Eigen::Rand::WishartGen
		 * 
		 * @tparam ScaleTy 
		 * @param df degrees of freedom
		 * @param scale scale matrix (should be positive definitive)
		 */
		template<typename ScaleTy>
		inline auto makeWishartGen(Index df, const MatrixBase<ScaleTy>& scale)
			-> WishartGen<typename MatrixBase<ScaleTy>::Scalar, MatrixBase<ScaleTy>::RowsAtCompileTime>
		{
			static_assert(
				MatrixBase<ScaleTy>::RowsAtCompileTime == MatrixBase<ScaleTy>::ColsAtCompileTime,
				"assert: scale.RowsAtCompileTime == scale.ColsAtCompileTime"
			);
			return { df, scale };
		}

		/**
		 * @brief helper function constructing Eigen::Rand::WishartGen
		 * 
		 * @tparam LTTy 
		 * @param df degrees of freedom
		 * @param lt lower triangular matrix of decomposed scale
		 */
		template<typename LTTy>
		inline auto makeWishartGenFromLt(Index df, const MatrixBase<LTTy>& lt)
			-> WishartGen<typename MatrixBase<LTTy>::Scalar, MatrixBase<LTTy>::RowsAtCompileTime>
		{
			static_assert(
				MatrixBase<LTTy>::RowsAtCompileTime == MatrixBase<LTTy>::ColsAtCompileTime,
				"assert: lt.RowsAtCompileTime == lt.ColsAtCompileTime"
				);
			return { df, lt, lower_triangular };
		}

		/**
		 * @brief Generator of real matrices on a inverse Wishart distribution
		 * 
		 * @tparam _Scalar 
		 * @tparam Dim number of dimensions, or `Eigen::Dynamic`
		 */
		template<typename _Scalar, Index Dim>
		class InvWishartGen : public MvMatGenBase<InvWishartGen<_Scalar, Dim>, _Scalar, Dim>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "`InvWishartGen` needs floating point types.");

			Index df;
			Matrix<_Scalar, Dim, Dim> chol;
			StdNormalGen<_Scalar> stdnorm;
			std::vector<ChiSquaredGen<_Scalar>> chisqs;
		public:
			/**
			 * @brief Construct a new inverse Wishart generator
			 * 
			 * @tparam ScaleTy 
			 * @param _df degrees of freedom
			 * @param _ilt lower triangular matrix of decomposed inverse scale
			 */
			template<typename ScaleTy>
			InvWishartGen(Index _df, const MatrixBase<ScaleTy>& _ilt, detail::InvLowerTriangular)
				: df{ _df }, chol{ _ilt }
			{
				eigen_assert(df > chol.rows() - 1);
				eigen_assert(chol.rows() == chol.cols());

				for (Index i = 0; i < chol.rows(); ++i)
				{
					chisqs.emplace_back(df - i);
				}
			}

			/**
			 * @brief Construct a new inverse Wishart generator
			 * 
			 * @tparam ScaleTy 
			 * @param _df degrees of freedom
			 * @param _scale scale matrix (should be positive definitive)
			 */
			template<typename ScaleTy>
			InvWishartGen(Index _df, const MatrixBase<ScaleTy>& _scale, detail::FullMatrix = {})
				: InvWishartGen{ _df, detail::template get_lt<_Scalar, Dim>(_scale.inverse()), inv_lower_triangular }
			{
				eigen_assert(_scale.rows() == _scale.cols());	
			}

			InvWishartGen(const InvWishartGen&) = default;
			InvWishartGen(InvWishartGen&&) = default;

			InvWishartGen& operator=(const InvWishartGen&) = default;
			InvWishartGen& operator=(InvWishartGen&&) = default;

			Index dims() const { return chol.rows(); }

			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng, Index samples)
			{
				const Index dim = chol.rows();
				const Index normSamples = samples * dim * (dim - 1) / 2;
				using ArrayXs = Array<_Scalar, -1, 1>;
				Matrix<_Scalar, Dim, -1> rand_mat(dim, dim * samples), tmp(dim, dim * samples);

				_Scalar* ptr = tmp.data();
				Map<ArrayXs>{ ptr, normSamples } = stdnorm.template generate<ArrayXs>(normSamples, 1, urng);
				for (Index j = 0; j < samples; ++j)
				{
					for (Index i = 0; i < dim - 1; ++i)
					{
						rand_mat.col(i + j * dim).tail(dim - 1 - i) = Map<ArrayXs>{ ptr, dim - 1 - i };
						ptr += dim - 1 - i;
					}
				}

				for (Index i = 0; i < dim; ++i)
				{
					_Scalar* ptr = tmp.data();
					Map<ArrayXs>{ ptr, samples } = chisqs[i].template generate<ArrayXs>(samples, 1, urng).sqrt();
					for (Index j = 0; j < samples; ++j)
					{
						rand_mat(i, i + j * dim) = *ptr++;
					}
				}

				for (Index j = 0; j < samples; ++j)
				{
					rand_mat.middleCols(j * dim, dim).template triangularView<StrictlyUpper>().setZero();
				}
				tmp.noalias() = chol * rand_mat;

				auto id = Eigen::Matrix<_Scalar, Dim, Dim>::Identity(dim, dim);
				for (Index j = 0; j < samples; ++j)
				{
					auto t = tmp.middleCols(j * dim, dim);
					auto u = rand_mat.middleCols(j * dim, dim);
					u.noalias() = t.template triangularView<Lower>().solve(id);
					t.noalias() = u.transpose() * u;
				}
				return tmp;
			}

			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng)
			{
				const Index dim = chol.rows();
				const Index normSamples = dim * (dim - 1) / 2;
				using ArrayXs = Array<_Scalar, -1, 1>;
				Matrix<_Scalar, Dim, Dim> rand_mat(dim, dim);
				Map<ArrayXs>{ rand_mat.data(), normSamples } = stdnorm.template generate<ArrayXs>(normSamples, 1, urng);

				for (Index i = 0; i < dim / 2; ++i)
				{
					rand_mat.col(dim - 2 - i).tail(i + 1) = rand_mat.col(i).head(i + 1);
				}

				for (Index i = 0; i < dim; ++i)
				{
					rand_mat(i, i) = chisqs[i].template generate<Array<_Scalar, 1, 1>>(1, 1, urng).sqrt()(0);
				}
				rand_mat.template triangularView<StrictlyUpper>().setZero();

				auto t = (chol * rand_mat).eval();
				auto id = Eigen::Matrix<_Scalar, Dim, Dim>::Identity(dim, dim);
				rand_mat.noalias() = t.template triangularView<Lower>().solve(id);

				return (rand_mat.transpose() * rand_mat).eval();
			}
		};

		/**
		 * @brief helper function constructing Eigen::Rand::InvWishartGen
		 * 
		 * @tparam ScaleTy 
		 * @param df degrees of freedom
		 * @param scale scale matrix
		 */
		template<typename ScaleTy>
		inline auto makeInvWishartGen(Index df, const MatrixBase<ScaleTy>& scale)
			-> InvWishartGen<typename MatrixBase<ScaleTy>::Scalar, MatrixBase<ScaleTy>::RowsAtCompileTime>
		{
			static_assert(
				MatrixBase<ScaleTy>::RowsAtCompileTime == MatrixBase<ScaleTy>::ColsAtCompileTime,
				"assert: scale.RowsAtCompileTime == scale.ColsAtCompileTime"
			);
			return { df, scale };
		}

		/**
		 * @brief helper function constructing Eigen::Rand::InvWishartGen
		 * 
		 * @tparam ILTTy 
		 * @param df degrees of freedom
		 * @param ilt lower triangular matrix of decomposed inverse scale
		 */
		template<typename ILTTy>
		inline auto makeInvWishartGenFromIlt(Index df, const MatrixBase<ILTTy>& ilt)
			-> InvWishartGen<typename MatrixBase<ILTTy>::Scalar, MatrixBase<ILTTy>::RowsAtCompileTime>
		{
			static_assert(
				MatrixBase<ILTTy>::RowsAtCompileTime == MatrixBase<ILTTy>::ColsAtCompileTime,
				"assert: ilt.RowsAtCompileTime == ilt.ColsAtCompileTime"
			);
			return { df, ilt, inv_lower_triangular };
		}
	}
}

#endif