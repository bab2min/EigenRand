/**
 * @file MvNormal.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.3.0
 * @date 2020-10-07
 *
 * @copyright Copyright (c) 2020
 *
 */

#ifndef EIGENRAND_MVDISTS_MVNORMAL_H
#define EIGENRAND_MVDISTS_MVNORMAL_H

namespace Eigen
{
	namespace Rand
	{
		/**
		* 
		*/
		template<typename _Scalar, Index Dim = -1>
		class MvNormalGen
		{
			Matrix<_Scalar, Dim, 1> mean;
			Matrix<_Scalar, Dim, Dim> lt;
			StdNormalGen<_Scalar> stdnorm;

			struct LowerTriangular {};

			template<typename MeanTy, typename LTTy>
			MvNormalGen(LowerTriangular, const MatrixBase<MeanTy>& _mean, const MatrixBase<LTTy>& _lt)
				: mean{ _mean }, lt{ _lt }
			{
				eigen_assert(_mean.cols() == 1 && _mean.rows() == _lt.rows() && _lt.rows() == _lt.cols());
			}

		public:

			template<typename MeanTy, typename CovTy>
			MvNormalGen(const MatrixBase<MeanTy>& _mean, const MatrixBase<CovTy>& _cov)
				: mean{ _mean }
			{
				eigen_assert(_mean.cols() == 1 && _mean.rows() == _cov.rows() && _cov.rows() == _cov.cols());

				LLT<Matrix<_Scalar, Dim, Dim>> llt(_cov);
				if (llt.info() == Eigen::Success)
				{
					lt = llt.matrixL();
				}
				else
				{
					SelfAdjointEigenSolver<Matrix<_Scalar, Dim, Dim>> solver(_cov);
					if (solver.info() != Eigen::Success)
					{
						throw std::runtime_error{ "Matrix `_cov` cannot be solved!" };
					}
					lt = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
				}
			}

			template<typename MeanTy, typename LTTy>
			static MvNormalGen fromLt(const MatrixBase<MeanTy>& _mean, const MatrixBase<LTTy>& _lt)
			{
				return MvNormalGen{ LowerTriangular{}, _mean, _lt };
			}

			MvNormalGen(const MvNormalGen&) = default;
			MvNormalGen(MvNormalGen&&) = default;

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

		template<typename MeanTy, typename CovTy>
		inline auto makeMvNormGen(const MatrixBase<MeanTy>& mean, const MatrixBase<CovTy>& cov)
			-> MvNormalGen<typename MatrixBase<MeanTy>::Scalar, MatrixBase<MeanTy>::RowsAtCompileTime>
		{
			static_assert(
				std::is_same<typename MatrixBase<MeanTy>::Scalar, MatrixBase<CovTy>::Scalar>::value,
				"Derived::Scalar must be the same with `mean` and `cov`'s Scalar."
			);
			static_assert(
				MatrixBase<MeanTy>::RowsAtCompileTime == MatrixBase<CovTy>::RowsAtCompileTime &&
				MatrixBase<CovTy>::RowsAtCompileTime == MatrixBase<CovTy>::ColsAtCompileTime,
				"assert: mean.RowsAtCompileTime == cov.RowsAtCompileTime && cov.RowsAtCompileTime == cov.ColsAtCompileTime"
			);
			return { mean, cov };
		}

		template<typename MeanTy, typename LTTy>
		inline auto makeMvNormGenFromLt(const MatrixBase<MeanTy>& mean, const MatrixBase<LTTy>& lt)
			-> MvNormalGen<typename MatrixBase<MeanTy>::Scalar, MatrixBase<MeanTy>::RowsAtCompileTime>
		{
			static_assert(
				std::is_same<typename MatrixBase<MeanTy>::Scalar, MatrixBase<LTTy>::Scalar>::value,
				"Derived::Scalar must be the same with `mean` and `lt`'s Scalar."
				);
			static_assert(
				MatrixBase<MeanTy>::RowsAtCompileTime == MatrixBase<LTTy>::RowsAtCompileTime &&
				MatrixBase<LTTy>::RowsAtCompileTime == MatrixBase<LTTy>::ColsAtCompileTime,
				"assert: mean.RowsAtCompileTime == lt.RowsAtCompileTime && lt.RowsAtCompileTime == lt.ColsAtCompileTime"
				);
			return MvNormalGen<typename MatrixBase<MeanTy>::Scalar, MatrixBase<MeanTy>::RowsAtCompileTime>::fromLt(mean, lt);
		}


		template<typename _Scalar, Index Dim>
		class WishartGen
		{
			Index df;
			Matrix<_Scalar, Dim, Dim> chol;
			StdNormalGen<_Scalar> stdnorm;
			std::vector<ChiSquaredGen<_Scalar>> chisqs;
		public:

			template<typename ScaleTy>
			WishartGen(Index _df, const MatrixBase<ScaleTy>& _scale)
				: df{ _df }
			{
				eigen_assert(df > _scale.rows() - 1);
				eigen_assert(_scale.rows() == _scale.cols());

				LLT<Matrix<_Scalar, Dim, Dim>> llt(_scale);
				if (llt.info() != Eigen::Success) throw std::runtime_error{ "Matrix `_scale` cannot be solved!" };
				chol = llt.matrixL();
				
				for (Index i = 0; i < chol.rows(); ++i)
				{
					chisqs.emplace_back(df - i);
				}
			}

			WishartGen(const WishartGen&) = default;
			WishartGen(WishartGen&&) = default;

			Index dims() const { return chol.rows(); }

			template<typename Urng>
			inline auto generate(Urng&& urng, Index samples)
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
			inline auto generate(Urng&& urng)
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

		template<typename _Scalar, Index Dim>
		class InvWishartGen
		{
			Index df;
			Matrix<_Scalar, Dim, Dim> chol;
			StdNormalGen<_Scalar> stdnorm;
			std::vector<ChiSquaredGen<_Scalar>> chisqs;
		public:

			template<typename ScaleTy>
			InvWishartGen(Index _df, const MatrixBase<ScaleTy>& _scale)
				: df{ _df }
			{
				eigen_assert(df > _scale.rows() - 1);
				eigen_assert(_scale.rows() == _scale.cols());
				using Mat = Eigen::Matrix<_Scalar, Dim, Dim>;
				LLT<Mat> llt(_scale.inverse());
				if (llt.info() != Eigen::Success) throw std::runtime_error{ "Matrix `_scale` cannot be solved!" };
				chol = llt.matrixL();

				for (Index i = 0; i < chol.rows(); ++i)
				{
					chisqs.emplace_back(df - i);
				}
			}

			InvWishartGen(const InvWishartGen&) = default;
			InvWishartGen(InvWishartGen&&) = default;

			Index dims() const { return chol.rows(); }

			template<typename Urng>
			inline auto generate(Urng&& urng, Index samples)
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
			inline auto generate(Urng&& urng)
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
	}
}

#endif