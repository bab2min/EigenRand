#ifndef EIGENRAND_CORE_H
#define EIGENRAND_CORE_H

#include <EigenRand/RandUtils.h>
#include <EigenRand/Dists/Basic.h>
#include <EigenRand/Dists/Discrete.h>
#include <EigenRand/Dists/NormalExp.h>

namespace Eigen
{
	namespace Rand
	{
		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>
			randBits(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_randbits_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>
			randBitsLike(Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_randbits_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>
			balanced(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_balanced_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>
			balancedLike(const Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_balanced_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>
			uniformReal(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>
			uniformRealLike(Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			normalDist(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			normalDistLike(Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}


		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>
			normalDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>
			normalDistLike(Derived& o, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			lognormalDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			lognormalDistLike(Derived& o, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>
			expDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), lambda)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>
			expDistLike(Derived& o, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), lambda)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>
			gammaDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), alpha, beta)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>
			gammaDistLike(Derived& o, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), alpha, beta)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>
			weibullDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>
			weibullDistLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>
			extremeValueDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>
			extremeValueDistLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>
			chiSquaredDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar n = 0)
		{
			return CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>
			chiSquaredDistLike(Derived& o, Urng&& urng, typename Derived::Scalar n = 0)
		{
			return CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
				);
		}

		template<typename Derived, typename Urng, typename RealIter>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>
			discreteDist(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), first, last)
				);
		}

		template<typename Derived, typename Urng, typename RealIter>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>
			discreteDistLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), first, last)
				);
		}

		template<typename Derived, typename Urng, typename Real>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>
			discreteDist(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), il.begin(), il.end())
				);
		}

		template<typename Derived, typename Urng, typename Real>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>
			discreteDistLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), il.begin(), il.end())
				);
		}

		template<typename Derived, typename Urng, typename RealIter>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>
			discreteDistDP(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>(
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), first, last)
				);
		}

		template<typename Derived, typename Urng, typename RealIter>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>
			discreteDistDPLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>(
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), first, last)
				);
		}

		template<typename Derived, typename Urng, typename Real>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>
			discreteDistDP(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>(
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), il.begin(), il.end())
				);
		}

		template<typename Derived, typename Urng, typename Real>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>
			discreteDistDPLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>(
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), il.begin(), il.end())
				);
		}


		template<typename Derived, typename Urng, typename RealIter>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>
			discreteDistI32(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>(
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), first, last)
				);
		}

		template<typename Derived, typename Urng, typename RealIter>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>
			discreteDistI32Like(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>(
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), first, last)
				);
		}

		template<typename Derived, typename Urng, typename Real>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>
			discreteDistI32(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>(
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), il.begin(), il.end())
				);
		}

		template<typename Derived, typename Urng, typename Real>
		inline const CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>
			discreteDistI32Like(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>(
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), il.begin(), il.end())
				);
		}
	}
}

#endif