/**
 * @file Core.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.2.0
 * @date 2020-06-22
 * 
 * @copyright Copyright (c) 2020
 * 
 */


#ifndef EIGENRAND_CORE_H
#define EIGENRAND_CORE_H

#include <EigenRand/RandUtils.h>
#include <EigenRand/Dists/Basic.h>
#include <EigenRand/Dists/Discrete.h>
#include <EigenRand/Dists/NormalExp.h>
#include <EigenRand/Dists/GammaPoisson.h>

namespace Eigen
{
	/**
	 * @brief namespace for EigenRand
	 * 
	 */
	namespace Rand
	{
		template<typename Derived, typename Urng>
		using RandBitsType = CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates integers with random bits
		 * 
		 * @tparam Derived
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const RandBitsType<Derived, Urng> 
			randBits(Index rows, Index cols, Urng&& urng) 
		{
			return { 
				rows, cols, internal::scalar_randbits_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng)) 
			};
		}

		/**
		 * @brief generates integers with random bits
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const RandBitsType<Derived, Urng> 
			randBitsLike(Derived& o, Urng&& urng)
		{
			return { 
				o.rows(), o.cols(), internal::scalar_randbits_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng)) 
			};
		}

		template<typename Derived, typename Urng>
		using UniformIntType = CwiseNullaryOp<internal::scalar_uniform_int_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates integers with a given range `[min, max]`
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param min, max the range of integers being generated
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const UniformIntType<Derived, Urng> 
			uniformInt(Index rows, Index cols, Urng&& urng, typename Derived::Scalar min, typename Derived::Scalar max) 
		{
			return {
				rows, cols, internal::scalar_uniform_int_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), min, max)
			};
		}

		/**
		 * @brief generates integers with a given range `[min, max]`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param min, max the range of integers being generated
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const UniformIntType<Derived, Urng> 
			uniformIntLike(Derived& o, Urng&& urng, typename Derived::Scalar min, typename Derived::Scalar max)
		{
			return {
				o.rows(), o.cols(), internal::scalar_uniform_int_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), min, max)
			};
		}

		template<typename Derived, typename Urng>
		using BalancedType = CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals in a range `[-1, 1]`
		 * 
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const BalancedType<Derived, Urng>
			balanced(Index rows, Index cols, Urng&& urng)
		{
			return {
				rows, cols, internal::scalar_balanced_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
			};
		}

		/**
		 * @brief generates reals in a range `[-1, 1]`
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const BalancedType<Derived, Urng> 
			balancedLike(const Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), internal::scalar_balanced_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
			};
		}

		template<typename Derived, typename Urng>
		using UniformRealType = CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals in a range `[0, 1)`
		 * 
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const UniformRealType<Derived, Urng> 
			uniformReal(Index rows, Index cols, Urng&& urng)
		{
			return {
				rows, cols, internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
			};
		}

		/**
		 * @brief generates reals in a range `[0, 1)`
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const UniformRealType<Derived, Urng> 
			uniformRealLike(Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
			};
		}
		
		template<typename Derived, typename Urng>
		using NormalType = CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on a standard normal distribution (`mean` = 0, `stdev`=1)
		 * 
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const NormalType<Derived, Urng>
			normal(Index rows, Index cols, Urng&& urng)
		{
			return {
				rows, cols, internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
			};
		}

		/**
		 * @brief generates reals on a standard normal distribution (`mean` = 0, `stdev`=1)
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const NormalType<Derived, Urng>
			normalLike(Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
			};
		}

		template<typename Derived, typename Urng>
		using Normal2Type = CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on a normal distribution with arbitrary `mean` and `stdev`.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const Normal2Type<Derived, Urng>
			normal(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return {
				rows, cols, internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
			};
		}

		/**
		 * @brief generates reals on a normal distribution with arbitrary `mean` and `stdev`.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const Normal2Type<Derived, Urng>
			normalLike(Derived& o, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
			};
		}

		template<typename Derived, typename Urng>
		using LognormalType = CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on a lognormal distribution with arbitrary `mean` and `stdev`.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const LognormalType<Derived, Urng>
			lognormal(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return {
				rows, cols, internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
			};
		}

		/**
		 * @brief generates reals on a lognormal distribution with arbitrary `mean` and `stdev`.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const LognormalType<Derived, Urng>
			lognormalLike(Derived& o, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
			};
		}

		template<typename Derived, typename Urng>
		using StudentTType = CwiseNullaryOp<internal::scalar_student_t_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the Student's t distribution with arbirtrary degress of freedom.
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param n degrees of freedom
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const StudentTType<Derived, Urng>
			studentT(Index rows, Index cols, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				rows, cols, internal::scalar_student_t_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
			};
		}

		/**
		 * @brief generates reals on the Student's t distribution with arbirtrary degress of freedom.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param n degrees of freedom
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const StudentTType<Derived, Urng>
			studentTLike(Derived& o, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_student_t_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
			};
		}

		template<typename Derived, typename Urng>
		using ExponentialType = CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on an exponential distribution with arbitrary scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param lambda a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const ExponentialType<Derived, Urng>
			exponential(Index rows, Index cols, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return {
				rows, cols, internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), lambda)
			};
		}

		/**
		 * @brief generates reals on an exponential distribution with arbitrary scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param lambda a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const ExponentialType<Derived, Urng>
			exponentialLike(Derived& o, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), lambda)
			};
		}

		template<typename Derived, typename Urng>
		using GammaType = CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on a gamma distribution with arbitrary shape and scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param alpha a shape parameter of the distribution
		 * @param beta a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const GammaType<Derived, Urng>
			gamma(Index rows, Index cols, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return {
				rows, cols, internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), alpha, beta)
			};
		}

		/**
		 * @brief generates reals on a gamma distribution with arbitrary shape and scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param alpha a shape parameter of the distribution
		 * @param beta a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const GammaType<Derived, Urng>
			gammaLike(Derived& o, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), alpha, beta)
			};
		}

		template<typename Derived, typename Urng>
		using WeibullType = CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on a Weibull distribution with arbitrary shape and scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a a shape parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const WeibullType<Derived, Urng>
			weibull(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		/**
		 * @brief generates reals on a Weibull distribution with arbitrary shape and scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a a shape parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const WeibullType<Derived, Urng>
			weibullLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		template<typename Derived, typename Urng>
		using ExtremeValueType = CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on an extreme value distribution 
		 * (a.k.a Gumbel Type I, log-Weibull, Fisher-Tippett Type I) with arbitrary shape and scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const ExtremeValueType<Derived, Urng>
			extremeValue(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		/**
		 * @brief generates reals on an extreme value distribution 
		 * (a.k.a Gumbel Type I, log-Weibull, Fisher-Tippett Type I) with arbitrary shape and scale parameter.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const ExtremeValueType<Derived, Urng>
			extremeValueLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		template<typename Derived, typename Urng>
		using ChiSquaredType = CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the Chi-squared distribution with arbitrary degrees of freedom.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param n the degrees of freedom of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const ChiSquaredType<Derived, Urng>
			chiSquared(Index rows, Index cols, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				rows, cols, internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
			};
		}

		/**
		 * @brief generates reals on the Chi-squared distribution with arbitrary degrees of freedom.
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param n the degrees of freedom of the distribution
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng>
		inline const ChiSquaredType<Derived, Urng>
			chiSquaredLike(Derived& o, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
			};
		}

		template<typename Derived, typename Urng>
		using CauchyType = CwiseNullaryOp<internal::scalar_cauchy_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the Cauchy distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const CauchyType<Derived, Urng>
			cauchy(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, internal::scalar_cauchy_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		/**
		 * @brief generates reals on the Cauchy distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const CauchyType<Derived, Urng>
			cauchyLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_cauchy_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		template<typename Derived, typename Urng>
		using FisherFType = CwiseNullaryOp<internal::scalar_fisher_f_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the Fisher's F distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param m degrees of freedom
		 * @param n degrees of freedom
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const FisherFType<Derived, Urng>
			fisherF(Index rows, Index cols, Urng&& urng, typename Derived::Scalar m = 1, typename Derived::Scalar n = 1)
		{
			return {
				rows, cols, internal::scalar_fisher_f_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), m, n)
			};
		}

		/**
		 * @brief generates reals on the Fisher's F distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param m degrees of freedom
		 * @param n degrees of freedom
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const FisherFType<Derived, Urng>
			fisherFLike(Derived& o, Urng&& urng, typename Derived::Scalar m = 1, typename Derived::Scalar n = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_fisher_f_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), m, n)
			};
		}

		template<typename Derived, typename Urng>
		using BetaType = CwiseNullaryOp<internal::scalar_beta_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the beta distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a,b shape parameter
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const BetaType<Derived, Urng>
			beta(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, internal::scalar_beta_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		/**
		 * @brief generates reals on the beta distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a,b shape parameter
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const BetaType<Derived, Urng>
			betaLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_beta_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
			};
		}

		template<typename Derived, typename Urng>
		using DiscreteFType = CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, float>, const Derived>;

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteFType<Derived, Urng>
			discreteF(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), first, last)
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteFType<Derived, Urng>
			discreteFLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), first, last)
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteFType<Derived, Urng>
			discreteF(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), il.begin(), il.end())
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteFType<Derived, Urng>
			discreteFLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), il.begin(), il.end())
			};
		}

		template<typename Derived, typename Urng>
		using DiscreteDType = CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>, const Derived>;

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteDType<Derived, Urng>
			discreteD(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), first, last)
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteDType<Derived, Urng>
			discreteDLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), first, last)
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteDType<Derived, Urng>
			discreteD(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), il.begin(), il.end())
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteDType<Derived, Urng>
			discreteDLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, double>(std::forward<Urng>(urng), il.begin(), il.end())
			};
		}

		template<typename Derived, typename Urng>
		using DiscreteType = CwiseNullaryOp<internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>, const Derived>;

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteType<Derived, Urng>
			discrete(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), first, last)
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteType<Derived, Urng>
			discreteLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), first, last)
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteType<Derived, Urng>
			discrete(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				rows, cols, internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), il.begin(), il.end())
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 * 
		 * @tparam Derived 
		 * @tparam Urng 
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o` 
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteType<Derived, Urng>
			discreteLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				o.rows(), o.cols(), internal::scalar_discrete_dist_op<typename Derived::Scalar, Urng, int32_t>(std::forward<Urng>(urng), il.begin(), il.end())
			};
		}

		template<typename Derived, typename Urng>
		using PoissonType = CwiseNullaryOp<internal::scalar_poisson_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the Poisson distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param mean rate parameter
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const PoissonType<Derived, Urng>
			poisson(Index rows, Index cols, Urng&& urng, double mean = 1)
		{
			return {
				rows, cols, internal::scalar_poisson_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean)
			};
		}

		/**
		 * @brief generates reals on the Poisson distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param mean rate parameter
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const PoissonType<Derived, Urng>
			poissonLike(Derived& o, Urng&& urng, double mean = 1)
		{
			return {
				o.rows(), o.cols(), internal::scalar_poisson_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean)
			};
		}

		template<typename Derived, typename Urng>
		using BinomialType = CwiseNullaryOp<internal::scalar_binomial_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the binomial distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param trials the number of trials
		 * @param p probability of a trial generating true
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const BinomialType<Derived, Urng>
			binomial(Index rows, Index cols, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				rows, cols, internal::scalar_binomial_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), trials, p)
			};
		}

		/**
		 * @brief generates reals on the binomial distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param trials the number of trials
		 * @param p probability of a trial generating true
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const BinomialType<Derived, Urng>
			binomialLike(Derived& o, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				o.rows(), o.cols(), internal::scalar_binomial_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), trials, p)
			};
		}

		template<typename Derived, typename Urng>
		using NegativeBinomialType = CwiseNullaryOp<internal::scalar_negative_binomial_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the negative binomial distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param trials the number of trial successes
		 * @param p probability of a trial generating true
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const NegativeBinomialType<Derived, Urng>
			negativeBinomial(Index rows, Index cols, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				rows, cols, internal::scalar_negative_binomial_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), trials, p)
			};
		}

		/**
		 * @brief generates reals on the negative binomial distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param trials the number of trial successes
		 * @param p probability of a trial generating true
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const NegativeBinomialType<Derived, Urng>
			negativeBinomialLike(Derived& o, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				o.rows(), o.cols(), internal::scalar_negative_binomial_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), trials, p)
			};
		}

		template<typename Derived, typename Urng>
		using GeometricType = CwiseNullaryOp<internal::scalar_geometric_dist_op<typename Derived::Scalar, Urng>, const Derived>;

		/**
		 * @brief generates reals on the geometric distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param p probability of a trial generating true
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const GeometricType<Derived, Urng>
			geometric(Index rows, Index cols, Urng&& urng, double p = 0.5)
		{
			return {
				rows, cols, internal::scalar_geometric_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), p)
			};
		}

		/**
		 * @brief generates reals on the geometric distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param p probability of a trial generating true
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const GeometricType<Derived, Urng>
			geometricLike(Derived& o, Urng&& urng, double p = 0.5)
		{
			return {
				o.rows(), o.cols(), internal::scalar_geometric_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), p)
			};
		}
	}
}

#endif