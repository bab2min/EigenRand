/**
 * @file Truncated.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.6.0
 * @date 2026-01-31
 *
 * @copyright Copyright (c) 2020-2026
 *
 */

#ifndef EIGENRAND_DISTS_TRUNCATED_H
#define EIGENRAND_DISTS_TRUNCATED_H

#include <cmath>

namespace Eigen
{
	namespace Rand
	{
		namespace detail
		{
			/**
			 * @brief Compute the inverse error function (float) using Mike Giles' approximation.
			 */
			template<typename _Scalar>
			EIGEN_STRONG_INLINE typename std::enable_if<
				std::is_same<_Scalar, float>::value, _Scalar
			>::type scalar_erfinv(_Scalar x)
			{
				_Scalar w = -std::log((1.0f - x) * (1.0f + x));
				_Scalar p;

				if (w < 5.0f)
				{
					w -= 2.5f;
					p = 2.81022636e-08f;
					p = 3.43273939e-07f + p * w;
					p = -3.5233877e-06f + p * w;
					p = -4.39150654e-06f + p * w;
					p = 0.00021858087f + p * w;
					p = -0.00125372503f + p * w;
					p = -0.00417768164f + p * w;
					p = 0.246640727f + p * w;
					p = 1.50140941f + p * w;
				}
				else
				{
					w = std::sqrt(w) - 3.0f;
					p = -0.000200214257f;
					p = 0.000100950558f + p * w;
					p = 0.00134934322f + p * w;
					p = -0.00367342844f + p * w;
					p = 0.00573950773f + p * w;
					p = -0.0076224613f + p * w;
					p = 0.00943887047f + p * w;
					p = 1.00167406f + p * w;
					p = 2.83297682f + p * w;
				}
				return p * x;
			}

			/**
			 * @brief Compute the inverse error function (double) using Mike Giles' approximation.
			 *
			 * Three regions for full double precision accuracy.
			 */
			template<typename _Scalar>
			EIGEN_STRONG_INLINE typename std::enable_if<
				std::is_same<_Scalar, double>::value, _Scalar
			>::type scalar_erfinv(_Scalar x)
			{
				_Scalar w = -std::log((1.0 - x) * (1.0 + x));
				_Scalar p;

				if (w < 6.25)
				{
					w -= 3.125;
					p = -3.6444120640178196996e-21;
					p = -1.685059138182016589e-19 + p * w;
					p = 1.2858480715256400167e-18 + p * w;
					p = 1.115787767802518096e-17 + p * w;
					p = -1.333171662854620906e-16 + p * w;
					p = 2.0972767875968561637e-17 + p * w;
					p = 6.6376381343583238325e-15 + p * w;
					p = -4.0545662729752068639e-14 + p * w;
					p = -8.1519341976054721522e-14 + p * w;
					p = 2.6335093153082322977e-12 + p * w;
					p = -1.2975133253453532498e-11 + p * w;
					p = -5.4154120542946279317e-11 + p * w;
					p = 1.051212273321532285e-09 + p * w;
					p = -4.1126339803469836976e-09 + p * w;
					p = -2.9070369957882005086e-08 + p * w;
					p = 4.2347877827932403518e-07 + p * w;
					p = -1.3654692000834678645e-06 + p * w;
					p = -1.3882523362786468719e-05 + p * w;
					p = 0.0001867342080340571352 + p * w;
					p = -0.00074070253416626697512 + p * w;
					p = -0.0060336708714301490533 + p * w;
					p = 0.24015818242558961693 + p * w;
					p = 1.6536545626831027356 + p * w;
				}
				else if (w < 16.0)
				{
					w = std::sqrt(w) - 3.25;
					p = 2.2137376921775787049e-09;
					p = 9.0756561938885390979e-08 + p * w;
					p = -2.7517406297064545428e-07 + p * w;
					p = 1.8239629214389227755e-08 + p * w;
					p = 1.5027403968909827627e-06 + p * w;
					p = -4.013867526981545969e-06 + p * w;
					p = 2.9234449089955446044e-06 + p * w;
					p = 1.2475304481671778723e-05 + p * w;
					p = -4.7318229009055733981e-05 + p * w;
					p = 6.8284851459573175448e-05 + p * w;
					p = 2.4031110387097893999e-05 + p * w;
					p = -0.0003550375203628474796 + p * w;
					p = 0.00095328937973738049703 + p * w;
					p = -0.0016882755560235047313 + p * w;
					p = 0.0024914420961078508066 + p * w;
					p = -0.0037512085075692412107 + p * w;
					p = 0.005370914553590063617 + p * w;
					p = 1.0052589676941592334 + p * w;
					p = 3.0838856104922207635 + p * w;
				}
				else
				{
					w = std::sqrt(w) - 5.0;
					p = -2.7109920616438573243e-11;
					p = -2.5556418169965252055e-10 + p * w;
					p = 1.5076572693500548083e-09 + p * w;
					p = -3.7894654401267369937e-09 + p * w;
					p = 7.6157012080783393804e-09 + p * w;
					p = -1.4960026627149240478e-08 + p * w;
					p = 2.9147953450901080826e-08 + p * w;
					p = -6.7711997758452339498e-08 + p * w;
					p = 2.2900482228026654717e-07 + p * w;
					p = -9.9298272942317002539e-07 + p * w;
					p = 4.5260625972231537039e-06 + p * w;
					p = -1.9681778105531670567e-05 + p * w;
					p = 7.5995277030017761139e-05 + p * w;
					p = -0.00021503011930044477347 + p * w;
					p = -0.00013871931833623122026 + p * w;
					p = 1.0103004648645343977 + p * w;
					p = 4.8499064014085844221 + p * w;
				}
				return p * x;
			}
		}

		/**
		 * @brief A pair of lower/upper bounds defining the support (truncation interval).
		 *
		 * @tparam Ty the scalar or vector type for the bounds
		 */
		template<typename Ty>
		struct Support
		{
			Ty lower;
			Ty upper;

			Support() = default;
			Support(const Ty& _lower, const Ty& _upper)
				: lower{ _lower }, upper{ _upper }
			{
				eigen_assert(_lower < _upper);
			}

			Support intersect(const Support& other) const
			{
				return {
					(lower > other.lower) ? lower : other.lower,
					(upper < other.upper) ? upper : other.upper
				};
			}
		};

		/**
		 * @brief Helper function to create a Support object with type deduction.
		 *
		 * @tparam Ty the scalar or vector type for the bounds
		 * @param lower lower bound of the support
		 * @param upper upper bound of the support
		 */
		template<typename Ty>
		inline Support<Ty> support(const Ty& lower, const Ty& upper)
		{
			return { lower, upper };
		}

		/**
		 * @brief Generator of reals on a truncated distribution (generic wrapper)
		 *
		 * @note This class is experimental and its interface may change in future versions.
		 *
		 * Uses rejection sampling with CompressMask to generate samples from
		 * any base distribution, rejecting values outside [lower, upper].
		 *
		 * @tparam BaseGen the underlying distribution generator type
		 */
		template<typename BaseGen>
		class TruncGen : OptCacheStore, public GenBase<TruncGen<BaseGen>, typename BaseGen::Scalar>
		{
			static_assert(std::is_floating_point<typename BaseGen::Scalar>::value, "truncDist needs floating point types.");
			using _Scalar = typename BaseGen::Scalar;
			int cache_rest_cnt = 0;
			BaseGen base;
			Support<_Scalar> support;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new truncated distribution generator
			 *
			 * @param _base the base distribution generator
			 * @param _lower lower bound of the truncation interval
			 * @param _upper upper bound of the truncation interval
			 */
			TruncGen(const BaseGen& _base,
				const Support<_Scalar>& _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support }
			{
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const BaseGen& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				for (int _i = 0; ; ++_i)
				{
					EIGENRAND_CHECK_INFINITY_LOOP();
					_Scalar v = base(rng);
					if (v >= support.lower && v <= support.upper) return v;
				}
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
				for (int _i = 0; ; ++_i)
				{
					EIGENRAND_CHECK_INFINITY_LOOP();
					Packet cands = base.template packetOp<Packet>(rng);
					auto mask = pand(pcmple(pset1<Packet>(support.lower), cands), pcmple(cands, pset1<Packet>(support.upper)));
					bool full = false;
					cache_rest_cnt = cm.compress_append(cands, mask,
						OptCacheStore::template get<Packet>(), cache_rest_cnt, full);
					if (full) return cands;
				}
			}
		};

		/**
		 * @brief Generator of reals on a truncated normal distribution (specialized, inverse CDF)
		 *
		 * @note This class is experimental and its interface may change in future versions.
		 *
		 * Uses the inverse CDF method: generates uniform samples in [Phi(a'), Phi(b')]
		 * then applies the inverse normal CDF via erfinv. O(1) per sample, no rejection.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncNormalGen : public GenBase<TruncNormalGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncNormalDist needs floating point types.");
			UniformRealGen<_Scalar> ur;
			_Scalar mean, stdev;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new truncated normal generator
			 *
			 * @param _mean mean of the normal distribution
			 * @param _stdev standard deviation of the normal distribution
			 * @param _lower lower bound of the truncation interval
			 * @param _upper upper bound of the truncation interval
			 */
			TruncNormalGen(_Scalar _mean = 0, _Scalar _stdev = 1,
				const Support<_Scalar>& _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: mean{ _mean }, stdev{ _stdev }
			{
				eigen_assert(_stdev > 0);
				// Compute Phi((lower - mean) / stdev) and Phi((upper - mean) / stdev)
				// where Phi(x) = 0.5 * erfc(-x / sqrt(2))
				_Scalar a_normalized = (_support.lower - _mean) / _stdev;
				_Scalar b_normalized = (_support.upper - _mean) / _stdev;
				_Scalar phi_lower = (_Scalar)0.5 * std::erfc(-a_normalized * (_Scalar)(1.0 / 1.4142135623730951));
				_Scalar phi_upper = (_Scalar)0.5 * std::erfc(-b_normalized * (_Scalar)(1.0 / 1.4142135623730951));
				ur = UniformRealGen<_Scalar>{ phi_lower, phi_upper };
			}
			TruncNormalGen(const TruncNormalGen&) = default;
			TruncNormalGen(TruncNormalGen&&) = default;

			TruncNormalGen& operator=(const TruncNormalGen&) = default;
			TruncNormalGen& operator=(TruncNormalGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				_Scalar u = ur(rng);
				// z = -sqrt(2) * erfinv(1 - 2*u)
				_Scalar z = (_Scalar)(-1.4142135623730951) * detail::scalar_erfinv((_Scalar)1 - (_Scalar)2 * u);
				return z * stdev + mean;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				Packet u = ur.template packetOp<Packet>(rng);
				// z = -sqrt(2) * erfinv(1 - 2*u)
				Packet arg = psub(pset1<Packet>(1), pmul(pset1<Packet>(2), u));
				Packet z = pmul(pset1<Packet>((_Scalar)-1.4142135623730951), perfinv(arg));
				return padd(pmul(z, pset1<Packet>(stdev)), pset1<Packet>(mean));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for NormalGen.
		 *
		 * When TruncGen is instantiated with NormalGen<_Scalar>, it automatically
		 * uses the inverse CDF method (TruncNormalGen) instead of rejection sampling.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<NormalGen<_Scalar>> : public GenBase<TruncGen<NormalGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			NormalGen<_Scalar> base;
			Support<_Scalar> support;
			TruncNormalGen<_Scalar> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const NormalGen<_Scalar>& _base,
				const Support<_Scalar>& _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  impl{ _base.mean, _base.stdev, _support }
			{
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const NormalGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return impl(std::forward<Rng>(rng));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return impl.template packetOp<Packet>(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for nested TruncGen.
		 *
		 * When TruncGen wraps another TruncGen, the bounds are merged (intersected)
		 * and the inner TruncGen is reconstructed with the tighter bounds.
		 * This works recursively for any depth of nesting.
		 *
		 * @tparam BaseGen the base generator type wrapped by the inner TruncGen
		 */
		template<typename BaseGen>
		class TruncGen<TruncGen<BaseGen>> : public GenBase<TruncGen<TruncGen<BaseGen>>, typename BaseGen::Scalar>
		{
			static_assert(std::is_floating_point<typename BaseGen::Scalar>::value, "truncDist needs floating point types.");
			using _Scalar = typename BaseGen::Scalar;
			TruncGen<BaseGen> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const TruncGen<BaseGen>& inner,
				const Support<_Scalar>& _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: impl{ inner.getBase(), inner.getSupport().intersect(_support) }
			{
			}
			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const TruncGen<BaseGen>& getBase() const { return impl; }
			Support<_Scalar> getSupport() const { return impl.getSupport(); }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return impl(std::forward<Rng>(rng));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return impl.template packetOp<Packet>(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for ExponentialGen.
		 *
		 * Uses the inverse CDF method: F(x) = 1 - exp(-lambda*x),
		 * F^{-1}(u) = -log(1-u) / lambda.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<ExponentialGen<_Scalar>> : public GenBase<TruncGen<ExponentialGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			ExponentialGen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> ur;
			_Scalar inv_lambda;

		public:
			using Scalar = _Scalar;

			TruncGen(const ExponentialGen<_Scalar>& _base,
				const Support<_Scalar>& _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  inv_lambda{ (_Scalar)1 / _base.lambda }
			{
				// Exponential is non-negative, clamp lower bound
				_Scalar eff_lower = support.lower < 0 ? (_Scalar)0 : support.lower;
				_Scalar phi_lower = (_Scalar)1 - std::exp(-_base.lambda * eff_lower);
				_Scalar phi_upper = (_Scalar)1 - std::exp(-_base.lambda * support.upper);
				ur = UniformRealGen<_Scalar>{ phi_lower, phi_upper };
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const ExponentialGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				_Scalar u = ur(rng);
				return -std::log((_Scalar)1 - u) * inv_lambda;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				Packet u = ur.template packetOp<Packet>(rng);
				return pmul(pnegate(plog(psub(pset1<Packet>((_Scalar)1), u))),
					pset1<Packet>(inv_lambda));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for CauchyGen.
		 *
		 * Uses the inverse CDF method: F(x) = 0.5 + atan((x-a)/b)/pi,
		 * F^{-1}(u) = a + b * tan(pi * (u - 0.5)).
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<CauchyGen<_Scalar>> : public GenBase<TruncGen<CauchyGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			CauchyGen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> ur;
			_Scalar a, b;

		public:
			using Scalar = _Scalar;

			TruncGen(const CauchyGen<_Scalar>& _base,
				const Support<_Scalar>& _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  a{ _base.a }, b{ _base.b }
			{
				_Scalar phi_lower = (_Scalar)0.5 + std::atan((support.lower - _base.a) / _base.b) / (_Scalar)constant::pi;
				_Scalar phi_upper = (_Scalar)0.5 + std::atan((support.upper - _base.a) / _base.b) / (_Scalar)constant::pi;
				ur = UniformRealGen<_Scalar>{ phi_lower, phi_upper };
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const CauchyGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				_Scalar u = ur(rng);
				return a + b * std::tan((_Scalar)constant::pi * (u - (_Scalar)0.5));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				Packet s, c;
				psincos(pmul(pset1<Packet>((_Scalar)constant::pi),
					psub(ur.template packetOp<Packet>(rng), pset1<Packet>((_Scalar)0.5))
				), s, c);
				return padd(pset1<Packet>(a), pmul(pset1<Packet>(b), pdiv(s, c)));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for LognormalGen.
		 *
		 * Uses the fact that if X ~ LogNormal(mu, sigma), then ln(X) ~ Normal(mu, sigma).
		 * Truncated LogNormal on [a, b] = exp(Truncated Normal on [ln(a), ln(b)]).
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<LognormalGen<_Scalar>> : public GenBase<TruncGen<LognormalGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			LognormalGen<_Scalar> base;
			Support<_Scalar> support;
			TruncNormalGen<_Scalar> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const LognormalGen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  impl{ _base.norm.mean, _base.norm.stdev,
					Support<_Scalar>{
						support.lower <= 0 ? std::numeric_limits<_Scalar>::lowest() : std::log(support.lower),
						std::log(support.upper) 
					}
				  }
			{
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const LognormalGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return std::exp(impl(std::forward<Rng>(rng)));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				return pexp(impl.template packetOp<Packet>(std::forward<Rng>(rng)));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for ExtremeValueGen (Gumbel).
		 *
		 * Uses the inverse CDF method: F(x) = exp(-exp(-(x-a)/b)),
		 * F^{-1}(u) = a - b * log(-log(u)).
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<ExtremeValueGen<_Scalar>> : public GenBase<TruncGen<ExtremeValueGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			ExtremeValueGen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> ur;
			_Scalar a, b;

		public:
			using Scalar = _Scalar;

			TruncGen(const ExtremeValueGen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  a{ _base.a }, b{ _base.b }
			{
				// CDF: F(x) = exp(-exp(-(x - a) / b))
				_Scalar phi_lower = std::exp(-std::exp(-(support.lower - _base.a) / _base.b));
				_Scalar phi_upper = std::exp(-std::exp(-(support.upper - _base.a) / _base.b));
				ur = UniformRealGen<_Scalar>{ phi_lower, phi_upper };
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const ExtremeValueGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				_Scalar u = ur(rng);
				return a - b * std::log(-std::log(u));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				Packet u = ur.template packetOp<Packet>(rng);
				return psub(pset1<Packet>(a),
					pmul(pset1<Packet>(b), plog(pnegate(plog(u)))));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for WeibullGen.
		 *
		 * Uses the inverse CDF method: F(x) = 1 - exp(-(x/b)^a),
		 * F^{-1}(u) = b * (-log(1-u))^{1/a} = b * exp(log(-log(1-u)) / a).
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<WeibullGen<_Scalar>> : public GenBase<TruncGen<WeibullGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			WeibullGen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> ur;
			_Scalar inv_a, b;

		public:
			using Scalar = _Scalar;

			TruncGen(const WeibullGen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  inv_a{ (_Scalar)1 / _base.a }, b{ _base.b }
			{
				// Weibull is non-negative, clamp lower bound
				_Scalar eff_lower = support.lower < 0 ? (_Scalar)0 : support.lower;
				// CDF: F(x) = 1 - exp(-(x/b)^a)
				_Scalar phi_lower = (_Scalar)1 - std::exp(-std::pow(eff_lower / _base.b, _base.a));
				_Scalar phi_upper = (_Scalar)1 - std::exp(-std::pow(support.upper / _base.b, _base.a));
				ur = UniformRealGen<_Scalar>{ phi_lower, phi_upper };
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const WeibullGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				_Scalar u = ur(rng);
				return std::pow(-std::log((_Scalar)1 - u), inv_a) * b;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				Packet u = ur.template packetOp<Packet>(rng);
				// b * exp(log(-log(1 - u)) / a)
				return pmul(pset1<Packet>(b),
					pexp(pmul(plog(pnegate(plog(psub(pset1<Packet>((_Scalar)1), u)))),
						pset1<Packet>(inv_a))));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for UniformRealGen.
		 *
		 * Truncating a uniform distribution is simply another uniform
		 * on the intersection of the original range and the truncation bounds.
		 * Zero runtime overhead.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<UniformRealGen<_Scalar>> : public GenBase<TruncGen<UniformRealGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			UniformRealGen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const UniformRealGen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support }
			{
				_Scalar orig_min = _base.bias;
				_Scalar orig_max = _base.bias + _base.slope;
				_Scalar eff_min = support.lower > orig_min ? support.lower : orig_min;
				_Scalar eff_max = support.upper < orig_max ? support.upper : orig_max;
				eigen_assert(eff_min < eff_max);
				impl = UniformRealGen<_Scalar>{ eff_min, eff_max };
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const UniformRealGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return impl(std::forward<Rng>(rng));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return impl.template packetOp<Packet>(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for StdNormalGen.
		 *
		 * Delegates to TruncNormalGen with mean=0, stdev=1.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<StdNormalGen<_Scalar>> : public GenBase<TruncGen<StdNormalGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			StdNormalGen<_Scalar> base;
			Support<_Scalar> support;
			TruncNormalGen<_Scalar> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const StdNormalGen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  impl{ 0, 1, _support }
			{
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const StdNormalGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return impl(std::forward<Rng>(rng));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return impl.template packetOp<Packet>(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for StdUniformRealGen.
		 *
		 * Truncating [0, 1) is simply UniformRealGen on the intersected range.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<StdUniformRealGen<_Scalar>> : public GenBase<TruncGen<StdUniformRealGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			StdUniformRealGen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const StdUniformRealGen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  impl{ support.lower > (_Scalar)0 ? support.lower : (_Scalar)0,
					support.upper < (_Scalar)1 ? support.upper : (_Scalar)1 }
			{
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const StdUniformRealGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return impl(std::forward<Rng>(rng));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return impl.template packetOp<Packet>(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for BalancedGen.
		 *
		 * Truncating [-1, 1] is simply UniformRealGen on the intersected range.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<BalancedGen<_Scalar>> : public GenBase<TruncGen<BalancedGen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			BalancedGen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const BalancedGen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  impl{ support.lower > (_Scalar)-1 ? support.lower : (_Scalar)-1,
					support.upper < (_Scalar)1 ? support.upper : (_Scalar)1 }
			{
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const BalancedGen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return impl(std::forward<Rng>(rng));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return impl.template packetOp<Packet>(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Partial specialization of TruncGen for Balanced2Gen.
		 *
		 * Truncating [a, b] is simply UniformRealGen on the intersected range.
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class TruncGen<Balanced2Gen<_Scalar>> : public GenBase<TruncGen<Balanced2Gen<_Scalar>>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "truncDist needs floating point types.");
			Balanced2Gen<_Scalar> base;
			Support<_Scalar> support;
			UniformRealGen<_Scalar> impl;

		public:
			using Scalar = _Scalar;

			TruncGen(const Balanced2Gen<_Scalar>& _base,
				Support<_Scalar> _support = Support<_Scalar>{ std::numeric_limits<_Scalar>::lowest(), std::numeric_limits<_Scalar>::max() })
				: base{ _base }, support{ _support },
				  impl{ support.lower > _base.bias ? support.lower : _base.bias,
					support.upper < _base.bias + _base.slope ? support.upper : _base.bias + _base.slope }
			{
			}

			TruncGen(const TruncGen&) = default;
			TruncGen(TruncGen&&) = default;

			TruncGen& operator=(const TruncGen&) = default;
			TruncGen& operator=(TruncGen&&) = default;

			const Balanced2Gen<_Scalar>& getBase() const { return base; }
			Support<_Scalar> getSupport() const { return support; }

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return impl(std::forward<Rng>(rng));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return impl.template packetOp<Packet>(std::forward<Rng>(rng));
			}
		};

		template<typename Derived, typename BaseGen, typename Urng>
		using TruncType = CwiseNullaryOp<internal::scalar_rng_adaptor<TruncGen<BaseGen>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on a truncated distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @tparam BaseGen the underlying distribution generator type
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param baseGen the base distribution generator
		 * @param support a Support<Scalar> object specifying the truncation interval
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 *
		 * @see Eigen::Rand::TruncGen
		 */
		template<typename Derived, typename Urng, typename BaseGen>
		inline const TruncType<Derived, BaseGen, Urng>
			truncated(Index rows, Index cols, Urng&& urng, const BaseGen& baseGen,
				const Support<typename Derived::Scalar>& support)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), TruncGen<BaseGen>{baseGen, support} }
			};
		}

		/**
		 * @brief generates reals on a truncated distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @tparam BaseGen the underlying distribution generator type
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param baseGen the base distribution generator
		 * @param support a Support<Scalar> object specifying the truncation interval
		 * @return a random matrix expression of the same shape as `o`
		 *
		 * @see Eigen::Rand::TruncGen
		 */
		template<typename Derived, typename Urng, typename BaseGen>
		inline const TruncType<Derived, BaseGen, Urng>
			truncatedLike(Derived& o, Urng&& urng, const BaseGen& baseGen,
				const Support<typename Derived::Scalar>& support)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), TruncGen<BaseGen>{baseGen, support} }
			};
		}

		template<typename Derived, typename Urng>
		using TruncNormalType = CwiseNullaryOp<internal::scalar_rng_adaptor<TruncNormalGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on a truncated normal distribution using the inverse CDF method.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the normal distribution
		 * @param stdev a standard deviation of the normal distribution
		 * @param support a Support<Scalar> object specifying the truncation interval
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 *
		 * @see Eigen::Rand::TruncNormalGen
		 */
		template<typename Derived, typename Urng>
		inline const TruncNormalType<Derived, Urng>
			truncatedNormal(Index rows, Index cols, Urng&& urng,
				typename Derived::Scalar mean, typename Derived::Scalar stdev,
				const Support<typename Derived::Scalar>& support)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), TruncNormalGen<typename Derived::Scalar>{mean, stdev, support} }
			};
		}

		/**
		 * @brief generates reals on a truncated normal distribution using the inverse CDF method.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the normal distribution
		 * @param stdev a standard deviation of the normal distribution
		 * @param support a Support<Scalar> object specifying the truncation interval
		 * @return a random matrix expression of the same shape as `o`
		 *
		 * @see Eigen::Rand::TruncNormalGen
		 */
		template<typename Derived, typename Urng>
		inline const TruncNormalType<Derived, Urng>
			truncatedNormalLike(Derived& o, Urng&& urng,
				typename Derived::Scalar mean, typename Derived::Scalar stdev,
				const Support<typename Derived::Scalar>& support)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), TruncNormalGen<typename Derived::Scalar>{mean, stdev, support} }
			};
		}
	}
}

#endif
