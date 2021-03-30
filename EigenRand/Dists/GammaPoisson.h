/**
 * @file GammaPoisson.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.3.3
 * @date 2021-03-31
 *
 * @copyright Copyright (c) 2020-2021
 *
 */

#ifndef EIGENRAND_DISTS_GAMMAPOISSON_H
#define EIGENRAND_DISTS_GAMMAPOISSON_H

#include <memory>
#include <iterator>
#include <limits>

namespace Eigen
{
	namespace Rand
	{
		/**
		 * @brief Generator of integers on a negative binomial distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class NegativeBinomialGen : public GenBase<NegativeBinomialGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "negativeBinomial needs integral types."); 
			UniformRealGen<float> ur;
			GammaGen<float> gamma;
		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new negative binomial generator
			 * 
			 * @param _trials the number of trial successes
			 * @param _p probability of a trial generating true
			 */
			NegativeBinomialGen(_Scalar _trials = 1, double _p = 0.5)
				: gamma{ (float)_trials, (float)((1 - _p) / _p) }

			{
			}

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				float v = gamma(rng);
				return PoissonGen<_Scalar>{v}(rng);
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using ur_base = UniformRealGen<float>;
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));

				auto mean = gamma.template packetOp<PacketType>(rng);
				auto res = pset1<Packet>(0);
				PacketType val = pset1<PacketType>(1), pne_mean = pexp(pnegate(mean));
				if (pmovemask(pcmplt(pset1<PacketType>(12), mean)) == 0)
				{
					while (1)
					{
						val = pmul(val, ur.template packetOp<PacketType>(rng));
						auto c = reinterpret_to_int(pcmplt(pne_mean, val));
						if (pmovemask(c) == 0) break;
						res = padd(res, pnegate(c));
					}
					return res;
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					const PacketType ppi = pset1<PacketType>(constant::pi),
						psqrt_tmean = psqrt(pmul(pset1<PacketType>(2), mean)),
						plog_mean = plog(mean),
						pg1 = psub(pmul(mean, plog_mean), plgamma_approx(padd(mean, pset1<PacketType>(1))));
					while (1)
					{
						PacketType fres, yx, psin, pcos;
						psincos(pmul(ppi, ur.template packetOp<PacketType>(rng)), psin, pcos);
						yx = pdiv(psin, pcos);
						fres = ptruncate(padd(pmul(psqrt_tmean, yx), mean));

						auto p1 = pmul(padd(pmul(yx, yx), pset1<PacketType>(1)), pset1<PacketType>(0.9));
						auto p2 = pexp(psub(psub(pmul(fres, plog_mean), plgamma_approx(padd(fres, pset1<PacketType>(1)))), pg1));

						auto c1 = pcmple(pset1<PacketType>(0), fres);
						auto c2 = pcmple(ur.template packetOp<PacketType>(rng), pmul(p1, p2));

						auto cands = fres;
						bool full = false;
						gamma.cache_rest_cnt = cm.compress_append(cands, pand(c1, c2),
							gamma.template get<PacketType>(), gamma.cache_rest_cnt, full);
						if (full) return pcast<PacketType, Packet>(cands);
					}
				}
			}
		};

		template<typename Derived, typename Urng>
		using NegativeBinomialType = CwiseNullaryOp<internal::scalar_rng_adaptor<NegativeBinomialGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

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
		 * 
		 * @see Eigen::Rand::NegativeBinomialGen
		 */
		template<typename Derived, typename Urng>
		inline const NegativeBinomialType<Derived, Urng>
			negativeBinomial(Index rows, Index cols, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), NegativeBinomialGen<typename Derived::Scalar>{trials, p} }
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
		 * 
		 * @see Eigen::Rand::NegativeBinomialGen
		 */
		template<typename Derived, typename Urng>
		inline const NegativeBinomialType<Derived, Urng>
			negativeBinomialLike(Derived& o, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), NegativeBinomialGen<typename Derived::Scalar>{trials, p} }
			};
		}
	}
}
#endif
