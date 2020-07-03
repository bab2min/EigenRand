#pragma once
/**
 * @file GammaPoisson.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.2.0
 * @date 2020-06-22
 *
 * @copyright Copyright (c) 2020
 *
 */

#ifndef EIGENRAND_DISTS_GAMMAPOISSON_H
#define EIGENRAND_DISTS_GAMMAPOISSON_H

#include <memory>
#include <iterator>
#include <limits>

namespace Eigen
{
	namespace internal
	{
		template<typename Scalar, typename Rng>
		struct scalar_negative_binomial_dist_op : public scalar_gamma_dist_op<float, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "uniformInt needs integral types.");

			scalar_negative_binomial_dist_op(const Rng& _rng, Scalar _trials = 1, double _p = 0.5)
				: scalar_gamma_dist_op<float, Rng>{ _rng, (float)_trials, (float)((1 - _p) / _p) }

			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				float v = scalar_gamma_dist_op<float, Rng>::operator()();
				return scalar_poisson_dist_op<Scalar, Rng>{ this->rng, v }();
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using ur_base = scalar_uniform_real_op<float, Rng>;
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));
				
				auto mean = scalar_gamma_dist_op<float, Rng>::template packetOp<PacketType>();
				auto res = pset1<Packet>(0);
				PacketType val = pset1<PacketType>(1), pne_mean = pexp(pnegate(mean));
				if (pmovemask(pcmplt(pset1<PacketType>(12), mean)) == 0)
				{
					while (1)
					{
						val = pmul(val, ur_base::template packetOp<PacketType>());
						auto c = reinterpret_to_int(pcmplt(pne_mean, val));
						if (pmovemask(c) == 0) break;
						res = padd(res, pnegate(c));
					}
					return res;
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					thread_local PacketType cache_rest;
					thread_local int cache_rest_cnt;
					thread_local const scalar_negative_binomial_dist_op* cache_ptr = nullptr;
					if (cache_ptr != this)
					{
						cache_ptr = this;
						cache_rest = pset1<PacketType>(0);
						cache_rest_cnt = 0;
					}

					const PacketType ppi = pset1<PacketType>(constant::pi),
						psqrt_tmean = psqrt(pmul(pset1<PacketType>(2), mean)),
						plog_mean = plog(mean),
						pg1 = psub(pmul(mean, plog_mean), plgamma(padd(mean, pset1<PacketType>(1))));
					while (1)
					{
						PacketType fres, yx, psin, pcos;
						psincos(pmul(ppi, ur_base::template packetOp<PacketType>()), psin, pcos);
						yx = pdiv(psin, pcos);
						fres = ptruncate(padd(pmul(psqrt_tmean, yx), mean));

						auto p1 = pmul(padd(pmul(yx, yx), pset1<PacketType>(1)), pset1<PacketType>(0.9));
						auto p2 = pexp(psub(psub(pmul(fres, plog_mean), plgamma(padd(fres, pset1<PacketType>(1)))), pg1));

						auto c1 = pcmple(pset1<PacketType>(0), fres);
						auto c2 = pcmple(ur_base::template packetOp<PacketType>(), pmul(p1, p2));

						auto cands = fres;
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pand(c1, c2),
							cache_rest, cache_rest_cnt, full);
						if (full) return pcast<PacketType, Packet>(cands);
					}
				}
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_negative_binomial_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};
	}
}
#endif
