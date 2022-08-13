/**
 * @file RandUtils.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.4.1
 * @date 2022-08-13
 *
 * @copyright Copyright (c) 2020-2021
 *
 */

#ifndef EIGENRAND_RAND_UTILS_NEON_H
#define EIGENRAND_RAND_UTILS_NEON_H

#include <arm_neon.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Rng, typename RngResult>
		struct RawbitsMaker<Packet4i, Rng, RngResult, Rand::RandomEngineType::scalar_fullbit>
		{
			EIGEN_STRONG_INLINE Packet4i rawbits(Rng& rng)
			{
				if (sizeof(RngResult) == 8)
				{
					uint64_t v[2];
					v[0] = rng();
					v[1] = rng();
					return vld1q_s32((int32_t*)v);
				}
				else
				{
					uint32_t v[4];
					v[0] = rng();
					v[1] = rng();
					v[2] = rng();
					v[3] = rng();
					return vld1q_s32((int32_t*)v);
				}
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_34(Rng& rng)
			{
				if (sizeof(RngResult) == 8)
				{
					uint64_t v[2];
					v[0] = rng();
					v[1] = rng();
					return vld1q_s32((int32_t*)v);
				}
				else
				{
					uint32_t v[4];
					v[0] = rng();
					v[1] = rng();
					v[2] = rng();
					v[3] = rng();
					return vld1q_s32((int32_t*)v);
				}
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					uint64_t v[2];
					v[0] = rng();
					v[1] = 0;
					return vld1q_s32((int32_t*)v);
				}
				else
				{
					uint32_t v[4];
					v[0] = rng();
					v[1] = rng();
					v[2] = 0;
					v[3] = 0;
					return vld1q_s32((int32_t*)v);
				}
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet4i, Rng, Packet4i, Rand::RandomEngineType::packet>
		{
			EIGEN_STRONG_INLINE Packet4i rawbits(Rng& rng)
			{
				return rng();
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_34(Rng& rng)
			{
				return rng();
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				return rng();
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet4f, Rng> : public RawbitsMaker<Packet4i, Rng>
		{
			EIGEN_STRONG_INLINE Packet4f zero_to_one(Rng& rng)
			{
				return pdiv((Packet4f)vcvtq_f32_s32(pand(this->rawbits(rng), pset1<Packet4i>(0x7FFFFFFF))),
					pset1<Packet4f>(0x7FFFFFFF));
			}

			EIGEN_STRONG_INLINE Packet4f uniform_real(Rng& rng)
			{
				return bit_to_ur_float(this->rawbits_34(rng));
			}
		};

		template<typename Gen, typename Urng, bool _mutable>
		struct functor_traits<scalar_rng_adaptor<Gen, double, Urng, _mutable> >
		{
			enum { Cost = HugeCost, PacketAccess = 0, IsRepeatable = false };
		};
	}
}
#endif
