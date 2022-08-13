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

#ifndef EIGENRAND_RAND_UTILS_SSE_H
#define EIGENRAND_RAND_UTILS_SSE_H

#include <xmmintrin.h>

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
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), rng(), rng());
				}
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_34(Rng& rng)
			{
				if (sizeof(RngResult) == 8)
				{
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
#ifdef EIGEN_VECTORIZE_SSSE3
					Packet4i p = _mm_setr_epi32(rng(), rng(), rng(), 0);
					return _mm_shuffle_epi8(p, _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11));
#else
					return _mm_set_epi32(rng(), rng(), rng(), rng());
#endif
				}
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm_set_epi64x(0, rng());
				}
				else
				{
					return _mm_setr_epi32(rng(), rng(), 0, 0);
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
				return pdiv((Packet4f)_mm_cvtepi32_ps(pand(this->rawbits(rng), pset1<Packet4i>(0x7FFFFFFF))),
					pset1<Packet4f>(0x7FFFFFFF));
			}

			EIGEN_STRONG_INLINE Packet4f uniform_real(Rng& rng)
			{
				return bit_to_ur_float(this->rawbits_34(rng));
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet2d, Rng> : public RawbitsMaker<Packet4i, Rng>
		{
			EIGEN_STRONG_INLINE Packet2d zero_to_one(Rng& rng)
			{
				return pdiv((Packet2d)_mm_cvtepi32_pd(pand(this->rawbits_half(rng), pset1<Packet4i>(0x7FFFFFFF))),
					pset1<Packet2d>(0x7FFFFFFF));
			}

			EIGEN_STRONG_INLINE Packet2d uniform_real(Rng& rng)
			{
				return bit_to_ur_double(this->rawbits(rng));
			}
		};
	}
}
#endif