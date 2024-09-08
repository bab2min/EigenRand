/**
 * @file RandUtils.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.5.1
 * @date 2024-09-08
 *
 * @copyright Copyright (c) 2020-2024
 *
 */

#ifndef EIGENRAND_RAND_UTILS_AVX512_H
#define EIGENRAND_RAND_UTILS_AVX512_H

#include <immintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Rng>
		struct RawbitsMaker<Packet8i, Rng, Packet16i, Rand::RandomEngineType::packet>
		{
			EIGEN_STRONG_INLINE Packet8i rawbits(Rng& rng)
			{
				return rng.half();
			}

			EIGEN_STRONG_INLINE Packet8i rawbits_34(Rng& rng)
			{
				return rng.half();
			}

			EIGEN_STRONG_INLINE Packet8i rawbits_half(Rng& rng)
			{
				return rng.half();
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet16i, Rng, Packet8i, Rand::RandomEngineType::packet>
		{
			EIGEN_STRONG_INLINE Packet16i rawbits(Rng& rng)
			{
				return _mm512_inserti64x4(_mm512_castsi256_si512(rng()), rng(), 1);
			}

			EIGEN_STRONG_INLINE Packet16i rawbits_34(Rng& rng)
			{
				return _mm512_inserti64x4(_mm512_castsi256_si512(rng()), rng(), 1);
			}

			EIGEN_STRONG_INLINE Packet8i rawbits_half(Rng& rng)
			{
				return rng();
			}
		};

		template<typename Rng, typename RngResult>
		struct RawbitsMaker<Packet16i, Rng, RngResult, Rand::RandomEngineType::scalar_fullbit>
		{
			EIGEN_STRONG_INLINE Packet16i rawbits(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm512_set_epi64(rng(), rng(), rng(), rng(),
						rng(), rng(), rng(), rng());
				}
				else
				{
					return _mm512_set_epi32(rng(), rng(), rng(), rng(),
						rng(), rng(), rng(), rng(),
						rng(), rng(), rng(), rng(),
						rng(), rng(), rng(), rng());
				}
			}

			EIGEN_STRONG_INLINE Packet16i rawbits_34(Rng& rng)
			{
				return rawbits(rng);
			}

			EIGEN_STRONG_INLINE Packet8i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm256_set_epi64x(rng(), rng(), rng(), rng());
				}
				else
				{
					return _mm256_set_epi32(rng(), rng(), rng(), rng(),
						rng(), rng(), rng(), rng());
				}
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet16i, Rng, Packet16i, Rand::RandomEngineType::packet>
		{
			EIGEN_STRONG_INLINE Packet16i rawbits(Rng& rng)
			{
				return rng();
			}

			EIGEN_STRONG_INLINE Packet16i rawbits_34(Rng& rng)
			{
				return rng();
			}

			EIGEN_STRONG_INLINE Packet8i rawbits_half(Rng& rng)
			{
				return rng.half();
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet16f, Rng> : public RawbitsMaker<Packet16i, Rng>
		{
			EIGEN_STRONG_INLINE Packet16f zero_to_one(Rng& rng)
			{
				return pdiv(_mm512_cvtepi32_ps(pand(this->rawbits(rng), pset1<Packet16i>(0x7FFFFFFF))),
					pset1<Packet16f>(0x7FFFFFFF));
			}

			EIGEN_STRONG_INLINE Packet16f uniform_real(Rng& rng)
			{
				return bit_to_ur_float(this->rawbits_34(rng));
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet8d, Rng> : public RawbitsMaker<Packet16i, Rng>
		{
			EIGEN_STRONG_INLINE Packet8d zero_to_one(Rng& rng)
			{
				return pdiv(_mm512_cvtepi32_pd(pand(this->rawbits_half(rng), pset1<Packet8i>(0x7FFFFFFF))),
					pset1<Packet8d>(0x7FFFFFFF));
			}

			EIGEN_STRONG_INLINE Packet8d uniform_real(Rng& rng)
			{
				return bit_to_ur_double(this->rawbits(rng));
			}
		};
	}
}
#endif
