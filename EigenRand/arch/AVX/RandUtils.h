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

#ifndef EIGENRAND_RAND_UTILS_AVX_H
#define EIGENRAND_RAND_UTILS_AVX_H

#include <immintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Rng>
		struct RawbitsMaker<Packet4i, Rng, Packet8i, Rand::RandomEngineType::packet>
		{
			EIGEN_STRONG_INLINE Packet4i rawbits(Rng& rng)
			{
				return rng.half();
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_34(Rng& rng)
			{
				return rng.half();
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				return rng.half();
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet8i, Rng, Packet4i, Rand::RandomEngineType::packet>
		{
			EIGEN_STRONG_INLINE Packet8i rawbits(Rng& rng)
			{
				return _mm256_insertf128_si256(_mm256_castsi128_si256(rng()), rng(), 1);
			}

			EIGEN_STRONG_INLINE Packet8i rawbits_34(Rng& rng)
			{
				return _mm256_insertf128_si256(_mm256_castsi128_si256(rng()), rng(), 1);
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				return rng();
			}
		};

		template<typename Rng, typename RngResult>
		struct RawbitsMaker<Packet8i, Rng, RngResult, Rand::RandomEngineType::scalar_fullbit>
		{
			EIGEN_STRONG_INLINE Packet8i rawbits(Rng& rng)
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

			EIGEN_STRONG_INLINE Packet8i rawbits_34(Rng& rng)
			{
				Packet8i p;
				if (sizeof(decltype(rng())) == 8)
				{
#ifdef EIGEN_VECTORIZE_AVX2
					p = _mm256_setr_epi64x(rng(), rng(), rng(), 0);
					p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 1, 2, 7, 3, 4, 5, 7));
					p = _mm256_shuffle_epi8(p, _mm256_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11,
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11
					));

#else
					auto v = rng();
					p = _mm256_setr_epi64x(rng(), v, rng(), v >> 32);
					Packet4i p1, p2, o = _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11);
					split_two(p, p1, p2);
					p = combine_two(_mm_shuffle_epi8(p1, o), _mm_shuffle_epi8(p2, o));
#endif
				}
				else
				{
					p = _mm256_setr_epi32(rng(), rng(), rng(), 0, rng(), rng(), rng(), 0);
#ifdef EIGEN_VECTORIZE_AVX2
					p = _mm256_shuffle_epi8(p, _mm256_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11,
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11
					));
#else
					Packet4i p1, p2, o = _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11);
					split_two(p, p1, p2);
					p = combine_two(_mm_shuffle_epi8(p1, o), _mm_shuffle_epi8(p2, o));
#endif
				}
				return p;
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), rng(), rng());
				}
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet8i, Rng, Packet8i, Rand::RandomEngineType::packet>
		{
			EIGEN_STRONG_INLINE Packet8i rawbits(Rng& rng)
			{
				return rng();
			}

			EIGEN_STRONG_INLINE Packet8i rawbits_34(Rng& rng)
			{
				return rng();
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				return rng.half();
			}
		};

#ifndef EIGEN_VECTORIZE_AVX2
		template<>
		EIGEN_STRONG_INLINE Packet8f bit_to_ur_float<Packet8i>(const Packet8i& x)
		{
			const Packet4i lower = pset1<Packet4i>(0x7FFFFF),
				upper = pset1<Packet4i>(127 << 23);
			const Packet8f one = pset1<Packet8f>(1);

			Packet4i x1, x2;
			split_two(x, x1, x2);

			return psub(reinterpret_to_float(
				combine_two(por(pand(x1, lower), upper), por(pand(x2, lower), upper)
				)), one);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d bit_to_ur_double<Packet8i>(const Packet8i& x)
		{
			const Packet4i lower = pseti64<Packet4i>(0xFFFFFFFFFFFFFull),
				upper = pseti64<Packet4i>(1023ull << 52);
			const Packet4d one = pset1<Packet4d>(1);

			Packet4i x1, x2;
			split_two(x, x1, x2);

			return psub(reinterpret_to_double(
				combine_two(por(pand(x1, lower), upper), por(pand(x2, lower), upper)
				)), one);
		}
#endif

		template<typename Rng>
		struct UniformRealUtils<Packet8f, Rng> : public RawbitsMaker<Packet8i, Rng>
		{
			EIGEN_STRONG_INLINE Packet8f zero_to_one(Rng& rng)
			{
				return pdiv(_mm256_cvtepi32_ps(pand(this->rawbits(rng), pset1<Packet8i>(0x7FFFFFFF))),
					pset1<Packet8f>(0x7FFFFFFF));
			}

			EIGEN_STRONG_INLINE Packet8f uniform_real(Rng& rng)
			{
				return bit_to_ur_float(this->rawbits_34(rng));
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet4d, Rng> : public RawbitsMaker<Packet8i, Rng>
		{
			EIGEN_STRONG_INLINE Packet4d zero_to_one(Rng& rng)
			{
				return pdiv(_mm256_cvtepi32_pd(pand(this->rawbits_half(rng), pset1<Packet4i>(0x7FFFFFFF))),
					pset1<Packet4d>(0x7FFFFFFF));
			}

			EIGEN_STRONG_INLINE Packet4d uniform_real(Rng& rng)
			{
				return bit_to_ur_double(this->rawbits(rng));
			}
		};
	}
}
#endif