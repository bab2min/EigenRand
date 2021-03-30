/**
 * @file RandUtils.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.3.3
 * @date 2021-03-31
 * 
 * @copyright Copyright (c) 2020-2021
 * 
 */

#ifndef EIGENRAND_RAND_UTILS_H
#define EIGENRAND_RAND_UTILS_H

#include <EigenRand/MorePacketMath.h>
#include <EigenRand/PacketFilter.h>
#include <EigenRand/PacketRandomEngine.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Packet, typename Rng,
			typename RngResult = typename std::remove_reference<Rng>::type::result_type,
			Rand::RandomEngineType reType = Rand::GetRandomEngineType<
			typename std::remove_reference<Rng>::type
		>::value>
		struct RawbitsMaker;

		template<typename PacketType, typename Rng>
		struct UniformRealUtils;

		template<typename PacketType, typename Rng>
		struct RandUtils : public UniformRealUtils<PacketType, Rng>
		{
			EIGEN_STRONG_INLINE PacketType balanced(Rng& rng)
			{
				return psub(pmul(this->zero_to_one(rng), pset1<PacketType>(2)), pset1<PacketType>(1));
			}

			EIGEN_STRONG_INLINE PacketType nonzero_uniform_real(Rng& rng)
			{
				constexpr auto epsilon = std::numeric_limits<typename unpacket_traits<PacketType>::type>::epsilon() / 8;
				return padd(this->uniform_real(rng), pset1<PacketType>(epsilon));
			}
		};

		EIGEN_STRONG_INLINE uint32_t collect_upper8bits(uint32_t a, uint32_t b, uint32_t c)
		{
			return ((a & 0xFF000000) >> 24) | ((b & 0xFF000000) >> 16) | ((c & 0xFF000000) >> 8);
		}
	}
}

#ifdef EIGEN_VECTORIZE_AVX
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
		struct RawbitsMaker<Packet8i, Rng, RngResult, Rand::RandomEngineType::scalar>
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

#ifdef EIGEN_VECTORIZE_SSE2
#include <xmmintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Rng, typename RngResult>
		struct RawbitsMaker<Packet4i, Rng, RngResult, Rand::RandomEngineType::scalar>
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

namespace Eigen
{
	namespace internal
	{
		template<typename Gen, typename _Scalar, typename Rng, bool _mutable = false>
		struct scalar_rng_adaptor
		{
			static_assert(
				Rand::IsScalarRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value, "Rng must satisfy RandomNumberEngine");

			Gen gen;
			Rng rng;

			scalar_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_rng_adaptor(const scalar_rng_adaptor& o) = default;
			scalar_rng_adaptor(scalar_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() () const
			{
				return gen(rng);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				return gen.template packetOp<Packet>(rng);
			}
		};

		template<typename Gen, typename _Scalar, typename Rng>
		struct scalar_rng_adaptor<Gen, _Scalar, Rng, true>
		{
			static_assert(
				Rand::IsScalarRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value, "Rng must satisfy RandomNumberEngine");

			mutable Gen gen;
			Rng rng;

			scalar_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_rng_adaptor(const scalar_rng_adaptor& o) = default;
			scalar_rng_adaptor(scalar_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() () const
			{
				return gen(rng);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				return gen.template packetOp<Packet>(rng);
			}
		};

		template<typename Gen, typename _Scalar, typename Urng, bool _mutable>
		struct functor_traits<scalar_rng_adaptor<Gen, _Scalar, Urng, _mutable> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<_Scalar>::Vectorizable, IsRepeatable = false };
		};
	}
}

#endif
