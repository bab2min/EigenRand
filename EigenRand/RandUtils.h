#ifndef EIGENRAND_RAND_UTILS_H
#define EIGENRAND_RAND_UTILS_H

#include <EigenRand/MorePacketMath.h>
#include <EigenRand/PacketFilter.h>
#include <EigenRand/PacketRandomEngine.h>

namespace Eigen
{
	namespace internal
	{
		template<typename IntPacket>
		EIGEN_STRONG_INLINE auto bit_to_ur_float(const IntPacket& x) -> decltype(reinterpret_to_float(x))
		{
			using FloatPacket = decltype(reinterpret_to_float(x));

			const IntPacket lower = pset1<IntPacket>(0x7FFFFF),
				upper = pset1<IntPacket>(127 << 23);
			const FloatPacket one = pset1<FloatPacket>(1);

			return psub(reinterpret_to_float(por(pand(x, lower), upper)), one);
		}

		template<typename IntPacket>
		EIGEN_STRONG_INLINE auto bit_to_ur_double(const IntPacket& x) -> decltype(reinterpret_to_double(x))
		{
			using DoublePacket = decltype(reinterpret_to_double(x));

			const IntPacket lower = pseti64<IntPacket>(0xFFFFFFFFFFFFFull),
				upper = pseti64<IntPacket>(1023ull << 52);
			const DoublePacket one = pset1<DoublePacket>(1);

			return psub(reinterpret_to_double(por(pand(x, lower), upper)), one);
		}

		template<typename Scalar>
		struct bit_scalar;

		template<>
		struct bit_scalar<float>
		{
			float to_ur(uint32_t x)
			{
				union
				{
					uint32_t u;
					float f;
				};
				u = (x & 0x7FFFFF) | (127 << 23);
				return f - 1.f;
			}

			float to_nzur(uint32_t x)
			{
				return to_ur(x) + std::numeric_limits<float>::epsilon() / 8;
			}
		};

		template<>
		struct bit_scalar<double>
		{
			double to_ur(uint64_t x)
			{
				union
				{
					uint64_t u;
					double f;
				};
				u = (x & 0xFFFFFFFFFFFFFull) | (1023ull << 52);
				return f - 1.;
			}

			double to_nzur(uint64_t x)
			{
				return to_ur(x) + std::numeric_limits<double>::epsilon() / 8;
			}
		};

		template<typename Packet, typename Rng,
			typename RngResult = typename std::remove_reference<Rng>::type::result_type,
			Rand::RandomEngineType reType = Rand::GetRandomEngieType<
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
					Packet4i p = _mm_setr_epi32(rng(), rng(), rng(), 0);
					return _mm_shuffle_epi8(p, _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11));
				}
			}

			EIGEN_STRONG_INLINE Packet4i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm_set_epi64x(rng(), 0);
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), 0, 0);
				}
			}
		};

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
		EIGEN_STRONG_INLINE  Packet8f bit_to_ur_float<Packet8i>(const Packet8i& x)
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
		EIGEN_STRONG_INLINE  Packet4d bit_to_ur_double<Packet8i>(const Packet8i& x)
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

#elif defined(EIGEN_VECTORIZE_SSE2)
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
					return _mm_setr_epi64x(rng(), 0);
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
				return pdiv(_mm_cvtepi32_ps(pand(this->rawbits(rng), pset1<Packet4i>(0x7FFFFFFF))),
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
				return pdiv(_mm_cvtepi32_pd(pand(this->rawbits_half(rng), pset1<Packet4i>(0x7FFFFFFF))),
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
		template<typename Scalar, typename Rng>
		struct scalar_base_rng
		{
			static_assert(
				Rand::IsScalarRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value, "Rng must satisfy RandomNumberEngine");

			Rng rng;

			scalar_base_rng(const Rng& _rng) : rng{ _rng }
			{
			}

			scalar_base_rng(const scalar_base_rng& o)
				: rng{ o.rng }
			{
			}

			scalar_base_rng(scalar_base_rng&& o)
				: rng{ std::move(o.rng) }
			{
			}
		};
	}
}

#endif