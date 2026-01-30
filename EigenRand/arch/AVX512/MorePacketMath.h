/**
 * @file MorePacketMath.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.6.0
 * @date 2026-01-31
 *
 * @copyright Copyright (c) 2020-2026
 *
 */

#ifndef EIGENRAND_MORE_PACKET_MATH_AVX512_H
#define EIGENRAND_MORE_PACKET_MATH_AVX512_H

#include <immintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<>
		struct IsIntPacket<Packet16i> : std::true_type {};

		template<>
		struct HalfPacket<Packet16i>
		{
			using type = Packet8i;
		};

		template<>
		struct HalfPacket<Packet16f>
		{
			using type = Packet8f;
		};

		template<>
		struct IsFloatPacket<Packet16f> : std::true_type {};

		template<>
		struct IsDoublePacket<Packet8d> : std::true_type {};

		template<>
		struct reinterpreter<Packet8d>
		{
			EIGEN_STRONG_INLINE Packet16f to_float(const Packet8d& x)
			{
				return _mm512_castpd_ps(x);
			}

			EIGEN_STRONG_INLINE Packet8d to_double(const Packet8d& x)
			{
				return x;
			}

#ifdef EIGENRAND_EIGEN_50_MODE
			// Eigen 5.x: return Packet8l (8x int64)
			EIGEN_STRONG_INLINE Packet8l to_int(const Packet8d& x)
			{
				return Packet8l(_mm512_castpd_si512(__m512d(x)));
			}

			// For 32-bit int operations, combine to Packet16i
			EIGEN_STRONG_INLINE Packet16i to_int32(const Packet8d& x)
			{
				return _mm512_castpd_si512(__m512d(x));
			}
#else
			EIGEN_STRONG_INLINE Packet16i to_int(const Packet8d& x)
			{
				return _mm512_castpd_si512(x);
			}
#endif
		};

		template<>
		struct reinterpreter<Packet16i>
		{
			EIGEN_STRONG_INLINE Packet16f to_float(const Packet16i& x)
			{
				return _mm512_castsi512_ps(__m512i(x));
			}

			EIGEN_STRONG_INLINE Packet8d to_double(const Packet16i& x)
			{
				return _mm512_castsi512_pd(__m512i(x));
			}

			EIGEN_STRONG_INLINE Packet16i to_int(const Packet16i& x)
			{
				return x;
			}

			EIGEN_STRONG_INLINE Packet16i to_int32(const Packet16i& x)
			{
				return x;
			}
		};

		template<>
		struct reinterpreter<Packet16f>
		{
			EIGEN_STRONG_INLINE Packet16f to_float(const Packet16f& x)
			{
				return x;
			}

			EIGEN_STRONG_INLINE Packet8d to_double(const Packet16f& x)
			{
				return _mm512_castps_pd(__m512(x));
			}

			EIGEN_STRONG_INLINE Packet16i to_int(const Packet16f& x)
			{
				return _mm512_castps_si512(__m512(x));
			}

			EIGEN_STRONG_INLINE Packet16i to_int32(const Packet16f& x)
			{
				return _mm512_castps_si512(__m512(x));
			}
		};

#ifdef EIGENRAND_EIGEN_50_MODE
		template<>
		struct IsIntPacket<Packet8l> : std::true_type {};

		template<>
		struct reinterpreter<Packet8l>
		{
			EIGEN_STRONG_INLINE Packet16f to_float(const Packet8l& x)
			{
				return _mm512_castsi512_ps(__m512i(x));
			}

			EIGEN_STRONG_INLINE Packet8d to_double(const Packet8l& x)
			{
				return _mm512_castsi512_pd(__m512i(x));
			}

			EIGEN_STRONG_INLINE Packet8l to_int(const Packet8l& x)
			{
				return x;
			}

			// Reinterpret as Packet16i (same underlying __m512i)
			EIGEN_STRONG_INLINE Packet16i to_int32(const Packet8l& x)
			{
				return Packet16i(__m512i(x));
			}
		};

		template<>
		struct BitShifter<Packet8l>
		{
			template<int b>
			EIGEN_STRONG_INLINE Packet8l sll(const Packet8l& a)
			{
				return Packet8l(_mm512_slli_epi64(__m512i(a), b));
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet8l srl(const Packet8l& a, int _b = b)
			{
				if (b >= 0)
				{
					return Packet8l(_mm512_srli_epi64(__m512i(a), b));
				}
				else
				{
					return Packet8l(_mm512_srli_epi64(__m512i(a), _b));
				}
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet8l sll64(const Packet8l& a)
			{
				return Packet8l(_mm512_slli_epi64(__m512i(a), b));
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet8l srl64(const Packet8l& a)
			{
				return Packet8l(_mm512_srli_epi64(__m512i(a), b));
			}
		};

		template<>
		EIGEN_STRONG_INLINE Packet8l pseti64<Packet8l>(uint64_t a)
		{
			return Packet8l(_mm512_set1_epi64(a));
		}

		template<>
		EIGEN_STRONG_INLINE Packet8l padd64<Packet8l>(const Packet8l& a, const Packet8l& b)
		{
			return Packet8l(_mm512_add_epi64(__m512i(a), __m512i(b)));
		}

		template<>
		EIGEN_STRONG_INLINE Packet8l psub64<Packet8l>(const Packet8l& a, const Packet8l& b)
		{
			return Packet8l(_mm512_sub_epi64(__m512i(a), __m512i(b)));
		}

		template<>
		EIGEN_STRONG_INLINE Packet8l pcmpeq64<Packet8l>(const Packet8l& a, const Packet8l& b)
		{
			__mmask8 mask = _mm512_cmpeq_epi64_mask(__m512i(a), __m512i(b));
			return Packet8l(_mm512_movm_epi64(mask));
		}

		// Combine low 32-bits of each 64-bit element from two Packet8l into one Packet16i
		EIGEN_STRONG_INLINE Packet16i combine_low32(const Packet8l& a, const Packet8l& b)
		{
			// Use vpmovqd to compress 64-bit to 32-bit (takes lower 32 bits)
			__m256i lo_a = _mm512_cvtepi64_epi32(__m512i(a));
			__m256i lo_b = _mm512_cvtepi64_epi32(__m512i(b));
			return _mm512_inserti64x4(_mm512_castsi256_si512(lo_a), lo_b, 1);
		}

		// Note: pcast64 for Packet8l is defined later after int64_to_double_avx512/double_to_int64_avx512
#endif

		template<>
		EIGEN_STRONG_INLINE Packet16i pseti64<Packet16i>(uint64_t a)
		{
			return _mm512_set1_epi64(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16i padd64<Packet16i>(const Packet16i& a, const Packet16i& b)
		{
			return _mm512_add_epi64(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16i psub64<Packet16i>(const Packet16i& a, const Packet16i& b)
		{
			return _mm512_sub_epi64(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16i pcmpeq<Packet16i>(const Packet16i& a, const Packet16i& b)
		{
			return pcmp_eq(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16f pcmpeq<Packet16f>(const Packet16f& a, const Packet16f& b)
		{
			return pcmp_eq(a, b);
		}

		// Eigen 5.x already provides pnegate<Packet16i>
#ifndef EIGENRAND_EIGEN_50_MODE
		template<>
		EIGEN_STRONG_INLINE Packet16i pnegate<Packet16i>(const Packet16i& a)
		{
			return _mm512_sub_epi32(pset1<Packet16i>(0), a);
		}
#endif

		template<>
		struct BitShifter<Packet16i>
		{
			template<int b>
			EIGEN_STRONG_INLINE Packet16i sll(const Packet16i& a)
			{
				return _mm512_slli_epi32(a, b);
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet16i srl(const Packet16i& a, int _b = b)
			{
				if (b >= 0)
				{
					return _mm512_srli_epi32(a, b);
				}
				else
				{
					return _mm512_srli_epi32(a, _b);
				}
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet16i sll64(const Packet16i& a)
			{
				return _mm512_slli_epi64(a, b);
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet16i srl64(const Packet16i& a)
			{
				return _mm512_srli_epi64(a, b);
			}
		};

		template<> EIGEN_STRONG_INLINE bool predux_all(const Packet16i& x)
		{
			return _mm512_movepi32_mask(x) == 0xFFFF;
		}

		template<> EIGEN_STRONG_INLINE bool predux_all(const Packet16f& x)
		{
			return predux_all(_mm512_castps_si512(x));
		}

		template<>
		EIGEN_STRONG_INLINE Packet16i pcmplt<Packet16i>(const Packet16i& a, const Packet16i& b)
		{
			__mmask16 mask = _mm512_cmp_epi32_mask(a, b, _MM_CMPINT_LT);
			return _mm512_movm_epi32(mask);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16f pcmplt<Packet16f>(const Packet16f& a, const Packet16f& b)
		{
			return pcmp_lt(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16f pcmple<Packet16f>(const Packet16f& a, const Packet16f& b)
		{
			return pcmp_le(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8d pcmplt<Packet8d>(const Packet8d& a, const Packet8d& b)
		{
			return pcmp_lt(a, b);
		}
		template<>
		EIGEN_STRONG_INLINE Packet8d pcmple<Packet8d>(const Packet8d& a, const Packet8d& b)
		{
			return pcmp_le(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16f pblendv(const Packet16i& ifPacket, const Packet16f& thenPacket, const Packet16f& elsePacket)
		{
			__mmask16 mask = _mm512_movepi32_mask(ifPacket);
			return _mm512_mask_blend_ps(mask, elsePacket, thenPacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16f pblendv(const Packet16f& ifPacket, const Packet16f& thenPacket, const Packet16f& elsePacket)
		{
			return pblendv(_mm512_castps_si512(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16i pblendv(const Packet16i& ifPacket, const Packet16i& thenPacket, const Packet16i& elsePacket)
		{
			__mmask16 mask = _mm512_movepi32_mask(ifPacket);
			return _mm512_mask_blend_epi32(mask, elsePacket, thenPacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8d pblendv(const Packet16i& ifPacket, const Packet8d& thenPacket, const Packet8d& elsePacket)
		{
			__mmask8 mask = _mm512_movepi64_mask(ifPacket);
			return _mm512_mask_blend_pd(mask, elsePacket, thenPacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8d pblendv(const Packet8d& ifPacket, const Packet8d& thenPacket, const Packet8d& elsePacket)
		{
			return pblendv(_mm512_castpd_si512(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16i pgather<Packet16i>(const int* addr, const Packet16i& index)
		{
			return _mm512_i32gather_epi32(index, addr, 4);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16f pgather<Packet16i>(const float* addr, const Packet16i& index)
		{
			return _mm512_i32gather_ps(index, addr, 4);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8d pgather<Packet16i>(const double* addr, const Packet16i& index, bool upperhalf)
		{
			return _mm512_i32gather_pd(_mm512_castsi512_si256(index), addr, 8);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16f ptruncate<Packet16f>(const Packet16f& a)
		{
			return _mm512_roundscale_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8d ptruncate<Packet8d>(const Packet8d& a)
		{
			return _mm512_roundscale_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
		}

		template<>
		EIGEN_STRONG_INLINE Packet16i pcmpeq64<Packet16i>(const Packet16i& a, const Packet16i& b)
		{
			__mmask8 mask = _mm512_cmp_epi64_mask(a, b, _MM_CMPINT_EQ);
			return _mm512_movm_epi64(mask);
		}

		EIGEN_STRONG_INLINE __m512d int64_to_double_avx512(__m512i x) {
			x = padd64(x, _mm512_castpd_si512(_mm512_set1_pd(0x0018000000000000)));
			return _mm512_sub_pd(_mm512_castsi512_pd(x), _mm512_set1_pd(0x0018000000000000));
		}

		EIGEN_STRONG_INLINE __m512i double_to_int64_avx512(__m512d x) {
			x = _mm512_add_pd(_mm512_floor_pd(x), _mm512_set1_pd(0x0018000000000000));
			return psub64(
				_mm512_castpd_si512(x),
				_mm512_castpd_si512(_mm512_set1_pd(0x0018000000000000))
			);
		}
		template<>
		EIGEN_STRONG_INLINE Packet16i pcast64<Packet8d, Packet16i>(const Packet8d& a)
		{
			return double_to_int64_avx512(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8d pcast64<Packet16i, Packet8d>(const Packet16i& a)
		{
			return int64_to_double_avx512(a);
		}

#ifdef EIGENRAND_EIGEN_50_MODE
		template<>
		EIGEN_STRONG_INLINE Packet8d pcast64<Packet8l, Packet8d>(const Packet8l& a)
		{
			return int64_to_double_avx512(Packet16i(__m512i(a)));
		}

		template<>
		EIGEN_STRONG_INLINE Packet8l pcast64<Packet8d, Packet8l>(const Packet8d& a)
		{
			return Packet8l(__m512i(double_to_int64_avx512(__m512d(a))));
		}
#endif

		// pmovemask for AVX512 packet types
		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet16f>(const Packet16f& a)
		{
			return _mm512_movepi32_mask(_mm512_castps_si512(__m512(a)));
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet8d>(const Packet8d& a)
		{
			return _mm512_movepi64_mask(_mm512_castpd_si512(__m512d(a)));
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet16i>(const Packet16i& a)
		{
			return _mm512_movepi32_mask(__m512i(a));
		}

#ifdef EIGENRAND_EIGEN_50_MODE
		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet8l>(const Packet8l& a)
		{
			return _mm512_movepi64_mask(__m512i(a));
		}
#endif

#ifndef EIGENRAND_EIGEN_50_MODE
		template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
			Packet8d psin<Packet8d>(const Packet8d& x)
		{
			return _psin(x);
		}
#endif
	}
}

#endif
