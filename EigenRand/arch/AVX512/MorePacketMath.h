/**
 * @file MorePacketMath.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.5.1
 * @date 2024-09-08
 *
 * @copyright Copyright (c) 2020-2024
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
		struct reinterpreter<Packet16i>
		{
			EIGEN_STRONG_INLINE Packet16f to_float(const Packet16i& x)
			{
				return _mm512_castsi512_ps(x);
			}

			EIGEN_STRONG_INLINE Packet8d to_double(const Packet16i& x)
			{
				return _mm512_castsi512_pd(x);
			}

			EIGEN_STRONG_INLINE Packet16i to_int(const Packet16i& x)
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
				return _mm512_castps_pd(x);
			}

			EIGEN_STRONG_INLINE Packet16i to_int(const Packet16f& x)
			{
				return _mm512_castps_si512(x);
			}
		};

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

			EIGEN_STRONG_INLINE Packet16i to_int(const Packet8d& x)
			{
				return _mm512_castpd_si512(x);
			}
		};

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

		template<>
		EIGEN_STRONG_INLINE Packet16i pnegate<Packet16i>(const Packet16i& a)
		{
			return _mm512_sub_epi32(pset1<Packet16i>(0), a);
		}

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

		template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
			Packet8d psin<Packet8d>(const Packet8d& x)
		{
			return _psin(x);
		}
	}
}

#endif
