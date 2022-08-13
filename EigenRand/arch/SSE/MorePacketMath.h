/**
 * @file MorePacketMath.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.4.1
 * @date 2022-08-13
 *
 * @copyright Copyright (c) 2020-2021
 *
 */

#ifndef EIGENRAND_MORE_PACKET_MATH_SSE_H
#define EIGENRAND_MORE_PACKET_MATH_SSE_H

#include <xmmintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<>
		struct IsIntPacket<Packet4i> : std::true_type {};

		template<>
		struct IsFloatPacket<Packet4f> : std::true_type {};

		template<>
		struct IsDoublePacket<Packet2d> : std::true_type {};

		template<>
		struct HalfPacket<Packet4i>
		{
			using type = uint64_t;
		};

#ifdef EIGEN_VECTORIZE_AVX
#else
		template<>
		struct HalfPacket<Packet4f>
		{
			//using type = Packet2f;
		};
#endif
		template<>
		struct reinterpreter<Packet4i>
		{
			EIGEN_STRONG_INLINE Packet4f to_float(const Packet4i& x)
			{
				return _mm_castsi128_ps(x);
			}

			EIGEN_STRONG_INLINE Packet2d to_double(const Packet4i& x)
			{
				return _mm_castsi128_pd(x);
			}

			EIGEN_STRONG_INLINE Packet4i to_int(const Packet4i& x)
			{
				return x;
			}
		};

		template<>
		struct reinterpreter<Packet4f>
		{
			EIGEN_STRONG_INLINE Packet4f to_float(const Packet4f& x)
			{
				return x;
			}

			EIGEN_STRONG_INLINE Packet2d to_double(const Packet4f& x)
			{
				return _mm_castps_pd(x);
			}

			EIGEN_STRONG_INLINE Packet4i to_int(const Packet4f& x)
			{
				return _mm_castps_si128(x);
			}
		};

		template<>
		struct reinterpreter<Packet2d>
		{
			EIGEN_STRONG_INLINE Packet4f to_float(const Packet2d& x)
			{
				return _mm_castpd_ps(x);
			}

			EIGEN_STRONG_INLINE Packet2d to_double(const Packet2d& x)
			{
				return x;
			}

			EIGEN_STRONG_INLINE Packet4i to_int(const Packet2d& x)
			{
				return _mm_castpd_si128(x);
			}
		};

		template<>
		EIGEN_STRONG_INLINE void split_two<Packet4i>(const Packet4i& x, uint64_t& a, uint64_t& b)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			a = _mm_extract_epi64(x, 0);
			b = _mm_extract_epi64(x, 1);
#else
			uint64_t u[2];
			_mm_storeu_si128((__m128i*)u, x);
			a = u[0];
			b = u[1];
#endif
		}

		EIGEN_STRONG_INLINE Packet4i combine_low32(const Packet4i& a, const Packet4i& b)
		{
			auto sa = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 2, 0));
			auto sb = _mm_shuffle_epi32(b, _MM_SHUFFLE(2, 0, 3, 1));
			sa = _mm_and_si128(sa, _mm_setr_epi32(-1, -1, 0, 0));
			sb = _mm_and_si128(sb, _mm_setr_epi32(0, 0, -1, -1));
			return _mm_or_si128(sa, sb);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pseti64<Packet4i>(uint64_t a)
		{
			return _mm_set1_epi64x(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i padd64<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return _mm_add_epi64(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i psub64<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return _mm_sub_epi64(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmpeq<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return _mm_cmpeq_epi32(a, b);
		}

		template<>
		struct BitShifter<Packet4i>
		{
			template<int b>
			EIGEN_STRONG_INLINE Packet4i sll(const Packet4i& a)
			{
				return _mm_slli_epi32(a, b);
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet4i srl(const Packet4i& a, int _b = b)
			{
				if (b >= 0)
				{
					return _mm_srli_epi32(a, b);
				}
				else
				{
					return _mm_srli_epi32(a, _b);
				}
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet4i sll64(const Packet4i& a)
			{
				return _mm_slli_epi64(a, b);
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet4i srl64(const Packet4i& a)
			{
				return _mm_srli_epi64(a, b);
			}
		};

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmplt<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return _mm_cmplt_epi32(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmplt64<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
#ifdef EIGEN_VECTORIZE_SSE4_2
			return _mm_cmpgt_epi64(b, a);
#else
			int64_t u[2], v[2];
			_mm_storeu_si128((__m128i*)u, a);
			_mm_storeu_si128((__m128i*)v, b);
			return _mm_set_epi64x(u[1] < v[1] ? -1 : 0, u[0] < v[0] ? -1 : 0);
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pcmplt<Packet4f>(const Packet4f& a, const Packet4f& b)
		{
			return _mm_cmplt_ps(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pcmple<Packet4f>(const Packet4f& a, const Packet4f& b)
		{
			return _mm_cmple_ps(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet2d pcmplt<Packet2d>(const Packet2d& a, const Packet2d& b)
		{
			return _mm_cmplt_pd(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet2d pcmple<Packet2d>(const Packet2d& a, const Packet2d& b)
		{
			return _mm_cmple_pd(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pblendv(const Packet4f& ifPacket, const Packet4f& thenPacket, const Packet4f& elsePacket)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			return _mm_blendv_ps(elsePacket, thenPacket, ifPacket);
#else
			return _mm_or_ps(_mm_and_ps(ifPacket, thenPacket), _mm_andnot_ps(ifPacket, elsePacket));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pblendv(const Packet4i& ifPacket, const Packet4f& thenPacket, const Packet4f& elsePacket)
		{
			return pblendv(_mm_castsi128_ps(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pblendv(const Packet4i& ifPacket, const Packet4i& thenPacket, const Packet4i& elsePacket)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			return _mm_castps_si128(_mm_blendv_ps(_mm_castsi128_ps(elsePacket), _mm_castsi128_ps(thenPacket), _mm_castsi128_ps(ifPacket)));
#else
			return _mm_or_si128(_mm_and_si128(ifPacket, thenPacket), _mm_andnot_si128(ifPacket, elsePacket));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet2d pblendv(const Packet2d& ifPacket, const Packet2d& thenPacket, const Packet2d& elsePacket)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			return _mm_blendv_pd(elsePacket, thenPacket, ifPacket);
#else
			return _mm_or_pd(_mm_and_pd(ifPacket, thenPacket), _mm_andnot_pd(ifPacket, elsePacket));
#endif
		}


		template<>
		EIGEN_STRONG_INLINE Packet2d pblendv(const Packet4i& ifPacket, const Packet2d& thenPacket, const Packet2d& elsePacket)
		{
			return pblendv(_mm_castsi128_pd(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pgather<Packet4i>(const int* addr, const Packet4i& index)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm_i32gather_epi32(addr, index, 4);
#else
			uint32_t u[4];
			_mm_storeu_si128((__m128i*)u, index);
			return _mm_setr_epi32(addr[u[0]], addr[u[1]], addr[u[2]], addr[u[3]]);
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pgather<Packet4i>(const float* addr, const Packet4i& index)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm_i32gather_ps(addr, index, 4);
#else
			uint32_t u[4];
			_mm_storeu_si128((__m128i*)u, index);
			return _mm_setr_ps(addr[u[0]], addr[u[1]], addr[u[2]], addr[u[3]]);
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet2d pgather<Packet4i>(const double* addr, const Packet4i& index, bool upperhalf)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm_i32gather_pd(addr, index, 8);
#else
			uint32_t u[4];
			_mm_storeu_si128((__m128i*)u, index);
			if (upperhalf)
			{
				return _mm_setr_pd(addr[u[2]], addr[u[3]]);
			}
			else
			{
				return _mm_setr_pd(addr[u[0]], addr[u[1]]);
			}
#endif
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet4f>(const Packet4f& a)
		{
			return _mm_movemask_ps(a);
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet2d>(const Packet2d& a)
		{
			return _mm_movemask_pd(a);
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet4i>(const Packet4i& a)
		{
			return pmovemask((Packet4f)_mm_castsi128_ps(a));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f ptruncate<Packet4f>(const Packet4f& a)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			return _mm_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
			auto round = _MM_GET_ROUNDING_MODE();
			_MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
			auto ret = _mm_cvtepi32_ps(_mm_cvtps_epi32(a));
			_MM_SET_ROUNDING_MODE(round);
			return ret;
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet2d ptruncate<Packet2d>(const Packet2d& a)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			return _mm_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
#else
			auto round = _MM_GET_ROUNDING_MODE();
			_MM_SET_ROUNDING_MODE(_MM_ROUND_TOWARD_ZERO);
			auto ret = _mm_cvtepi32_pd(_mm_cvtpd_epi32(a));
			_MM_SET_ROUNDING_MODE(round);
			return ret;
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmpeq64<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			return _mm_cmpeq_epi64(a, b);
#else
			Packet4i c = _mm_cmpeq_epi32(a, b);
			return pand(c, (Packet4i)_mm_shuffle_epi32(c, _MM_SHUFFLE(2, 3, 0, 1)));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pmuluadd64<Packet4i>(const Packet4i& a, uint64_t b, uint64_t c)
		{
			uint64_t u[2];
			_mm_storeu_si128((__m128i*)u, a);
			u[0] = u[0] * b + c;
			u[1] = u[1] * b + c;
			return _mm_loadu_si128((__m128i*)u);
		}

		EIGEN_STRONG_INLINE __m128d uint64_to_double(__m128i x) {
			x = _mm_or_si128(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
			return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0010000000000000));
		}

		EIGEN_STRONG_INLINE __m128d int64_to_double(__m128i x) {
			x = _mm_add_epi64(x, _mm_castpd_si128(_mm_set1_pd(0x0018000000000000)));
			return _mm_sub_pd(_mm_castsi128_pd(x), _mm_set1_pd(0x0018000000000000));
		}

		EIGEN_STRONG_INLINE __m128i double_to_int64(__m128d x) {
			int _mm_rounding = _MM_GET_ROUNDING_MODE();
			_MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
			x = _mm_add_pd(x, _mm_set1_pd(0x0018000000000000));
			_MM_SET_ROUNDING_MODE(_mm_rounding);
			return _mm_sub_epi64(
				_mm_castpd_si128(x),
				_mm_castpd_si128(_mm_set1_pd(0x0018000000000000))
			);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pcast64<Packet2d, Packet4i>(const Packet2d& a)
		{
			return double_to_int64(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet2d pcast64<Packet4i, Packet2d>(const Packet4i& a)
		{
			return int64_to_double(a);
		}

		template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
			Packet2d psin<Packet2d>(const Packet2d& x)
		{
			return _psin(x);
		}
#ifdef EIGENRAND_EIGEN_33_MODE
		template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
			Packet2d plog<Packet2d>(const Packet2d& _x)
		{
			Packet2d x = _x;
			_EIGEN_DECLARE_CONST_Packet2d(1, 1.0f);
			_EIGEN_DECLARE_CONST_Packet2d(half, 0.5f);
			_EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);

			auto inv_mant_mask = _mm_castsi128_pd(pseti64<Packet4i>(~0x7ff0000000000000));
			auto min_norm_pos = _mm_castsi128_pd(pseti64<Packet4i>(0x10000000000000));
			auto minus_inf = _mm_castsi128_pd(pseti64<Packet4i>(0xfff0000000000000));

			/* natural logarithm computed for 4 simultaneous float
			  return NaN for x <= 0
			*/
			_EIGEN_DECLARE_CONST_Packet2d(cephes_SQRTHF, 0.707106781186547524);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p0, 7.0376836292E-2);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p1, -1.1514610310E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p2, 1.1676998740E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p3, -1.2420140846E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p4, +1.4249322787E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p5, -1.6668057665E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p6, +2.0000714765E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p7, -2.4999993993E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_p8, +3.3333331174E-1);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_q1, -2.12194440e-4);
			_EIGEN_DECLARE_CONST_Packet2d(cephes_log_q2, 0.693359375);


			Packet4i emm0;

			Packet2d invalid_mask = _mm_cmpnge_pd(x, _mm_setzero_pd()); // not greater equal is true if x is NaN
			Packet2d iszero_mask = _mm_cmpeq_pd(x, _mm_setzero_pd());

			x = pmax(x, min_norm_pos);  /* cut off denormalized stuff */
			emm0 = _mm_srli_epi64(_mm_castpd_si128(x), 52);

			/* keep only the fractional part */
			x = _mm_and_pd(x, inv_mant_mask);
			x = _mm_or_pd(x, p2d_half);

			Packet2d e = _mm_sub_pd(uint64_to_double(emm0), pset1<Packet2d>(1022));

			/* part2:
			   if( x < SQRTHF ) {
				 e -= 1;
				 x = x + x - 1.0;
			   } else { x = x - 1.0; }
			*/
			Packet2d mask = _mm_cmplt_pd(x, p2d_cephes_SQRTHF);
			Packet2d tmp = pand(x, mask);
			x = psub(x, p2d_1);
			e = psub(e, pand(p2d_1, mask));
			x = padd(x, tmp);

			Packet2d x2 = pmul(x, x);
			Packet2d x3 = pmul(x2, x);

			Packet2d y, y1, y2;
			y = pmadd(p2d_cephes_log_p0, x, p2d_cephes_log_p1);
			y1 = pmadd(p2d_cephes_log_p3, x, p2d_cephes_log_p4);
			y2 = pmadd(p2d_cephes_log_p6, x, p2d_cephes_log_p7);
			y = pmadd(y, x, p2d_cephes_log_p2);
			y1 = pmadd(y1, x, p2d_cephes_log_p5);
			y2 = pmadd(y2, x, p2d_cephes_log_p8);
			y = pmadd(y, x3, y1);
			y = pmadd(y, x3, y2);
			y = pmul(y, x3);

			y1 = pmul(e, p2d_cephes_log_q1);
			tmp = pmul(x2, p2d_half);
			y = padd(y, y1);
			x = psub(x, tmp);
			y2 = pmul(e, p2d_cephes_log_q2);
			x = padd(x, y);
			x = padd(x, y2);
			// negative arg will be NAN, 0 will be -INF
			return pblendv(iszero_mask, minus_inf, _mm_or_pd(x, invalid_mask));
		}
	#endif
	}
}

#endif
