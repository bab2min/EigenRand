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

#ifndef EIGENRAND_MORE_PACKET_MATH_AVX_H
#define EIGENRAND_MORE_PACKET_MATH_AVX_H

#include <immintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<>
		struct IsIntPacket<Packet8i> : std::true_type {};

		template<>
		struct HalfPacket<Packet8i>
		{
			using type = Packet4i;
		};

		template<>
		struct HalfPacket<Packet8f>
		{
			using type = Packet4f;
		};

		template<>
		struct IsFloatPacket<Packet8f> : std::true_type {};

		template<>
		struct IsDoublePacket<Packet4d> : std::true_type {};

		template<>
		struct reinterpreter<Packet8i>
		{
			EIGEN_STRONG_INLINE Packet8f to_float(const Packet8i& x)
			{
				return _mm256_castsi256_ps(x);
			}

			EIGEN_STRONG_INLINE Packet4d to_double(const Packet8i& x)
			{
				return _mm256_castsi256_pd(x);
			}

			EIGEN_STRONG_INLINE Packet8i to_int(const Packet8i& x)
			{
				return x;
			}
		};

		template<>
		struct reinterpreter<Packet8f>
		{
			EIGEN_STRONG_INLINE Packet8f to_float(const Packet8f& x)
			{
				return x;
			}

			EIGEN_STRONG_INLINE Packet4d to_double(const Packet8f& x)
			{
				return _mm256_castps_pd(x);
			}

			EIGEN_STRONG_INLINE Packet8i to_int(const Packet8f& x)
			{
				return _mm256_castps_si256(x);
			}
		};

		template<>
		struct reinterpreter<Packet4d>
		{
			EIGEN_STRONG_INLINE Packet8f to_float(const Packet4d& x)
			{
				return _mm256_castpd_ps(x);
			}

			EIGEN_STRONG_INLINE Packet4d to_double(const Packet4d& x)
			{
				return x;
			}

			EIGEN_STRONG_INLINE Packet8i to_int(const Packet4d& x)
			{
				return _mm256_castpd_si256(x);
			}
		};

		template<>
		EIGEN_STRONG_INLINE void split_two<Packet8i>(const Packet8i& x, Packet4i& a, Packet4i& b)
		{
			a = _mm256_extractf128_si256(x, 0);
			b = _mm256_extractf128_si256(x, 1);
		}

		EIGEN_STRONG_INLINE Packet8i combine_two(const Packet4i& a, const Packet4i& b)
		{
			return _mm256_insertf128_si256(_mm256_castsi128_si256(a), b, 1);
		}

		template<>
		EIGEN_STRONG_INLINE void split_two<Packet8f>(const Packet8f& x, Packet4f& a, Packet4f& b)
		{
			a = _mm256_extractf128_ps(x, 0);
			b = _mm256_extractf128_ps(x, 1);
		}

		EIGEN_STRONG_INLINE Packet8f combine_two(const Packet4f& a, const Packet4f& b)
		{
			return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
		}


		EIGEN_STRONG_INLINE Packet4i combine_low32(const Packet8i& a)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(a, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
#else
			auto sc = _mm256_permutevar_ps(_mm256_castsi256_ps(a), _mm256_setr_epi32(0, 2, 1, 3, 1, 3, 0, 2));
			return _mm_castps_si128(_mm_blend_ps(_mm256_extractf128_ps(sc, 0), _mm256_extractf128_ps(sc, 1), 0b1100));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pseti64<Packet8i>(uint64_t a)
		{
			return _mm256_set1_epi64x(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i padd64<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_add_epi64(a, b);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_add_epi64(a1, b1), (Packet4i)_mm_add_epi64(a2, b2));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i psub64<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_sub_epi64(a, b);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_sub_epi64(a1, b1), (Packet4i)_mm_sub_epi64(a2, b2));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pcmpeq<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_cmpeq_epi32(a, b);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_cmpeq_epi32(a1, b1), (Packet4i)_mm_cmpeq_epi32(a2, b2));
#endif
		}

		template<>
		struct BitShifter<Packet8i>
		{
			template<int b>
			EIGEN_STRONG_INLINE Packet8i sll(const Packet8i& a)
			{
#ifdef EIGEN_VECTORIZE_AVX2
				return _mm256_slli_epi32(a, b);
#else
				Packet4i a1, a2;
				split_two(a, a1, a2);
				return combine_two((Packet4i)_mm_slli_epi32(a1, b), (Packet4i)_mm_slli_epi32(a2, b));
#endif
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet8i srl(const Packet8i& a, int _b = b)
			{
#ifdef EIGEN_VECTORIZE_AVX2
				if (b >= 0)
				{
					return _mm256_srli_epi32(a, b);
				}
				else
				{
					return _mm256_srli_epi32(a, _b);
				}
#else
				Packet4i a1, a2;
				split_two(a, a1, a2);
				if (b >= 0)
				{
					return combine_two((Packet4i)_mm_srli_epi32(a1, b), (Packet4i)_mm_srli_epi32(a2, b));
				}
				else
				{
					return combine_two((Packet4i)_mm_srli_epi32(a1, _b), (Packet4i)_mm_srli_epi32(a2, _b));
				}
#endif
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet8i sll64(const Packet8i& a)
			{
#ifdef EIGEN_VECTORIZE_AVX2
				return _mm256_slli_epi64(a, b);
#else
				Packet4i a1, a2;
				split_two(a, a1, a2);
				return combine_two((Packet4i)_mm_slli_epi64(a1, b), (Packet4i)_mm_slli_epi64(a2, b));
#endif
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet8i srl64(const Packet8i& a)
			{
#ifdef EIGEN_VECTORIZE_AVX2
				return _mm256_srli_epi64(a, b);
#else
				Packet4i a1, a2;
				split_two(a, a1, a2);
				return combine_two((Packet4i)_mm_srli_epi64(a1, b), (Packet4i)_mm_srli_epi64(a2, b));
#endif
			}
		};
#ifdef EIGENRAND_EIGEN_33_MODE
		template<> EIGEN_STRONG_INLINE Packet8i padd<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
	#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_add_epi32(a, b);
	#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_add_epi32(a1, b1), (Packet4i)_mm_add_epi32(a2, b2));
	#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i psub<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
	#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_sub_epi32(a, b);
	#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_sub_epi32(a1, b1), (Packet4i)_mm_sub_epi32(a2, b2));
	#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i pand<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
	#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_and_si256(a, b);
	#else
			return reinterpret_to_int((Packet8f)_mm256_and_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
	#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i pandnot<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
	#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_andnot_si256(a, b);
	#else
			return reinterpret_to_int((Packet8f)_mm256_andnot_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
	#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i por<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
	#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_or_si256(a, b);
	#else
			return reinterpret_to_int((Packet8f)_mm256_or_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
	#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i pxor<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
	#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_xor_si256(a, b);
	#else
			return reinterpret_to_int((Packet8f)_mm256_xor_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
	#endif
		}
#endif
		template<>
		EIGEN_STRONG_INLINE Packet8i pcmplt<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_cmpgt_epi32(b, a);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_cmpgt_epi32(b1, a1), (Packet4i)_mm_cmpgt_epi32(b2, a2));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pcmplt64<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_cmpgt_epi64(b, a);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_cmpgt_epi64(b1, a1), (Packet4i)_mm_cmpgt_epi64(b2, a2));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8f pcmplt<Packet8f>(const Packet8f& a, const Packet8f& b)
		{
			return _mm256_cmp_ps(a, b, _CMP_LT_OQ);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8f pcmple<Packet8f>(const Packet8f& a, const Packet8f& b)
		{
			return _mm256_cmp_ps(a, b, _CMP_LE_OQ);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d pcmplt<Packet4d>(const Packet4d& a, const Packet4d& b)
		{
			return _mm256_cmp_pd(a, b, _CMP_LT_OQ);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d pcmple<Packet4d>(const Packet4d& a, const Packet4d& b)
		{
			return _mm256_cmp_pd(a, b, _CMP_LE_OQ);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8f pblendv(const Packet8f& ifPacket, const Packet8f& thenPacket, const Packet8f& elsePacket)
		{
			return _mm256_blendv_ps(elsePacket, thenPacket, ifPacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8f pblendv(const Packet8i& ifPacket, const Packet8f& thenPacket, const Packet8f& elsePacket)
		{
			return pblendv(_mm256_castsi256_ps(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pblendv(const Packet8i& ifPacket, const Packet8i& thenPacket, const Packet8i& elsePacket)
		{
			return _mm256_castps_si256(_mm256_blendv_ps(
				_mm256_castsi256_ps(elsePacket),
				_mm256_castsi256_ps(thenPacket),
				_mm256_castsi256_ps(ifPacket)
			));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d pblendv(const Packet4d& ifPacket, const Packet4d& thenPacket, const Packet4d& elsePacket)
		{
			return _mm256_blendv_pd(elsePacket, thenPacket, ifPacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d pblendv(const Packet8i& ifPacket, const Packet4d& thenPacket, const Packet4d& elsePacket)
		{
			return pblendv(_mm256_castsi256_pd(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pgather<Packet8i>(const int* addr, const Packet8i& index)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_i32gather_epi32(addr, index, 4);
#else
			uint32_t u[8];
			_mm256_storeu_si256((Packet8i*)u, index);
			return _mm256_setr_epi32(addr[u[0]], addr[u[1]], addr[u[2]], addr[u[3]],
				addr[u[4]], addr[u[5]], addr[u[6]], addr[u[7]]);
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8f pgather<Packet8i>(const float* addr, const Packet8i& index)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_i32gather_ps(addr, index, 4);
#else
			uint32_t u[8];
			_mm256_storeu_si256((Packet8i*)u, index);
			return _mm256_setr_ps(addr[u[0]], addr[u[1]], addr[u[2]], addr[u[3]],
				addr[u[4]], addr[u[5]], addr[u[6]], addr[u[7]]);
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d pgather<Packet8i>(const double* addr, const Packet8i& index, bool upperhalf)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_i32gather_pd(addr, _mm256_castsi256_si128(index), 8);
#else
			uint32_t u[8];
			_mm256_storeu_si256((Packet8i*)u, index);
			if (upperhalf)
			{
				return _mm256_setr_pd(addr[u[4]], addr[u[5]], addr[u[6]], addr[u[7]]);
			}
			else
			{
				return _mm256_setr_pd(addr[u[0]], addr[u[1]], addr[u[2]], addr[u[3]]);
			}
#endif
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet8f>(const Packet8f& a)
		{
			return _mm256_movemask_ps(a);
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet4d>(const Packet4d& a)
		{
			return _mm256_movemask_pd(a);
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet8i>(const Packet8i& a)
		{
			return pmovemask(_mm256_castsi256_ps(a));
		}

		template<>
		EIGEN_STRONG_INLINE Packet8f ptruncate<Packet8f>(const Packet8f& a)
		{
			return _mm256_round_ps(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d ptruncate<Packet4d>(const Packet4d& a)
		{
			return _mm256_round_pd(a, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pcmpeq64<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_cmpeq_epi64(a, b);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two((Packet4i)_mm_cmpeq_epi64(a1, b1), (Packet4i)_mm_cmpeq_epi64(a2, b2));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pmuluadd64<Packet8i>(const Packet8i& a, uint64_t b, uint64_t c)
		{
			uint64_t u[4];
			_mm256_storeu_si256((__m256i*)u, a);
			u[0] = u[0] * b + c;
			u[1] = u[1] * b + c;
			u[2] = u[2] * b + c;
			u[3] = u[3] * b + c;
			return _mm256_loadu_si256((__m256i*)u);
		}

		EIGEN_STRONG_INLINE __m256d uint64_to_double(__m256i x) {
			auto y = _mm256_or_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0010000000000000));
			return _mm256_sub_pd(y, _mm256_set1_pd(0x0010000000000000));
		}

		EIGEN_STRONG_INLINE __m256d int64_to_double(__m256i x) {
			x = padd64(x, _mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000)));
			return _mm256_sub_pd(_mm256_castsi256_pd(x), _mm256_set1_pd(0x0018000000000000));
		}

		EIGEN_STRONG_INLINE __m256i double_to_int64(__m256d x) {
			x = _mm256_add_pd(_mm256_floor_pd(x), _mm256_set1_pd(0x0018000000000000));
			return psub64(
				_mm256_castpd_si256(x),
				_mm256_castpd_si256(_mm256_set1_pd(0x0018000000000000))
			);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pcast64<Packet4d, Packet8i>(const Packet4d& a)
		{
			return double_to_int64(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4d pcast64<Packet8i, Packet4d>(const Packet8i& a)
		{
			return int64_to_double(a);
		}

		template<> EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED
			Packet4d psin<Packet4d>(const Packet4d& x)
		{
			return _psin(x);
		}

	#ifdef EIGENRAND_EIGEN_33_MODE
		template <>
		EIGEN_DEFINE_FUNCTION_ALLOWING_MULTIPLE_DEFINITIONS EIGEN_UNUSED Packet4d
			plog<Packet4d>(const Packet4d& _x) {
			Packet4d x = _x;
			_EIGEN_DECLARE_CONST_Packet4d(1, 1.0);
			_EIGEN_DECLARE_CONST_Packet4d(half, 0.5);

			auto inv_mant_mask = _mm256_castsi256_pd(pseti64<Packet8i>(~0x7ff0000000000000));
			auto min_norm_pos = _mm256_castsi256_pd(pseti64<Packet8i>(0x10000000000000));
			auto minus_inf = _mm256_castsi256_pd(pseti64<Packet8i>(0xfff0000000000000));

			// Polynomial coefficients.
			_EIGEN_DECLARE_CONST_Packet4d(cephes_SQRTHF, 0.707106781186547524);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p0, 7.0376836292E-2);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p1, -1.1514610310E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p2, 1.1676998740E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p3, -1.2420140846E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p4, +1.4249322787E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p5, -1.6668057665E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p6, +2.0000714765E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p7, -2.4999993993E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_p8, +3.3333331174E-1);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_q1, -2.12194440e-4);
			_EIGEN_DECLARE_CONST_Packet4d(cephes_log_q2, 0.693359375);

			Packet4d invalid_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_NGE_UQ); // not greater equal is true if x is NaN
			Packet4d iszero_mask = _mm256_cmp_pd(x, _mm256_setzero_pd(), _CMP_EQ_OQ);

			// Truncate input values to the minimum positive normal.
			x = pmax(x, min_norm_pos);

			Packet4d emm0 = uint64_to_double(psrl64<52>(_mm256_castpd_si256(x)));
			Packet4d e = psub(emm0, pset1<Packet4d>(1022));

			// Set the exponents to -1, i.e. x are in the range [0.5,1).
			x = _mm256_and_pd(x, inv_mant_mask);
			x = _mm256_or_pd(x, p4d_half);

			// part2: Shift the inputs from the range [0.5,1) to [sqrt(1/2),sqrt(2))
			// and shift by -1. The values are then centered around 0, which improves
			// the stability of the polynomial evaluation.
			//   if( x < SQRTHF ) {
			//     e -= 1;
			//     x = x + x - 1.0;
			//   } else { x = x - 1.0; }
			Packet4d mask = _mm256_cmp_pd(x, p4d_cephes_SQRTHF, _CMP_LT_OQ);
			Packet4d tmp = _mm256_and_pd(x, mask);
			x = psub(x, p4d_1);
			e = psub(e, _mm256_and_pd(p4d_1, mask));
			x = padd(x, tmp);

			Packet4d x2 = pmul(x, x);
			Packet4d x3 = pmul(x2, x);

			// Evaluate the polynomial approximant of degree 8 in three parts, probably
			// to improve instruction-level parallelism.
			Packet4d y, y1, y2;
			y = pmadd(p4d_cephes_log_p0, x, p4d_cephes_log_p1);
			y1 = pmadd(p4d_cephes_log_p3, x, p4d_cephes_log_p4);
			y2 = pmadd(p4d_cephes_log_p6, x, p4d_cephes_log_p7);
			y = pmadd(y, x, p4d_cephes_log_p2);
			y1 = pmadd(y1, x, p4d_cephes_log_p5);
			y2 = pmadd(y2, x, p4d_cephes_log_p8);
			y = pmadd(y, x3, y1);
			y = pmadd(y, x3, y2);
			y = pmul(y, x3);

			// Add the logarithm of the exponent back to the result of the interpolation.
			y1 = pmul(e, p4d_cephes_log_q1);
			tmp = pmul(x2, p4d_half);
			y = padd(y, y1);
			x = psub(x, tmp);
			y2 = pmul(e, p4d_cephes_log_q2);
			x = padd(x, y);
			x = padd(x, y2);

			// Filter out invalid inputs, i.e. negative arg will be NAN, 0 will be -INF.
			return pblendv(iszero_mask, minus_inf, _mm256_or_pd(x, invalid_mask));
		}
	#endif

	#if !(EIGEN_VERSION_AT_LEAST(3,3,5))
		template<> EIGEN_STRONG_INLINE Packet4f pcast<Packet4i, Packet4f>(const Packet4i& a) {
			return _mm_cvtepi32_ps(a);
		}

		template<> EIGEN_STRONG_INLINE Packet4i pcast<Packet4f, Packet4i>(const Packet4f& a) {
			return _mm_cvttps_epi32(a);
		}
	#endif
	}
}

#endif