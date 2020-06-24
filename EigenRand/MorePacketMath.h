#ifndef EIGENRAND_MORE_PACKET_MATH_H
#define EIGENRAND_MORE_PACKET_MATH_H

#include <Eigen/Dense>

namespace Eigen
{
	namespace internal
	{
		template<typename Packet>
		struct reinterpreter
		{
		};

		template<typename Packet>
		inline auto reinterpret_to_float(const Packet& x)
			-> decltype(reinterpreter<Packet>{}.to_float(x))
		{
			return reinterpreter<Packet>{}.to_float(x);
		}

		template<typename Packet>
		inline auto reinterpret_to_double(const Packet& x)
			-> decltype(reinterpreter<Packet>{}.to_double(x))
		{
			return reinterpreter<Packet>{}.to_double(x);
		}

		template<typename Packet>
		inline auto reinterpret_to_int(const Packet& x)
			-> decltype(reinterpreter<Packet>{}.to_int(x))
		{
			return reinterpreter<Packet>{}.to_int(x);
		}

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pseti64(uint64_t a);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmpeq(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psll(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psll64(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psrl64(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE void psincos(Packet x, Packet &s, Packet &c)
		{
			Packet xmm1, xmm2, xmm3 = pset1<Packet>(0), sign_bit_sin, y;
			using IntPacket = decltype(reinterpret_to_int(x));
			IntPacket emm0, emm2, emm4;

			sign_bit_sin = x;
			/* take the absolute value */
			x = pabs(x);
			/* extract the sign bit (upper one) */
			sign_bit_sin = reinterpret_to_float(
				pand(reinterpret_to_int(sign_bit_sin), pset1<IntPacket>(0x80000000))
			);

			/* scale by 4/Pi */
			y = pmul(x, pset1<Packet>(1.27323954473516));

			/* store the integer part of y in emm2 */
			emm2 = pcast<Packet, IntPacket>(y);

			/* j=(j+1) & (~1) (see the cephes sources) */
			emm2 = padd(emm2, pset1<IntPacket>(1));
			emm2 = pand(emm2, pset1<IntPacket>(~1));
			y = pcast<IntPacket, Packet>(emm2);

			emm4 = emm2;

			/* get the swap sign flag for the sine */
			emm0 = pand(emm2, pset1<IntPacket>(4));
			emm0 = psll(emm0, 29);
			Packet swap_sign_bit_sin = reinterpret_to_float(emm0);

			/* get the polynom selection mask for the sine*/
			emm2 = pand(emm2, pset1<IntPacket>(2));

			emm2 = pcmpeq(emm2, pset1<IntPacket>(0));
			Packet poly_mask = reinterpret_to_float(emm2);

			/* The magic pass: "Extended precision modular arithmetic"
			x = ((x - y * DP1) - y * DP2) - y * DP3; */
			xmm1 = pset1<Packet>(-0.78515625);
			xmm2 = pset1<Packet>(-2.4187564849853515625e-4);
			xmm3 = pset1<Packet>(-3.77489497744594108e-8);
			xmm1 = pmul(y, xmm1);
			xmm2 = pmul(y, xmm2);
			xmm3 = pmul(y, xmm3);
			x = padd(x, xmm1);
			x = padd(x, xmm2);
			x = padd(x, xmm3);

			emm4 = psub(emm4, pset1<IntPacket>(2));
			emm4 = pandnot(emm4, pset1<IntPacket>(4));
			emm4 = psll(emm4, 29);
			Packet sign_bit_cos = reinterpret_to_float(emm4);
			sign_bit_sin = pxor(sign_bit_sin, swap_sign_bit_sin);


			/* Evaluate the first polynom  (0 <= x <= Pi/4) */
			Packet z = pmul(x, x);
			y = pset1<Packet>(2.443315711809948E-005);

			y = pmul(y, z);
			y = padd(y, pset1<Packet>(-1.388731625493765E-003));
			y = pmul(y, z);
			y = padd(y, pset1<Packet>(4.166664568298827E-002));
			y = pmul(y, z);
			y = pmul(y, z);
			Packet tmp = pmul(z, pset1<Packet>(0.5));
			y = psub(y, tmp);
			y = padd(y, pset1<Packet>(1));

			/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

			Packet y2 = pset1<Packet>(-1.9515295891E-4);
			y2 = pmul(y2, z);
			y2 = padd(y2, pset1<Packet>(8.3321608736E-3));
			y2 = pmul(y2, z);
			y2 = padd(y2, pset1<Packet>(-1.6666654611E-1));
			y2 = pmul(y2, z);
			y2 = pmul(y2, x);
			y2 = padd(y2, x);

			/* select the correct result from the two polynoms */
			xmm3 = poly_mask;
			Packet ysin2 = pand(xmm3, y2);
			Packet ysin1 = pandnot(xmm3, y);
			y2 = psub(y2, ysin2);
			y = psub(y, ysin1);

			xmm1 = padd(ysin1, ysin2);
			xmm2 = padd(y, y2);

			/* update the sign */
			s = pxor(xmm1, sign_bit_sin);
			c = pxor(xmm2, sign_bit_cos);
		}

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmplt(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmple(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pblendv(const Packet& ifPacket, const Packet& thenPacket, const Packet& elsePacket);
	}
}

#ifdef EIGEN_VECTORIZE_AVX
#include <immintrin.h>

namespace Eigen
{
	namespace internal
	{
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

		EIGEN_STRONG_INLINE void split_two(const Packet8i& x, Packet4i& a, Packet4i& b)
		{
			a = _mm256_extractf128_si256(x, 0);
			b = _mm256_extractf128_si256(x, 1);
		}

		EIGEN_STRONG_INLINE Packet8i combine_two(const Packet4i& a, const Packet4i& b)
		{
			return _mm256_insertf128_si256(_mm256_castsi128_si256(a), b, 1);
		}

		EIGEN_STRONG_INLINE void split_two(const Packet8f& x, Packet4f& a, Packet4f& b)
		{
			a = _mm256_extractf128_ps(x, 0);
			b = _mm256_extractf128_ps(x, 1);
		}

		EIGEN_STRONG_INLINE Packet8f combine_two(const Packet4f& a, const Packet4f& b)
		{
			return _mm256_insertf128_ps(_mm256_castps128_ps256(a), b, 1);
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pseti64<Packet8i>(uint64_t a)
		{
			return _mm256_set1_epi64x(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pseti64<Packet4i>(uint64_t a)
		{
			return _mm_set1_epi64x(a);
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
			return combine_two(_mm_cmpeq_epi32(a1, b1), _mm_cmpeq_epi32(a2, b2));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i psll<Packet8i>(const Packet8i& a, int b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_slli_epi32(a, b);
#else
			Packet4i a1, a2;
			split_two(a, a1, a2);
			return combine_two(_mm_slli_epi32(a1, b), _mm_slli_epi32(a2, b));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i psll64<Packet8i>(const Packet8i& a, int b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_slli_epi64(a, b);
#else
			Packet4i a1, a2;
			split_two(a, a1, a2);
			return combine_two(_mm_slli_epi64(a1, b), _mm_slli_epi64(a2, b));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i psrl64<Packet8i>(const Packet8i& a, int b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_srli_epi64(a, b);
#else
			Packet4i a1, a2;
			split_two(a, a1, a2);
			return combine_two(_mm_srli_epi64(a1, b), _mm_srli_epi64(a2, b));
#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i padd<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_add_epi32(a, b);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two(_mm_add_epi32(a1, b1), _mm_add_epi32(a2, b2));
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
			return combine_two(_mm_sub_epi32(a1, b1), _mm_sub_epi32(a2, b2));
#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i pand<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_and_si256(a, b);
#else
			return reinterpret_to_int(_mm256_and_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i pandnot<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_andnot_si256(a, b);
#else
			return reinterpret_to_int(_mm256_andnot_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i por<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_or_si256(a, b);
#else
			return reinterpret_to_int(_mm256_or_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
#endif
		}

		template<> EIGEN_STRONG_INLINE Packet8i pxor<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_xor_si256(a, b);
#else
			return reinterpret_to_int(_mm256_xor_ps(reinterpret_to_float(a), reinterpret_to_float(b)));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pcmplt<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
			return _mm256_cmpgt_epi32(b, a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmplt<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return _mm_cmplt_epi32(a, b);
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
		EIGEN_STRONG_INLINE Packet8f pblendv(const Packet8f& ifPacket, const Packet8f& thenPacket, const Packet8f& elsePacket) 
		{
			return _mm256_blendv_ps(elsePacket, thenPacket, ifPacket);
		}

		template<> 
		EIGEN_STRONG_INLINE Packet4d pblendv(const Packet4d& ifPacket, const Packet4d& thenPacket, const Packet4d& elsePacket)
		{
			return _mm256_blendv_pd(elsePacket, thenPacket, ifPacket);
		}
	}
}

#elif defined(EIGEN_VECTORIZE_SSE2)
#include <xmmintrin.h>

namespace Eigen
{
	namespace internal
	{
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

		EIGEN_STRONG_INLINE void split_two(const Packet4i& x, uint64_t& a, uint64_t& b)
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
		Packet4i pcmpeq<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return _mm_cmpeq_epi32(a, b);
		}

		template<>
		Packet4i psll<Packet4i>(const Packet4i& a, int b)
		{
			return _mm_slli_epi32(a, b);
		}

		template<>
		Packet4i psll64<Packet4i>(const Packet4i& a, int b)
		{
			return _mm_slli_epi64(a, b);
		}

		template<>
		Packet4i psrl64<Packet4i>(const Packet4i& a, int b)
		{
			return _mm_srli_epi64(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmplt<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return _mm_cmplt_epi32(a, b);
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
#ifdef XXX_EIGEN_VECTORIZE_SSE4_1
			return _mm_blendv_ps(elsePacket, thenPacket, ifPacket);
#else
			return _mm_or_ps(_mm_and_ps(ifPacket, thenPacket), _mm_andnot_ps(ifPacket, elsePacket));
#endif
		}

		template<> 
		EIGEN_STRONG_INLINE Packet2d pblendv(const Packet2d& ifPacket, const Packet2d& thenPacket, const Packet2d& elsePacket) 
		{
#ifdef XXX_EIGEN_VECTORIZE_SSE4_1
			return _mm_blendv_pd(elsePacket, thenPacket, ifPacket);
#else
			return _mm_or_pd(_mm_and_pd(ifPacket, thenPacket), _mm_andnot_pd(ifPacket, elsePacket));
#endif
		}
	}
}
#endif

#endif
