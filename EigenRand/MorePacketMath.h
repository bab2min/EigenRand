/**
 * @file MorePacketMath.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.3.3
 * @date 2021-03-31
 * 
 * @copyright Copyright (c) 2020-2021
 * 
 */

#ifndef EIGENRAND_MORE_PACKET_MATH_H
#define EIGENRAND_MORE_PACKET_MATH_H

#include <Eigen/Dense>

namespace Eigen
{
	namespace internal
	{
		template<typename Ty>
		struct IsIntPacket : std::false_type {};

		template<typename Ty>
		struct IsFloatPacket : std::false_type {};

		template<typename Ty>
		struct IsDoublePacket : std::false_type {};

		template<typename Ty>
		struct HalfPacket;

#ifdef EIGEN_VECTORIZE_AVX2
		template<>
		struct IsIntPacket<Packet8i> : std::true_type {};

		template<>
		struct HalfPacket<Packet8i>
		{
			using type = Packet4i;
		};
#endif
#ifdef EIGEN_VECTORIZE_AVX
		template<>
		struct IsFloatPacket<Packet8f> : std::true_type {};

		template<>
		struct IsDoublePacket<Packet4d> : std::true_type {};
#endif
#ifdef EIGEN_VECTORIZE_SSE2
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
#endif
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
		EIGEN_STRONG_INLINE Packet padd64(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psub64(const Packet& a, const Packet& b);

		template <typename SrcPacket, typename TgtPacket>
		EIGEN_STRONG_INLINE TgtPacket pcast64(const SrcPacket& a);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmpeq(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psll(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psrl(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psll64(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psrl64(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE int pmovemask(const Packet& a);

		template<typename Packet>
		EIGEN_STRONG_INLINE typename std::enable_if<
			IsFloatPacket<Packet>::value, Packet
		>::type pext_sign(const Packet& a)
		{
			using IntPacket = decltype(reinterpret_to_int(a));
			return reinterpret_to_float(
				pand(reinterpret_to_int(a), pset1<IntPacket>(0x80000000))
			);
		}

		template<typename Packet>
		EIGEN_STRONG_INLINE typename std::enable_if<
			IsDoublePacket<Packet>::value, Packet
		>::type pext_sign(const Packet& a)
		{
			using IntPacket = decltype(reinterpret_to_int(a));
			return reinterpret_to_double(
				pand(reinterpret_to_int(a), pseti64<IntPacket>(0x8000000000000000))
			);
		}

		template<>
		EIGEN_STRONG_INLINE uint64_t psll64<uint64_t>(const uint64_t& a, int b)
		{
			return a << b;
		}

		template<>
		EIGEN_STRONG_INLINE uint64_t psrl64<uint64_t>(const uint64_t& a, int b)
		{
			return a >> b;
		}

		// approximation : lgamma(z) ~= (z+2.5)ln(z+3) - z - 3 + 0.5 ln (2pi) + 1/12/(z + 3) - ln (z(z+1)(z+2))
		template<typename Packet>
		EIGEN_STRONG_INLINE Packet plgamma_approx(const Packet& x)
		{
			auto x_3 = padd(x, pset1<Packet>(3));
			auto ret = pmul(padd(x_3, pset1<Packet>(-0.5)), plog(x_3));
			ret = psub(ret, x_3);
			ret = padd(ret, pset1<Packet>(0.9189385332046727));
			ret = padd(ret, pdiv(pset1<Packet>(1 / 12.), x_3));
			ret = psub(ret, plog(pmul(
				pmul(psub(x_3, pset1<Packet>(1)), psub(x_3, pset1<Packet>(2))), x)));
			return ret;
		}

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmplt(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmple(const Packet& a, const Packet& b);

		template<typename PacketIf, typename Packet>
		EIGEN_STRONG_INLINE Packet pblendv(const PacketIf& ifPacket, const Packet& thenPacket, const Packet& elsePacket);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pgather(const int* addr, const Packet& index);

		template<typename Packet>
		EIGEN_STRONG_INLINE auto pgather(const float* addr, const Packet& index) -> decltype(reinterpret_to_float(std::declval<Packet>()));

		template<typename Packet>
		EIGEN_STRONG_INLINE auto pgather(const double* addr, const Packet& index, bool upperhalf = false) -> decltype(reinterpret_to_double(std::declval<Packet>()));

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet ptruncate(const Packet& a);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmpeq64(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmplt64(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pmuluadd64(const Packet& a, uint64_t b, uint64_t c);

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

		template<typename _Scalar>
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


		struct float2
		{
			float f[2];
		};

		EIGEN_STRONG_INLINE float2 bit_to_ur_float(uint64_t x)
		{
			bit_scalar<float> bs;
			float2 ret;
			ret.f[0] = bs.to_ur(x & 0xFFFFFFFF);
			ret.f[1] = bs.to_ur(x >> 32);
			return ret;
		}

		template<typename Packet>
		EIGEN_STRONG_INLINE typename std::enable_if<
			IsFloatPacket<Packet>::value
		>::type psincos(Packet x, Packet& s, Packet& c)
		{
			Packet xmm1, xmm2, xmm3 = pset1<Packet>(0), sign_bit_sin, y;
			using IntPacket = decltype(reinterpret_to_int(x));
			IntPacket emm0, emm2, emm4;

			sign_bit_sin = x;
			/* take the absolute value */
			x = pabs(x);
			/* extract the sign bit (upper one) */
			sign_bit_sin = pext_sign(sign_bit_sin);

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
		EIGEN_STRONG_INLINE typename std::enable_if<
			IsDoublePacket<Packet>::value
		>::type psincos(Packet x, Packet& s, Packet& c)
		{
			Packet xmm1, xmm2, xmm3 = pset1<Packet>(0), sign_bit_sin, y;
			using IntPacket = decltype(reinterpret_to_int(x));
			IntPacket emm0, emm2, emm4;

			sign_bit_sin = x;
			/* take the absolute value */
			x = pabs(x);
			/* extract the sign bit (upper one) */
			sign_bit_sin = pext_sign(sign_bit_sin);

			/* scale by 4/Pi */
			y = pmul(x, pset1<Packet>(1.27323954473516));

			/* store the integer part of y in emm2 */
			emm2 = pcast64<Packet, IntPacket>(y);

			/* j=(j+1) & (~1) (see the cephes sources) */
			emm2 = padd64(emm2, pseti64<IntPacket>(1));
			emm2 = pand(emm2, pseti64<IntPacket>(~1ll));
			y = pcast64<IntPacket, Packet>(emm2);

			emm4 = emm2;

			/* get the swap sign flag for the sine */
			emm0 = pand(emm2, pseti64<IntPacket>(4));
			emm0 = psll64(emm0, 61);
			Packet swap_sign_bit_sin = reinterpret_to_double(emm0);

			/* get the polynom selection mask for the sine*/
			emm2 = pand(emm2, pseti64<IntPacket>(2));

			emm2 = pcmpeq64(emm2, pseti64<IntPacket>(0));
			Packet poly_mask = reinterpret_to_double(emm2);

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

			emm4 = psub64(emm4, pseti64<IntPacket>(2));
			emm4 = pandnot(emm4, pseti64<IntPacket>(4));
			emm4 = psll64(emm4, 61);
			Packet sign_bit_cos = reinterpret_to_double(emm4);
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
		EIGEN_STRONG_INLINE typename std::enable_if<
			IsDoublePacket<Packet>::value, Packet
		>::type _psin(Packet x)
		{
			Packet xmm1, xmm2, xmm3 = pset1<Packet>(0), sign_bit_sin, y;
			using IntPacket = decltype(reinterpret_to_int(x));
			IntPacket emm0, emm2;

			sign_bit_sin = x;
			/* take the absolute value */
			x = pabs(x);
			/* extract the sign bit (upper one) */
			sign_bit_sin = pext_sign(sign_bit_sin);

			/* scale by 4/Pi */
			y = pmul(x, pset1<Packet>(1.27323954473516));

			/* store the integer part of y in emm2 */
			emm2 = pcast64<Packet, IntPacket>(y);

			/* j=(j+1) & (~1) (see the cephes sources) */
			emm2 = padd64(emm2, pseti64<IntPacket>(1));
			emm2 = pand(emm2, pseti64<IntPacket>(~1ll));
			y = pcast64<IntPacket, Packet>(emm2);

			/* get the swap sign flag for the sine */
			emm0 = pand(emm2, pseti64<IntPacket>(4));
			emm0 = psll64(emm0, 61);
			Packet swap_sign_bit_sin = reinterpret_to_double(emm0);

			/* get the polynom selection mask for the sine*/
			emm2 = pand(emm2, pseti64<IntPacket>(2));

			emm2 = pcmpeq64(emm2, pseti64<IntPacket>(0));
			Packet poly_mask = reinterpret_to_double(emm2);

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

			xmm1 = padd(ysin1, ysin2);

			/* update the sign */
			return pxor(xmm1, sign_bit_sin);
		}
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
		EIGEN_STRONG_INLINE Packet8i psll<Packet8i>(const Packet8i& a, int b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_slli_epi32(a, b);
#else
			Packet4i a1, a2;
			split_two(a, a1, a2);
			return combine_two((Packet4i)_mm_slli_epi32(a1, b), (Packet4i)_mm_slli_epi32(a2, b));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i psrl<Packet8i>(const Packet8i& a, int b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_srli_epi32(a, b);
#else
			Packet4i a1, a2;
			split_two(a, a1, a2);
			return combine_two((Packet4i)_mm_srli_epi32(a1, b), (Packet4i)_mm_srli_epi32(a2, b));
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
			return combine_two((Packet4i)_mm_slli_epi64(a1, b), (Packet4i)_mm_slli_epi64(a2, b));
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
			return combine_two((Packet4i)_mm_srli_epi64(a1, b), (Packet4i)_mm_srli_epi64(a2, b));
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
		EIGEN_STRONG_INLINE Packet8f pgather<Packet8i>(const float *addr, const Packet8i& index)
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
		EIGEN_STRONG_INLINE Packet4d pgather<Packet8i>(const double *addr, const Packet8i& index, bool upperhalf)
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
			x = _mm256_add_pd(x, _mm256_set1_pd(0x0018000000000000));
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

			Packet4d emm0 = uint64_to_double(psrl64(_mm256_castpd_si256(x), 52));
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

#if EIGEN_VERSION_AT_LEAST(3,3,5)
#else
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

#ifdef EIGEN_VECTORIZE_SSE2
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
		EIGEN_STRONG_INLINE Packet4i psll<Packet4i>(const Packet4i& a, int b)
		{
			return _mm_slli_epi32(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i psrl<Packet4i>(const Packet4i& a, int b)
		{
			return _mm_srli_epi32(a, b);
		}


		template<>
		EIGEN_STRONG_INLINE Packet4i psll64<Packet4i>(const Packet4i& a, int b)
		{
			return _mm_slli_epi64(a, b);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i psrl64<Packet4i>(const Packet4i& a, int b)
		{
			return _mm_srli_epi64(a, b);
		}

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
			x = _mm_add_pd(x, _mm_set1_pd(0x0018000000000000));
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
	}
}
#endif

#endif
