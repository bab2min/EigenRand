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

#ifndef EIGENRAND_MORE_PACKET_MATH_H
#define EIGENRAND_MORE_PACKET_MATH_H

#include <Eigen/Dense>

#define EIGENRAND_PRINT_PACKET(p) do { using _MTy = typename std::remove_const<typename std::remove_reference<decltype(p)>::type>::type; typename std::conditional<Eigen::internal::IsFloatPacket<_MTy>::value, float, typename std::conditional<Eigen::internal::IsDoublePacket<_MTy>::value, double, int>::type>::type f[4]; Eigen::internal::pstore(f, p); std::cout << #p " " << f[0] << " " << f[1] << " " << f[2] << " " << f[3] << std::endl; } while(0)

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

		template<typename Packet>
		struct reinterpreter{};

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
		EIGEN_STRONG_INLINE void split_two(const Packet& p, typename HalfPacket<Packet>::type& a, typename HalfPacket<Packet>::type& b);

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
		struct BitShifter {};

		template<int b, typename Packet>
		EIGEN_STRONG_INLINE Packet psll(const Packet& a);

		template<int _b, typename Packet>
		EIGEN_STRONG_INLINE Packet psrl(const Packet& a, int b = _b);

		template<int b, typename Packet>
		EIGEN_STRONG_INLINE Packet psll64(const Packet& a);

		template<int b, typename Packet>
		EIGEN_STRONG_INLINE Packet psrl64(const Packet& a);

		/*template<typename Packet>
		EIGEN_STRONG_INLINE Packet psll(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psrl(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psll64(const Packet& a, int b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet psrl64(const Packet& a, int b);*/

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

		/*template<>
		EIGEN_STRONG_INLINE uint64_t psll64<uint64_t>(const uint64_t& a, int b)
		{
			return a << b;
		}

		template<>
		EIGEN_STRONG_INLINE uint64_t psrl64<uint64_t>(const uint64_t& a, int b)
		{
			return a >> b;
		}*/

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

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pbitnot(const Packet& a);

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
		struct BitScalar;

		template<>
		struct BitScalar<float>
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
		struct BitScalar<double>
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
			BitScalar<float> bs;
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
			emm0 = psll<29>(emm0);
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
	#if defined(EIGEN_VECTORIZE_NEON) || defined(EIGENRAND_EIGEN_34_MODE)
			emm4 = pandnot(pset1<IntPacket>(4), emm4);
	#else
			emm4 = pandnot(emm4, pset1<IntPacket>(4));
	#endif
			emm4 = psll<29>(emm4);
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
	#if defined(EIGEN_VECTORIZE_NEON) || defined(EIGENRAND_EIGEN_34_MODE)
			Packet ysin1 = pandnot(y, xmm3);
	#else
			Packet ysin1 = pandnot(xmm3, y);
	#endif
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
			emm0 = psll64<61>(emm0);
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
	#if defined(EIGEN_VECTORIZE_NEON) || defined(EIGENRAND_EIGEN_34_MODE)
			emm4 = pandnot(pseti64<IntPacket>(4), emm4);
	#else
			emm4 = pandnot(emm4, pseti64<IntPacket>(4));
	#endif
			emm4 = psll64<61>(emm4);
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
	#if defined(EIGEN_VECTORIZE_NEON) || defined(EIGENRAND_EIGEN_34_MODE)
			Packet ysin1 = pandnot(y, xmm3);
	#else
			Packet ysin1 = pandnot(xmm3, y);
	#endif
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
			emm0 = psll64<61>(emm0);
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
	#if defined(EIGEN_VECTORIZE_NEON) || defined(EIGENRAND_EIGEN_34_MODE)
			Packet ysin1 = pandnot(y, xmm3);
	#else
			Packet ysin1 = pandnot(xmm3, y);
	#endif

			xmm1 = padd(ysin1, ysin2);

			/* update the sign */
			return pxor(xmm1, sign_bit_sin);
		}
	}
}

#ifdef EIGEN_VECTORIZE_AVX
#include "arch/AVX/MorePacketMath.h"
#endif

#ifdef EIGEN_VECTORIZE_SSE2
#include "arch/SSE/MorePacketMath.h"
#endif

#ifdef EIGEN_VECTORIZE_NEON
#include "arch/NEON/MorePacketMath.h"
#endif

namespace Eigen
{
	namespace internal
	{
		template<int b, typename Packet>
		EIGEN_STRONG_INLINE Packet psll(const Packet& a)
		{
			return BitShifter<Packet>{}.template sll<b>(a);
		}

		template<int _b, typename Packet>
		EIGEN_STRONG_INLINE Packet psrl(const Packet& a, int b)
		{
			return BitShifter<Packet>{}.template srl<_b>(a, b);
		}

		template<int b, typename Packet>
		EIGEN_STRONG_INLINE Packet psll64(const Packet& a)
		{
			return BitShifter<Packet>{}.template sll64<b>(a);
		}

		template<int b, typename Packet>
		EIGEN_STRONG_INLINE Packet psrl64(const Packet& a)
		{
			return BitShifter<Packet>{}.template srl64<b>(a);
		}
	}
}

#endif