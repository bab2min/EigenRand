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

#ifndef EIGENRAND_MORE_PACKET_MATH_H
#define EIGENRAND_MORE_PACKET_MATH_H

#include <Eigen/Dense>
#include <cstdint>

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

		// For Eigen 5.x compatibility: reinterpret as 32-bit int packet
		// This is needed because Packet4d -> Packet4l (64-bit) is natural in Eigen 5.x,
		// but some code expects Packet8i (32-bit)
		template<typename Packet>
		inline auto reinterpret_to_int32(const Packet& x)
			-> decltype(reinterpreter<Packet>{}.to_int32(x))
		{
			return reinterpreter<Packet>{}.to_int32(x);
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

		template<typename Packet> 
		EIGEN_STRONG_INLINE bool predux_all(const Packet& a);

		template<typename Packet>
		EIGEN_STRONG_INLINE bool predux_any(const Packet& a);

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

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pnew_andnot(const Packet& a, const Packet& b)
		{
#if defined(EIGEN_VECTORIZE_NEON) || defined(EIGENRAND_EIGEN_34_MODE)
			return pandnot(a, b);
#else
			return pandnot(b, a);
#endif
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
			emm4 = pnew_andnot(pset1<IntPacket>(4), emm4);
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
			Packet ysin1 = pnew_andnot(y, xmm3);
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
			emm4 = pnew_andnot(pseti64<IntPacket>(4), emm4);
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
			Packet ysin1 = pnew_andnot(y, xmm3);
			y2 = psub(y2, ysin2);
			y = psub(y, ysin1);

			xmm1 = padd(ysin1, ysin2);
			xmm2 = padd(y, y2);

			/* update the sign */
			s = pxor(xmm1, sign_bit_sin);
			c = pxor(xmm2, sign_bit_cos);
		}

		/**
		 * @brief Compute the inverse error function (erfinv) for float packets.
		 *
		 * Based on Mike Giles' approximation (GPU Computing Gems, 2010).
		 * Two regions, no Newton refinement needed for float precision.
		 */
		template<typename Packet>
		EIGEN_STRONG_INLINE typename std::enable_if<
			IsFloatPacket<Packet>::value, Packet
		>::type perfinv(const Packet& x)
		{
			// w = -log((1-x)*(1+x))
			Packet w = pnegate(plog(pmul(psub(pset1<Packet>(1.f), x), padd(pset1<Packet>(1.f), x))));
			const Packet threshold = pset1<Packet>(5.f);

			// Region 1: w < 5
			Packet w1 = psub(w, pset1<Packet>(2.5f));
			Packet r1 = pset1<Packet>(2.81022636e-08f);
			r1 = padd(pmul(r1, w1), pset1<Packet>(3.43273939e-07f));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-3.5233877e-06f));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-4.39150654e-06f));
			r1 = padd(pmul(r1, w1), pset1<Packet>(0.00021858087f));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-0.00125372503f));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-0.00417768164f));
			r1 = padd(pmul(r1, w1), pset1<Packet>(0.246640727f));
			r1 = padd(pmul(r1, w1), pset1<Packet>(1.50140941f));

			// Region 2: w >= 5
			Packet w2 = psub(psqrt(w), pset1<Packet>(3.f));
			Packet r2 = pset1<Packet>(-0.000200214257f);
			r2 = padd(pmul(r2, w2), pset1<Packet>(0.000100950558f));
			r2 = padd(pmul(r2, w2), pset1<Packet>(0.00134934322f));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-0.00367342844f));
			r2 = padd(pmul(r2, w2), pset1<Packet>(0.00573950773f));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-0.0076224613f));
			r2 = padd(pmul(r2, w2), pset1<Packet>(0.00943887047f));
			r2 = padd(pmul(r2, w2), pset1<Packet>(1.00167406f));
			r2 = padd(pmul(r2, w2), pset1<Packet>(2.83297682f));

			auto mask = pcmplt(w, threshold);
			Packet p = pblendv(mask, r1, r2);
			return pmul(p, x);
		}

		/**
		 * @brief Compute the inverse error function (erfinv) for double packets.
		 *
		 * Based on Mike Giles' approximation (GPU Computing Gems, 2010).
		 * Three regions for full double precision, no Newton refinement needed.
		 */
		template<typename Packet>
		EIGEN_STRONG_INLINE typename std::enable_if<
			IsDoublePacket<Packet>::value, Packet
		>::type perfinv(const Packet& x)
		{
			Packet w = pnegate(plog(pmul(psub(pset1<Packet>(1.0), x), padd(pset1<Packet>(1.0), x))));

			// Region 1: w < 6.25
			Packet w1 = psub(w, pset1<Packet>(3.125));
			Packet r1 = pset1<Packet>(-3.6444120640178196996e-21);
			r1 = padd(pmul(r1, w1), pset1<Packet>(-1.685059138182016589e-19));
			r1 = padd(pmul(r1, w1), pset1<Packet>(1.2858480715256400167e-18));
			r1 = padd(pmul(r1, w1), pset1<Packet>(1.115787767802518096e-17));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-1.333171662854620906e-16));
			r1 = padd(pmul(r1, w1), pset1<Packet>(2.0972767875968561637e-17));
			r1 = padd(pmul(r1, w1), pset1<Packet>(6.6376381343583238325e-15));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-4.0545662729752068639e-14));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-8.1519341976054721522e-14));
			r1 = padd(pmul(r1, w1), pset1<Packet>(2.6335093153082322977e-12));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-1.2975133253453532498e-11));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-5.4154120542946279317e-11));
			r1 = padd(pmul(r1, w1), pset1<Packet>(1.051212273321532285e-09));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-4.1126339803469836976e-09));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-2.9070369957882005086e-08));
			r1 = padd(pmul(r1, w1), pset1<Packet>(4.2347877827932403518e-07));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-1.3654692000834678645e-06));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-1.3882523362786468719e-05));
			r1 = padd(pmul(r1, w1), pset1<Packet>(0.0001867342080340571352));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-0.00074070253416626697512));
			r1 = padd(pmul(r1, w1), pset1<Packet>(-0.0060336708714301490533));
			r1 = padd(pmul(r1, w1), pset1<Packet>(0.24015818242558961693));
			r1 = padd(pmul(r1, w1), pset1<Packet>(1.6536545626831027356));

			// Region 2: 6.25 <= w < 16
			Packet w2 = psub(psqrt(w), pset1<Packet>(3.25));
			Packet r2 = pset1<Packet>(2.2137376921775787049e-09);
			r2 = padd(pmul(r2, w2), pset1<Packet>(9.0756561938885390979e-08));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-2.7517406297064545428e-07));
			r2 = padd(pmul(r2, w2), pset1<Packet>(1.8239629214389227755e-08));
			r2 = padd(pmul(r2, w2), pset1<Packet>(1.5027403968909827627e-06));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-4.013867526981545969e-06));
			r2 = padd(pmul(r2, w2), pset1<Packet>(2.9234449089955446044e-06));
			r2 = padd(pmul(r2, w2), pset1<Packet>(1.2475304481671778723e-05));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-4.7318229009055733981e-05));
			r2 = padd(pmul(r2, w2), pset1<Packet>(6.8284851459573175448e-05));
			r2 = padd(pmul(r2, w2), pset1<Packet>(2.4031110387097893999e-05));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-0.0003550375203628474796));
			r2 = padd(pmul(r2, w2), pset1<Packet>(0.00095328937973738049703));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-0.0016882755560235047313));
			r2 = padd(pmul(r2, w2), pset1<Packet>(0.0024914420961078508066));
			r2 = padd(pmul(r2, w2), pset1<Packet>(-0.0037512085075692412107));
			r2 = padd(pmul(r2, w2), pset1<Packet>(0.005370914553590063617));
			r2 = padd(pmul(r2, w2), pset1<Packet>(1.0052589676941592334));
			r2 = padd(pmul(r2, w2), pset1<Packet>(3.0838856104922207635));

			// Region 3: w >= 16
			Packet w3 = psub(psqrt(w), pset1<Packet>(5.0));
			Packet r3 = pset1<Packet>(-2.7109920616438573243e-11);
			r3 = padd(pmul(r3, w3), pset1<Packet>(-2.5556418169965252055e-10));
			r3 = padd(pmul(r3, w3), pset1<Packet>(1.5076572693500548083e-09));
			r3 = padd(pmul(r3, w3), pset1<Packet>(-3.7894654401267369937e-09));
			r3 = padd(pmul(r3, w3), pset1<Packet>(7.6157012080783393804e-09));
			r3 = padd(pmul(r3, w3), pset1<Packet>(-1.4960026627149240478e-08));
			r3 = padd(pmul(r3, w3), pset1<Packet>(2.9147953450901080826e-08));
			r3 = padd(pmul(r3, w3), pset1<Packet>(-6.7711997758452339498e-08));
			r3 = padd(pmul(r3, w3), pset1<Packet>(2.2900482228026654717e-07));
			r3 = padd(pmul(r3, w3), pset1<Packet>(-9.9298272942317002539e-07));
			r3 = padd(pmul(r3, w3), pset1<Packet>(4.5260625972231537039e-06));
			r3 = padd(pmul(r3, w3), pset1<Packet>(-1.9681778105531670567e-05));
			r3 = padd(pmul(r3, w3), pset1<Packet>(7.5995277030017761139e-05));
			r3 = padd(pmul(r3, w3), pset1<Packet>(-0.00021503011930044477347));
			r3 = padd(pmul(r3, w3), pset1<Packet>(-0.00013871931833623122026));
			r3 = padd(pmul(r3, w3), pset1<Packet>(1.0103004648645343977));
			r3 = padd(pmul(r3, w3), pset1<Packet>(4.8499064014085844221));

			// Select region: w < 6.25 ? r1 : (w < 16 ? r2 : r3)
			auto mask1 = pcmplt(w, pset1<Packet>(6.25));
			auto mask2 = pcmplt(w, pset1<Packet>(16.0));
			Packet p = pblendv(mask2, r2, r3);
			p = pblendv(mask1, r1, p);
			return pmul(p, x);
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
			Packet ysin1 = pnew_andnot(y, xmm3);

			xmm1 = padd(ysin1, ysin2);

			/* update the sign */
			return pxor(xmm1, sign_bit_sin);
		}
	}
}

#ifdef EIGEN_VECTORIZE_AVX512
#include "arch/AVX512/MorePacketMath.h"
#endif

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
