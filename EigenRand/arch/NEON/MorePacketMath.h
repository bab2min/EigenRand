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

#ifndef EIGENRAND_MORE_PACKET_MATH_NEON_H
#define EIGENRAND_MORE_PACKET_MATH_NEON_H

#include <arm_neon.h>

// device func of casting for Eigen ~3.3.9
#ifdef EIGENRAND_EIGEN_33_MODE
namespace Eigen
{
	namespace internal
	{
		template<>
		EIGEN_DEVICE_FUNC inline Packet4f pcast<Packet4i, Packet4f>(const Packet4i& a)
		{
			return vcvtq_f32_s32(a);
		}

		template<>
		EIGEN_DEVICE_FUNC inline Packet4i pcast<Packet4f, Packet4i>(const Packet4f& a)
		{
			return vcvtq_s32_f32(a);
		}

	}
}
#endif

namespace Eigen
{
	namespace internal
	{
		template<>
		struct IsIntPacket<Packet4i> : std::true_type {};

		template<>
		struct IsFloatPacket<Packet4f> : std::true_type {};

		template<>
		struct HalfPacket<Packet4i>
		{
			using type = uint64_t;
		};

		template<>
		struct reinterpreter<Packet4i>
		{
			EIGEN_STRONG_INLINE Packet4f to_float(const Packet4i& x)
			{
				return (Packet4f)vreinterpretq_f32_s32(x);
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

			EIGEN_STRONG_INLINE Packet4i to_int(const Packet4f& x)
			{
				return (Packet4i)vreinterpretq_s32_f32(x);
			}
		};

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmpeq<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return vreinterpretq_s32_u32(vceqq_s32(a, b));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pcmpeq<Packet4f>(const Packet4f& a, const Packet4f& b)
		{
			return vreinterpretq_f32_u32(vceqq_f32(a, b));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pbitnot<Packet4i>(const Packet4i& a)
		{
			return vmvnq_s32(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pbitnot<Packet4f>(const Packet4f& a)
		{
			return (Packet4f)vreinterpretq_f32_s32(pbitnot((Packet4i)vreinterpretq_s32_f32(a)));
		}

		template<>
		struct BitShifter<Packet4i>
		{
			template<int b>
			EIGEN_STRONG_INLINE Packet4i sll(const Packet4i& a)
			{
				return vreinterpretq_s32_u32(vshlq_n_u32(vreinterpretq_u32_s32(a), b));
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet4i srl(const Packet4i& a, int _b = b)
			{
				if (b > 0)
				{
					return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), b > 0 ? b : 1));
				}
				else
				{
					switch (_b)
					{
					case 0: return a;
					case 1: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 1));
					case 2: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 2));
					case 3: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 3));
					case 4: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 4));
					case 5: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 5));
					case 6: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 6));
					case 7: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 7));
					case 8: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 8));
					case 9: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 9));
					case 10: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 10));
					case 11: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 11));
					case 12: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 12));
					case 13: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 13));
					case 14: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 14));
					case 15: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 15));
					case 16: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 16));
					case 17: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 17));
					case 18: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 18));
					case 19: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 19));
					case 20: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 20));
					case 21: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 21));
					case 22: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 22));
					case 23: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 23));
					case 24: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 24));
					case 25: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 25));
					case 26: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 26));
					case 27: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 27));
					case 28: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 28));
					case 29: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 29));
					case 30: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 30));
					case 31: return vreinterpretq_s32_u32(vshrq_n_u32(vreinterpretq_u32_s32(a), 31));
					}
					return vdupq_n_s32(0);
				}
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet4i sll64(const Packet4i& a)
			{
				return vreinterpretq_s32_u64(vshlq_n_u64(vreinterpretq_u64_s32(a), b));
			}

			template<int b>
			EIGEN_STRONG_INLINE Packet4i srl64(const Packet4i& a)
			{
				return vreinterpretq_s32_u64(vshrq_n_u64(vreinterpretq_u64_s32(a), b));
			}
		};

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmplt<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return vreinterpretq_s32_u32(vcltq_s32(a, b));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pcmplt<Packet4f>(const Packet4f& a, const Packet4f& b)
		{
			return vreinterpretq_f32_u32(vcltq_f32(a, b));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pcmple<Packet4f>(const Packet4f& a, const Packet4f& b)
		{
			return vreinterpretq_f32_u32(vcleq_f32(a, b));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pblendv(const Packet4f& ifPacket, const Packet4f& thenPacket, const Packet4f& elsePacket)
		{
			return vbslq_f32(vreinterpretq_u32_f32(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pblendv(const Packet4i& ifPacket, const Packet4f& thenPacket, const Packet4f& elsePacket)
		{
			return vbslq_f32(vreinterpretq_u32_s32(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pblendv(const Packet4i& ifPacket, const Packet4i& thenPacket, const Packet4i& elsePacket)
		{
			return vbslq_s32(vreinterpretq_u32_s32(ifPacket), thenPacket, elsePacket);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pgather<Packet4i>(const int* addr, const Packet4i& index)
		{
			int32_t u[4];
			vst1q_s32(u, index);
			int32_t t[4];
			t[0] = addr[u[0]];
			t[1] = addr[u[1]];
			t[2] = addr[u[2]];
			t[3] = addr[u[3]];
			return vld1q_s32(t);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f pgather<Packet4i>(const float* addr, const Packet4i& index)
		{
			int32_t u[4];
			vst1q_s32(u, index);
			float t[4];
			t[0] = addr[u[0]];
			t[1] = addr[u[1]];
			t[2] = addr[u[2]];
			t[3] = addr[u[3]];
			return vld1q_f32(t);
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet4f>(const Packet4f& a)
		{
			int32_t bits[4] = { 1, 2, 4, 8 };
			auto r = vbslq_s32(vreinterpretq_u32_f32(a), vld1q_s32(bits), vdupq_n_s32(0));
			auto s = vadd_s32(vget_low_s32(r), vget_high_s32(r));
			return vget_lane_s32(vpadd_s32(s, s), 0);
		}

		template<>
		EIGEN_STRONG_INLINE int pmovemask<Packet4i>(const Packet4i& a)
		{
			return pmovemask((Packet4f)vreinterpretq_f32_s32(a));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f ptruncate<Packet4f>(const Packet4f& a)
		{
			return vrndq_f32(a);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pseti64<Packet4i>(uint64_t a)
		{
			return vreinterpretq_s32_u64(vdupq_n_u64(a));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pcmpeq64<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
			return vreinterpretq_s32_u64(vceqq_s64(vreinterpretq_s64_s32(a), vreinterpretq_s64_s32(b)));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pmuluadd64<Packet4i>(const Packet4i& a, uint64_t b, uint64_t c)
		{
			uint64_t u[2];
			vst1q_u64(u, vreinterpretq_u64_s32(a));
			u[0] = u[0] * b + c;
			u[1] = u[1] * b + c;
			return vreinterpretq_s32_u64(vld1q_u64(u));
		}

	#ifdef EIGENRAND_EIGEN_33_MODE
		template<>
		EIGEN_STRONG_INLINE Packet4f plog<Packet4f>(const Packet4f& _x)
		{
			Packet4f x = _x;
			_EIGEN_DECLARE_CONST_Packet4f(1, 1.0f);
			_EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);
			_EIGEN_DECLARE_CONST_Packet4i(0x7f, 0x7f);

			const Packet4f p4f_inv_mant_mask = (Packet4f)vreinterpretq_f32_s32(pset1<Packet4i>(~0x7f800000));

			/* the smallest non denormalized float number */
			const Packet4f p4f_min_norm_pos = (Packet4f)vreinterpretq_f32_s32(pset1<Packet4i>(0x00800000));
			const Packet4f p4f_minus_inf = (Packet4f)vreinterpretq_f32_s32(pset1<Packet4i>(0xff800000));

			/* natural logarithm computed for 4 simultaneous float
			  return NaN for x <= 0
			*/
			_EIGEN_DECLARE_CONST_Packet4f(cephes_SQRTHF, 0.707106781186547524f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p0, 7.0376836292E-2f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p1, -1.1514610310E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p2, 1.1676998740E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p3, -1.2420140846E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p4, +1.4249322787E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p5, -1.6668057665E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p6, +2.0000714765E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p7, -2.4999993993E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_p8, +3.3333331174E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_q1, -2.12194440e-4f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_log_q2, 0.693359375f);


			Packet4i emm0;
			
			Packet4f invalid_mask = pbitnot(pcmple(pset1<Packet4f>(0), x)); // not greater equal is true if x is NaN
			Packet4f iszero_mask = pcmpeq(x, pset1<Packet4f>(0));

			x = pmax(x, p4f_min_norm_pos);  /* cut off denormalized stuff */
			emm0 = BitShifter<Packet4i>{}.template srl<23>((Packet4i)vreinterpretq_s32_f32(x));

			/* keep only the fractional part */
			x = pand(x, p4f_inv_mant_mask);
			x = por(x, p4f_half);

			emm0 = psub(emm0, p4i_0x7f);
			Packet4f e = padd(Packet4f(vcvtq_f32_s32(emm0)), p4f_1);

			/* part2:
			   if( x < SQRTHF ) {
				 e -= 1;
				 x = x + x - 1.0;
			   } else { x = x - 1.0; }
			*/
			Packet4f mask = pcmplt(x, p4f_cephes_SQRTHF);
			Packet4f tmp = pand(x, mask);
			x = psub(x, p4f_1);
			e = psub(e, pand(p4f_1, mask));
			x = padd(x, tmp);

			Packet4f x2 = pmul(x, x);
			Packet4f x3 = pmul(x2, x);

			Packet4f y, y1, y2;
			y = pmadd(p4f_cephes_log_p0, x, p4f_cephes_log_p1);
			y1 = pmadd(p4f_cephes_log_p3, x, p4f_cephes_log_p4);
			y2 = pmadd(p4f_cephes_log_p6, x, p4f_cephes_log_p7);
			y = pmadd(y, x, p4f_cephes_log_p2);
			y1 = pmadd(y1, x, p4f_cephes_log_p5);
			y2 = pmadd(y2, x, p4f_cephes_log_p8);
			y = pmadd(y, x3, y1);
			y = pmadd(y, x3, y2);
			y = pmul(y, x3);

			y1 = pmul(e, p4f_cephes_log_q1);
			tmp = pmul(x2, p4f_half);
			y = padd(y, y1);
			x = psub(x, tmp);
			y2 = pmul(e, p4f_cephes_log_q2);
			x = padd(x, y);
			x = padd(x, y2);
			// negative arg will be NAN, 0 will be -INF
			return pblendv(iszero_mask, p4f_minus_inf, por(x, invalid_mask));
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f psqrt<Packet4f>(const Packet4f& x)
		{
			return vsqrtq_f32(x);
		}

		template<>
		EIGEN_STRONG_INLINE Packet4f psin<Packet4f>(const Packet4f& _x)
		{
			Packet4f x = _x;
			_EIGEN_DECLARE_CONST_Packet4f(1, 1.0f);
			_EIGEN_DECLARE_CONST_Packet4f(half, 0.5f);

			_EIGEN_DECLARE_CONST_Packet4i(1, 1);
			_EIGEN_DECLARE_CONST_Packet4i(not1, ~1);
			_EIGEN_DECLARE_CONST_Packet4i(2, 2);
			_EIGEN_DECLARE_CONST_Packet4i(4, 4);

			const Packet4f p4f_sign_mask = (Packet4f)vreinterpretq_f32_s32(pset1<Packet4i>(0x80000000));

			_EIGEN_DECLARE_CONST_Packet4f(minus_cephes_DP1, -0.78515625f);
			_EIGEN_DECLARE_CONST_Packet4f(minus_cephes_DP2, -2.4187564849853515625e-4f);
			_EIGEN_DECLARE_CONST_Packet4f(minus_cephes_DP3, -3.77489497744594108e-8f);
			_EIGEN_DECLARE_CONST_Packet4f(sincof_p0, -1.9515295891E-4f);
			_EIGEN_DECLARE_CONST_Packet4f(sincof_p1, 8.3321608736E-3f);
			_EIGEN_DECLARE_CONST_Packet4f(sincof_p2, -1.6666654611E-1f);
			_EIGEN_DECLARE_CONST_Packet4f(coscof_p0, 2.443315711809948E-005f);
			_EIGEN_DECLARE_CONST_Packet4f(coscof_p1, -1.388731625493765E-003f);
			_EIGEN_DECLARE_CONST_Packet4f(coscof_p2, 4.166664568298827E-002f);
			_EIGEN_DECLARE_CONST_Packet4f(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

			Packet4f xmm1, xmm2, xmm3, sign_bit, y;

			Packet4i emm0, emm2;
			sign_bit = x;
			/* take the absolute value */
			x = pabs(x);

			/* take the modulo */

			/* extract the sign bit (upper one) */
			sign_bit = pand(sign_bit, p4f_sign_mask);

			/* scale by 4/Pi */
			y = pmul(x, p4f_cephes_FOPI);

			/* store the integer part of y in mm0 */
			emm2 = vcvtq_s32_f32(y);
			/* j=(j+1) & (~1) (see the cephes sources) */
			emm2 = padd(emm2, p4i_1);
			emm2 = pand(emm2, p4i_not1);
			y = vcvtq_f32_s32(emm2);
			/* get the swap sign flag */
			emm0 = pand(emm2, p4i_4);
			emm0 = BitShifter<Packet4i>{}.template sll<29>(emm0);
			/* get the polynom selection mask
			   there is one polynom for 0 <= x <= Pi/4
			   and another one for Pi/4<x<=Pi/2

			   Both branches will be computed.
			*/
			emm2 = pand(emm2, p4i_2);
			emm2 = pcmpeq(emm2, pset1<Packet4i>(0));

			Packet4f swap_sign_bit = (Packet4f)vreinterpretq_f32_s32(emm0);
			Packet4f poly_mask = (Packet4f)vreinterpretq_f32_s32(emm2);
			sign_bit = pxor(sign_bit, swap_sign_bit);

			/* The magic pass: "Extended precision modular arithmetic"
			   x = ((x - y * DP1) - y * DP2) - y * DP3; */
			xmm1 = pmul(y, p4f_minus_cephes_DP1);
			xmm2 = pmul(y, p4f_minus_cephes_DP2);
			xmm3 = pmul(y, p4f_minus_cephes_DP3);
			x = padd(x, xmm1);
			x = padd(x, xmm2);
			x = padd(x, xmm3);

			/* Evaluate the first polynom  (0 <= x <= Pi/4) */
			y = p4f_coscof_p0;
			Packet4f z = pmul(x, x);

			y = pmadd(y, z, p4f_coscof_p1);
			y = pmadd(y, z, p4f_coscof_p2);
			y = pmul(y, z);
			y = pmul(y, z);
			Packet4f tmp = pmul(z, p4f_half);
			y = psub(y, tmp);
			y = padd(y, p4f_1);

			/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

			Packet4f y2 = p4f_sincof_p0;
			y2 = pmadd(y2, z, p4f_sincof_p1);
			y2 = pmadd(y2, z, p4f_sincof_p2);
			y2 = pmul(y2, z);
			y2 = pmul(y2, x);
			y2 = padd(y2, x);

			/* select the correct result from the two polynoms */
			y = pblendv(poly_mask, y2, y);
			/* update the sign */
			return pxor(y, sign_bit);
		}
	#endif
	}
}

#endif
