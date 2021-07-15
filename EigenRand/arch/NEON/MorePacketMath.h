/**
 * @file MorePacketMath.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.4.0
 * @date 2021-04-26
 *
 * @copyright Copyright (c) 2020-2021
 *
 */

#ifndef EIGENRAND_MORE_PACKET_MATH_NEON_H
#define EIGENRAND_MORE_PACKET_MATH_NEON_H

#include <arm_neon.h>

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
				return vreinterpretq_f32_s32(x);
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
				return vreinterpretq_s32_f32(x);
			}
		};

	}
}

#endif