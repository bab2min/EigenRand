/**
 * @file PacketFilter.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.5.1
 * @date 2024-09-08
 *
 * @copyright Copyright (c) 2020-2024
 *
 */

#ifndef EIGENRAND_PACKET_FILTER_AVX512_H
#define EIGENRAND_PACKET_FILTER_AVX512_H

#include <immintrin.h>

namespace Eigen
{
	namespace Rand
	{
		namespace detail
		{
			template<>
			class CompressMask<64>
			{
				CompressMask() {}

			public:
				enum { full_size = 16 };
				static const CompressMask& get_inst()
				{
					static CompressMask cm;
					return cm;
				}

				template<typename Packet>
				EIGEN_STRONG_INLINE int compress_append(Packet& _value, const Packet& _mask,
					Packet& _rest, int rest_cnt, bool& full) const
				{
					auto& value = reinterpret_cast<internal::Packet16f&>(_value);
					auto& mask = reinterpret_cast<const internal::Packet16f&>(_mask);
					auto& rest = reinterpret_cast<internal::Packet16f&>(_rest);

					const __mmask16 m = _mm512_movepi32_mask(_mm512_castps_si512(mask));

					if (m == 0xFFFF)
					{
						full = true;
						return rest_cnt;
					}

					const int cnt_m = _mm_popcnt_u32(m);

					const __m512i counting = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
					__m512i rotate = _mm512_sub_epi32(counting, _mm512_set1_epi32(cnt_m));
					__m512 rot_rest = _mm512_permutexvar_ps(rotate, rest);

					__m512 p1 = _mm512_mask_compress_ps(rot_rest, m, value);

					auto new_cnt = rest_cnt + cnt_m;
					if (new_cnt >= full_size)
					{
						rest = rot_rest;
						value = p1;
						full = true;
						return new_cnt - full_size;
					}
					else
					{
						rest = p1;
						full = false;
						return new_cnt;
					}
				}
			};
		}
	}
}
#endif
