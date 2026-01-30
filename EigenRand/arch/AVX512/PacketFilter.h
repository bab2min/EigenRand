/**
 * @file PacketFilter.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.6.0
 * @date 2026-01-31
 *
 * @copyright Copyright (c) 2020-2026
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

				// Helper to extract raw __m512i from packet types (works with both
				// Eigen 3.x raw types and Eigen 5.x eigen_packet_wrapper types)
				template<typename Packet>
				static EIGEN_STRONG_INLINE __m512i to_raw_m512i(const Packet& p)
				{
					return reinterpret_cast<const __m512i&>(p);
				}

				// Helper to convert raw __m512i back to Packet type
				template<typename Packet>
				static EIGEN_STRONG_INLINE Packet from_raw_m512i(__m512i v)
				{
					return reinterpret_cast<const Packet&>(v);
				}

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
					// Convert to raw __m512 for SIMD operations
					__m512 value = _mm512_castsi512_ps(to_raw_m512i(_value));
					__m512 mask_ps = _mm512_castsi512_ps(to_raw_m512i(_mask));
					__m512 rest_val = _mm512_castsi512_ps(to_raw_m512i(_rest));

					const __mmask16 m = _mm512_movepi32_mask(_mm512_castps_si512(mask_ps));

					if (m == 0xFFFF)
					{
						full = true;
						return rest_cnt;
					}

					const int cnt_m = _mm_popcnt_u32(m);

					const __m512i counting = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
					__m512i rotate = _mm512_sub_epi32(counting, _mm512_set1_epi32(cnt_m));
					__m512 rot_rest = _mm512_permutexvar_ps(rotate, rest_val);

					__m512 p1 = _mm512_mask_compress_ps(rot_rest, m, value);

					auto new_cnt = rest_cnt + cnt_m;
					if (new_cnt >= full_size)
					{
						_rest = from_raw_m512i<Packet>(_mm512_castps_si512(rot_rest));
						_value = from_raw_m512i<Packet>(_mm512_castps_si512(p1));
						full = true;
						return new_cnt - full_size;
					}
					else
					{
						_rest = from_raw_m512i<Packet>(_mm512_castps_si512(p1));
						full = false;
						return new_cnt;
					}
				}
			};
		}
	}
}
#endif
