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

#ifndef EIGENRAND_PACKET_FILTER_SSE_H
#define EIGENRAND_PACKET_FILTER_SSE_H

#include <xmmintrin.h>

namespace Eigen
{
	namespace Rand
	{
		namespace detail
		{
			template<>
			class CompressMask<16>
			{
				std::array<std::array<uint8_t, 16>, 7> idx;
				std::array<internal::Packet4f, 4> selector;
				std::array<uint8_t, 16> cnt;

				static uint8_t make_compress(int mask, int offset = 0)
				{
					uint8_t ret = 0;
					int n = offset;
					for (int i = 0; i < 4; ++i)
					{
						int l = mask & 1;
						mask >>= 1;
						if (l)
						{
							if (n >= 0) ret |= (i & 3) << (2 * n);
							if (++n >= 4) break;
						}
					}
					return ret;
				}

				static uint8_t count(int mask)
				{
					uint8_t ret = 0;
					for (int i = 0; i < 4; ++i)
					{
						ret += mask & 1;
						mask >>= 1;
					}
					return ret;
				}

				CompressMask()
				{
					for (int i = 0; i < 16; ++i)
					{
						for (int o = 0; o < 7; ++o)
						{
							idx[o][i] = make_compress(i, o < 4 ? o : o - 7);
						}

						cnt[i] = count(i);
					}

					selector[0] = _mm_castsi128_ps(_mm_setr_epi32(0, 0, 0, 0));
					selector[1] = _mm_castsi128_ps(_mm_setr_epi32(-1, 0, 0, 0));
					selector[2] = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, 0, 0));
					selector[3] = _mm_castsi128_ps(_mm_setr_epi32(-1, -1, -1, 0));
				}

				static EIGEN_STRONG_INLINE internal::Packet4f permute(const internal::Packet4f& p, uint8_t i)
				{
					float u[4];
					_mm_storeu_ps(u, __m128(p));
					return _mm_setr_ps(u[i & 3], u[(i >> 2) & 3], u[(i >> 4) & 3], u[(i >> 6) & 3]);
				}

				static EIGEN_STRONG_INLINE __m128 permute_raw(const __m128& p, uint8_t i)
				{
					float u[4];
					_mm_storeu_ps(u, p);
					return _mm_setr_ps(u[i & 3], u[(i >> 2) & 3], u[(i >> 4) & 3], u[(i >> 6) & 3]);
				}

				// Helper to extract raw __m128i from packet types (works with both
				// Eigen 3.x raw types and Eigen 5.x eigen_packet_wrapper types)
				template<typename Packet>
				static EIGEN_STRONG_INLINE __m128i to_raw_m128i(const Packet& p)
				{
					return reinterpret_cast<const __m128i&>(p);
				}

				// Helper to convert raw __m128i back to Packet type
				template<typename Packet>
				static EIGEN_STRONG_INLINE Packet from_raw_m128i(__m128i v)
				{
					return reinterpret_cast<const Packet&>(v);
				}

			public:

				enum { full_size = 4 };

				static const CompressMask& get_inst()
				{
					static CompressMask cm;
					return cm;
				}

				template<typename Packet>
				EIGEN_STRONG_INLINE int compress_append(Packet& _value, const Packet& _mask,
					Packet& _rest, int rest_cnt, bool& full) const
				{
					// Convert to raw __m128 for SIMD operations
					__m128 value = _mm_castsi128_ps(to_raw_m128i(_value));
					__m128 mask_ps = _mm_castsi128_ps(to_raw_m128i(_mask));
					__m128 rest_val = _mm_castsi128_ps(to_raw_m128i(_rest));

					int m = _mm_movemask_ps(mask_ps);
					if (cnt[m] == full_size)
					{
						full = true;
						return rest_cnt;
					}

					auto p1 = permute_raw(value, idx[rest_cnt][m]);
					p1 = _mm_blendv_ps(p1, rest_val, _mm_castsi128_ps(to_raw_m128i(selector[rest_cnt])));

					auto new_cnt = rest_cnt + cnt[m];
					if (new_cnt >= full_size)
					{
						if (new_cnt > full_size)
						{
							rest_val = permute_raw(value, idx[new_cnt - cnt[m] + full_size - 1][m]);
							_rest = from_raw_m128i<Packet>(_mm_castps_si128(rest_val));
						}
						_value = from_raw_m128i<Packet>(_mm_castps_si128(p1));
						full = true;
						return new_cnt - full_size;
					}
					else
					{
						_rest = from_raw_m128i<Packet>(_mm_castps_si128(p1));
						full = false;
						return new_cnt;
					}
				}
			};
		}
	}
}
#endif
