/**
 * @file PacketFilter.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.4.1
 * @date 2022-08-13
 *
 * @copyright Copyright (c) 2020-2021
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
					_mm_storeu_ps(u, p);
					return _mm_setr_ps(u[i & 3], u[(i >> 2) & 3], u[(i >> 4) & 3], u[(i >> 6) & 3]);
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
					auto& value = reinterpret_cast<internal::Packet4f&>(_value);
					auto& mask = reinterpret_cast<const internal::Packet4f&>(_mask);
					auto& rest = reinterpret_cast<internal::Packet4f&>(_rest);

					int m = _mm_movemask_ps(mask);
					if (cnt[m] == full_size)
					{
						full = true;
						return rest_cnt;
					}

					auto p1 = permute(value, idx[rest_cnt][m]);
					p1 = internal::pblendv(selector[rest_cnt], rest, p1);

					auto new_cnt = rest_cnt + cnt[m];
					if (new_cnt >= full_size)
					{
						if (new_cnt > full_size)
						{
							rest = permute(value, idx[new_cnt - cnt[m] + full_size - 1][m]);
						}
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
