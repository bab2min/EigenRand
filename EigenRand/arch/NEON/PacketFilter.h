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

#ifndef EIGENRAND_PACKET_FILTER_NEON_H
#define EIGENRAND_PACKET_FILTER_NEON_H

#include <arm_neon.h>

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

					uint32_t v[4] = { 0, };

					selector[0] = (internal::Packet4f)vreinterpretq_f32_u32(vld1q_u32(v));
					v[0] = -1;
					selector[1] = (internal::Packet4f)vreinterpretq_f32_u32(vld1q_u32(v));
					v[1] = -1;
					selector[2] = (internal::Packet4f)vreinterpretq_f32_u32(vld1q_u32(v));
					v[2] = -1;
					selector[3] = (internal::Packet4f)vreinterpretq_f32_u32(vld1q_u32(v));
				}

				static EIGEN_STRONG_INLINE internal::Packet4f permute(const internal::Packet4f& p, uint8_t i)
				{
					float u[4];
					vst1q_f32(u, p);
					float t[4];
					t[0] = u[i & 3];
					t[1] = u[(i >> 2) & 3];
					t[2] = u[(i >> 4) & 3];
					t[3] = u[(i >> 6) & 3];
					return vld1q_f32(t);
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

					int m = internal::pmovemask(mask);
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
