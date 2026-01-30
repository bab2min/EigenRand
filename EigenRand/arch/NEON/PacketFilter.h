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
					vst1q_f32(u, float32x4_t(p));
					float t[4];
					t[0] = u[i & 3];
					t[1] = u[(i >> 2) & 3];
					t[2] = u[(i >> 4) & 3];
					t[3] = u[(i >> 6) & 3];
					return vld1q_f32(t);
				}

				static EIGEN_STRONG_INLINE float32x4_t permute_raw(const float32x4_t& p, uint8_t i)
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

				// Helper to extract raw int32x4_t from packet types (works with both
				// Eigen 3.x raw types and Eigen 5.x eigen_packet_wrapper types)
				template<typename Packet>
				static EIGEN_STRONG_INLINE int32x4_t to_raw_int32x4(const Packet& p)
				{
					return int32x4_t(p);
				}

				// Helper to convert raw int32x4_t back to Packet type
				template<typename Packet>
				static EIGEN_STRONG_INLINE Packet from_raw_int32x4(int32x4_t v)
				{
					return Packet(v);
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
					// Convert to raw float32x4_t for NEON operations
					float32x4_t value = vreinterpretq_f32_s32(to_raw_int32x4(_value));
					float32x4_t mask_f = vreinterpretq_f32_s32(to_raw_int32x4(_mask));
					float32x4_t rest_val = vreinterpretq_f32_s32(to_raw_int32x4(_rest));

					int m = internal::pmovemask(internal::Packet4f(mask_f));
					if (cnt[m] == full_size)
					{
						full = true;
						return rest_cnt;
					}
					auto p1 = permute_raw(value, idx[rest_cnt][m]);
					// pblendv for NEON
					float32x4_t sel = float32x4_t(selector[rest_cnt]);
					uint32x4_t sel_mask = vreinterpretq_u32_f32(sel);
					p1 = vbslq_f32(sel_mask, rest_val, p1);

					auto new_cnt = rest_cnt + cnt[m];
					if (new_cnt >= full_size)
					{
						if (new_cnt > full_size)
						{
							rest_val = permute_raw(value, idx[new_cnt - cnt[m] + full_size - 1][m]);
							_rest = from_raw_int32x4<Packet>(vreinterpretq_s32_f32(rest_val));
						}
						_value = from_raw_int32x4<Packet>(vreinterpretq_s32_f32(p1));
						full = true;
						return new_cnt - full_size;
					}
					else
					{
						_rest = from_raw_int32x4<Packet>(vreinterpretq_s32_f32(p1));
						full = false;
						return new_cnt;
					}
				}
			};
		}
	}
}
#endif
