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

#ifndef EIGENRAND_PACKET_FILTER_AVX_H
#define EIGENRAND_PACKET_FILTER_AVX_H

#include <immintrin.h>

namespace Eigen
{
	namespace Rand
	{
		namespace detail
		{
			template<>
			class CompressMask<32>
			{
				std::array<std::array<internal::Packet8i, 256>, 15> idx;
				std::array<internal::Packet8f, 8> selector;
				std::array<uint8_t, 256> cnt;

				static internal::Packet8i make_compress(int mask, int offset = 0)
				{
					int32_t ret[8] = { 0, };
					int n = offset;
					for (int i = 0; i < 8; ++i)
					{
						int l = mask & 1;
						mask >>= 1;
						if (l)
						{
							if (n >= 0) ret[n] = i;
							if (++n >= 8) break;
						}
					}
					return _mm256_loadu_si256((__m256i const*)ret);
				}

				static uint8_t count(int mask)
				{
					uint8_t ret = 0;
					for (int i = 0; i < 8; ++i)
					{
						ret += mask & 1;
						mask >>= 1;
					}
					return ret;
				}

				CompressMask()
				{
					for (int i = 0; i < 256; ++i)
					{
						for (int o = 0; o < 15; ++o)
						{
							idx[o][i] = make_compress(i, o < 8 ? o : o - 15);
						}

						cnt[i] = count(i);
					}

					selector[0] = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, 0, 0, 0, 0, 0));
					selector[1] = _mm256_castsi256_ps(_mm256_setr_epi32(-1, 0, 0, 0, 0, 0, 0, 0));
					selector[2] = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0));
					selector[3] = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, 0, 0, 0, 0, 0));
					selector[4] = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0));
					selector[5] = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, 0, 0, 0));
					selector[6] = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0));
					selector[7] = _mm256_castsi256_ps(_mm256_setr_epi32(-1, -1, -1, -1, -1, -1, -1, 0));
				}

				static EIGEN_STRONG_INLINE internal::Packet8f permute(const internal::Packet8f& p, const internal::Packet8i& i)
				{
#ifdef EIGEN_VECTORIZE_AVX2
					return _mm256_permutevar8x32_ps(p, i);
#else
					auto l = _mm256_permutevar_ps(p, i);
					auto h = _mm256_permutevar_ps(_mm256_permute2f128_ps(p, p, 0x01), i);
					internal::Packet4i i1, i2;
					internal::split_two(i, i1, i2);
					i1 = _mm_slli_epi32(i1, 29);
					i2 = _mm_slli_epi32(i2, 29);
					auto c = _mm256_castsi256_ps(
						internal::combine_two(
							_mm_cmplt_epi32(i1, internal::pset1<internal::Packet4i>(0)),
							_mm_cmplt_epi32(internal::pset1<internal::Packet4i>(-1), i2)
						)
					);
					return internal::pblendv(c, h, l);
#endif
				}

			public:
				enum { full_size = 8 };
				static const CompressMask& get_inst()
				{
					static CompressMask cm;
					return cm;
				}

				// Helper to extract raw __m256i from packet types (works with both
				// Eigen 3.x raw types and Eigen 5.x eigen_packet_wrapper types)
				template<typename Packet>
				static EIGEN_STRONG_INLINE __m256i to_raw_m256i(const Packet& p)
				{
					return reinterpret_cast<const __m256i&>(p);
				}

				// Helper to convert raw __m256i back to Packet type
				template<typename Packet>
				static EIGEN_STRONG_INLINE Packet from_raw_m256i(__m256i v)
				{
					return reinterpret_cast<const Packet&>(v);
				}

				static EIGEN_STRONG_INLINE __m256 permute_raw(const __m256& p, const internal::Packet8i& i)
				{
#ifdef EIGEN_VECTORIZE_AVX2
					return _mm256_permutevar8x32_ps(p, __m256i(i));
#else
					auto l = _mm256_permutevar_ps(p, __m256i(i));
					auto h = _mm256_permutevar_ps(_mm256_permute2f128_ps(p, p, 0x01), __m256i(i));
					__m128i i_raw = _mm256_castsi256_si128(__m256i(i));
					__m128i i1 = i_raw;
					__m128i i2 = _mm256_extractf128_si256(__m256i(i), 1);
					i1 = _mm_slli_epi32(i1, 29);
					i2 = _mm_slli_epi32(i2, 29);
					auto c = _mm256_castsi256_ps(
						_mm256_insertf128_si256(
							_mm256_castsi128_si256(_mm_cmplt_epi32(i1, _mm_set1_epi32(0))),
							_mm_cmplt_epi32(_mm_set1_epi32(-1), i2),
							1
						)
					);
					return _mm256_blendv_ps(l, h, c);
#endif
				}

				template<typename Packet>
				EIGEN_STRONG_INLINE int compress_append(Packet& _value, const Packet& _mask,
					Packet& _rest, int rest_cnt, bool& full) const
				{
					// Convert to raw __m256 for SIMD operations
					__m256 value = _mm256_castsi256_ps(to_raw_m256i(_value));
					__m256 mask_ps = _mm256_castsi256_ps(to_raw_m256i(_mask));
					__m256 rest_val = _mm256_castsi256_ps(to_raw_m256i(_rest));

					int m = _mm256_movemask_ps(mask_ps);
					if (cnt[m] == full_size)
					{
						full = true;
						return rest_cnt;
					}

					auto p1 = permute_raw(value, idx[rest_cnt][m]);
					p1 = _mm256_blendv_ps(p1, rest_val, _mm256_castsi256_ps(to_raw_m256i(selector[rest_cnt])));

					auto new_cnt = rest_cnt + cnt[m];
					if (new_cnt >= full_size)
					{
						if (new_cnt > full_size)
						{
							rest_val = permute_raw(value, idx[new_cnt - cnt[m] + full_size - 1][m]);
							_rest = from_raw_m256i<Packet>(_mm256_castps_si256(rest_val));
						}
						_value = from_raw_m256i<Packet>(_mm256_castps_si256(p1));
						full = true;
						return new_cnt - full_size;
					}
					else
					{
						_rest = from_raw_m256i<Packet>(_mm256_castps_si256(p1));
						full = false;
						return new_cnt;
					}
				}
			};
		}
	}
}
#endif
