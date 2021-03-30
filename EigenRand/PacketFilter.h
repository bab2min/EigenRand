/**
 * @file PacketFilter.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.3.3
 * @date 2021-03-31
 * 
 * @copyright Copyright (c) 2020-2021
 * 
 */

#ifndef EIGENRAND_PACKET_FILTER_H
#define EIGENRAND_PACKET_FILTER_H

#include <array>
#include <EigenRand/MorePacketMath.h>

namespace Eigen
{
	namespace Rand
	{
		namespace detail
		{
			template<size_t PacketSize>
			class CompressMask;
		}
	}
}
#ifdef EIGEN_VECTORIZE_AVX
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
					return _mm256_loadu_si256((internal::Packet8i*)ret);
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

				template<typename Packet>
				EIGEN_STRONG_INLINE int compress_append(Packet& _value, const Packet& _mask,
					Packet& _rest, int rest_cnt, bool& full) const
				{
					auto& value = reinterpret_cast<internal::Packet8f&>(_value);
					auto& mask = reinterpret_cast<const internal::Packet8f&>(_mask);
					auto& rest = reinterpret_cast<internal::Packet8f&>(_rest);

					int m = _mm256_movemask_ps(mask);
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

#ifdef EIGEN_VECTORIZE_SSE2
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
				std::array<uint8_t, 64> cnt;

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

#endif