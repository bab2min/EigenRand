/**
* EigenRand
* Author: bab2min@gmail.com
* Date: 2020-06-22
*/

#ifndef EIGENRAND_PACKET_RANDOM_ENGINE_H
#define EIGENRAND_PACKET_RANDOM_ENGINE_H

#include <array>
#include <random>
#include <type_traits>
#include <EigenRand/MorePacketMath.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Ty>
		struct IsIntPacket : std::false_type {};

		template<typename Ty>
		struct HalfPacket;

#ifdef EIGEN_VECTORIZE_AVX2
		template<>
		struct IsIntPacket<Packet8i> : std::true_type {};

		template<>
		struct HalfPacket<Packet8i>
		{
			using type = Packet4i;
		};
#endif
#ifdef EIGEN_VECTORIZE_SSE2
		template<>
		struct IsIntPacket<Packet4i> : std::true_type {};

		template<>
		struct HalfPacket<Packet4i>
		{
			using type = uint64_t;
		};
#endif
		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pcmpeq64(const Packet& a, const Packet& b);

		template<typename Packet>
		EIGEN_STRONG_INLINE Packet pmuluadd64(const Packet& a, uint64_t b, uint64_t c);

#ifdef EIGEN_VECTORIZE_AVX
		template<>
		EIGEN_STRONG_INLINE Packet8i pcmpeq64<Packet8i>(const Packet8i& a, const Packet8i& b)
		{
#ifdef EIGEN_VECTORIZE_AVX2
			return _mm256_cmpeq_epi64(a, b);
#else
			Packet4i a1, a2, b1, b2;
			split_two(a, a1, a2);
			split_two(b, b1, b2);
			return combine_two(_mm_cmpeq_epi64(a1, b1), _mm_cmpeq_epi64(a2, b2));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet8i pmuluadd64<Packet8i>(const Packet8i& a, uint64_t b, uint64_t c)
		{
			uint64_t u[4];
			_mm256_storeu_si256((__m256i*)u, a);
			u[0] = u[0] * b + c;
			u[1] = u[1] * b + c;
			u[2] = u[2] * b + c;
			u[3] = u[3] * b + c;
			return _mm256_loadu_si256((__m256i*)u);
		}
#endif

#ifdef EIGEN_VECTORIZE_SSE2
		template<>
		EIGEN_STRONG_INLINE Packet4i pcmpeq64<Packet4i>(const Packet4i& a, const Packet4i& b)
		{
#ifdef EIGEN_VECTORIZE_SSE4_1
			return _mm_cmpeq_epi64(a, b);
#else
			Packet4i c = _mm_cmpeq_epi32(a, b);
			return pand(c, _mm_shuffle_epi32(c, _MM_SHUFFLE(2, 3, 0, 1)));
#endif
		}

		template<>
		EIGEN_STRONG_INLINE Packet4i pmuluadd64<Packet4i>(const Packet4i& a, uint64_t b, uint64_t c)
		{
			uint64_t u[2];
			_mm_storeu_si128((__m128i*)u, a);
			u[0] = u[0] * b + c;
			u[1] = u[1] * b + c;
			return _mm_loadu_si128((__m128i*)u);
		}

#endif
	}

	namespace Rand
	{
		namespace detail
		{
			template<typename T>
			auto test_integral_result_type(int) -> std::integral_constant<bool, std::is_integral<typename T::result_type>::value>;

			template<typename T>
			auto test_integral_result_type(...) -> std::false_type;

			template<typename T>
			auto test_intpacket_result_type(int)->std::integral_constant<bool, internal::IsIntPacket<typename T::result_type>::value>;

			template<typename T>
			auto test_intpacket_result_type(...)->std::false_type;
		}

		template<typename Ty>
		struct IsScalarRandomEngine : decltype(detail::test_integral_result_type<Ty>(0))
		{
		};

		template<typename Ty>
		struct IsPacketRandomEngine : decltype(detail::test_intpacket_result_type<Ty>(0))
		{
		};

		enum class RandomEngineType
		{
			none, scalar, packet
		};

		template<typename Ty>
		struct GetRandomEngineType : std::integral_constant <
			RandomEngineType,
			IsPacketRandomEngine<Ty>::value ? RandomEngineType::packet :
			(IsScalarRandomEngine<Ty>::value ? RandomEngineType::scalar : RandomEngineType::none)
		>
		{
		};

#ifndef EIGEN_DONT_VECTORIZE
		template<typename Packet,
			int _Nx, int _Mx,
			int _Rx, uint64_t _Px,
			int _Ux, uint64_t _Dx,
			int _Sx, uint64_t _Bx,
			int _Tx, uint64_t _Cx,
			int _Lx, uint64_t _Fx>
		class MersenneTwister
		{
		public:
			using result_type = Packet;

			static constexpr int word_size = 64;
			static constexpr int state_size = _Nx;
			static constexpr int shift_size = _Mx;
			static constexpr int mask_bits = _Rx;
			static constexpr uint64_t parameter_a = _Px;
			static constexpr int output_u = _Ux;
			static constexpr int output_s = _Sx;
			static constexpr uint64_t output_b = _Bx;
			static constexpr int output_t = _Tx;
			static constexpr uint64_t output_c = _Cx;
			static constexpr int output_l = _Lx;

			static constexpr uint64_t default_seed = 5489U;

			MersenneTwister(uint64_t x0 = default_seed)
			{
				using namespace Eigen::internal;
				std::array<uint64_t, unpacket_traits<Packet>::size / 2> seeds;
				for (uint64_t i = 0; i < seeds.size(); ++i)
				{
					seeds[i] = x0 + i;
				}
				seed(ploadu<Packet>((int*)seeds.data()));
			}

			MersenneTwister(Packet x0)
			{
				seed(x0);
			}

			void seed(Packet x0)
			{
				using namespace Eigen::internal;
				Packet prev = state[0] = x0;
				for (int i = 1; i < _Nx; ++i)
				{
					prev = state[i] = pmuluadd64(pxor(prev, psrl64(prev, word_size - 2)), _Fx, i);
				}
				stateIdx = _Nx;
			}

			uint64_t min() const
			{
				return 0;
			}

			uint64_t max() const
			{
				return _wMask;
			}

			result_type operator()()
			{
				if (stateIdx == _Nx)
					refill_upper();
				else if (2 * _Nx <= stateIdx)
					refill_lower();

				using namespace Eigen::internal;

				Packet res = state[stateIdx++];
				res = pxor(res, pand(psrl64(res, _Ux), pseti64<Packet>(_Dx)));
				res = pxor(res, pand(psll64(res, _Sx), pseti64<Packet>(_Bx)));
				res = pxor(res, pand(psll64(res, _Tx), pseti64<Packet>(_Cx)));
				res = pxor(res, psrl64(res, _Lx));
				return res;
			}

			void discard(unsigned long long _Nskip)
			{
				for (; 0 < _Nskip; --_Nskip)
				{
					operator()();
				}
			}

			typename internal::HalfPacket<Packet>::type half()
			{
				if (valid)
				{
					valid = false;
					return cache;
				}
				typename internal::HalfPacket<Packet>::type a;
				internal::split_two(operator()(), a, cache);
				valid = true;
				return a;
			}

		protected:

			void refill_lower()
			{
				using namespace Eigen::internal;

				auto hmask = pseti64<Packet>(_hMask),
					lmask = pseti64<Packet>(_lMask),
					px = pseti64<Packet>(_Px),
					one = pseti64<Packet>(1);

				int i;
				for (i = 0; i < _Nx - _Mx; ++i)
				{
					Packet tmp = por(pand(state[i + _Nx], hmask),
						pand(state[i + _Nx + 1], lmask));
					
					state[i] = pxor(pxor(
						psrl64(tmp, 1),
						pand(pcmpeq64(pand(tmp, one), one), px)),
						state[i + _Nx + _Mx]
					);
				}

				for (; i < _Nx - 1; ++i)
				{
					Packet tmp = por(pand(state[i + _Nx], hmask),
						pand(state[i + _Nx + 1], lmask));

					state[i] = pxor(pxor(
						psrl64(tmp, 1),
						pand(pcmpeq64(pand(tmp, one), one), px)),
						state[i - _Nx + _Mx]
					);
				}

				Packet tmp = por(pand(state[i + _Nx], hmask), 
					pand(state[0], lmask));
				state[i] = pxor(pxor(
					psrl64(tmp, 1),
					pand(pcmpeq64(pand(tmp, one), one), px)),
					state[_Mx - 1]
				);
				stateIdx = 0;
			}

			void refill_upper()
			{
				using namespace Eigen::internal;

				auto hmask = pseti64<Packet>(_hMask),
					lmask = pseti64<Packet>(_lMask),
					px = pseti64<Packet>(_Px),
					one = pseti64<Packet>(1);

				for (int i = _Nx; i < 2 * _Nx; ++i)
				{
					Packet tmp = por(pand(state[i - _Nx], hmask),
						pand(state[i - _Nx + 1], lmask));

					state[i] = pxor(pxor(
						psrl64(tmp, 1),
						pand(pcmpeq64(pand(tmp, one), one), px)),
						state[i - _Nx + _Mx]
					);
				}
			}

			std::array<Packet, _Nx * 2> state;
			size_t stateIdx = 0;
			typename internal::HalfPacket<Packet>::type cache;
			bool valid = false;

			static constexpr uint64_t _wMask = (uint64_t)-1;
			static constexpr uint64_t _hMask = (_wMask << _Rx) & _wMask;
			static constexpr uint64_t _lMask = ~_hMask & _wMask;
		};

		template<typename Packet>
		using pmt19937_64 = MersenneTwister<Packet, 312, 156, 31,
			0xb5026f5aa96619e9, 29,
			0x5555555555555555, 17,
			0x71d67fffeda60000, 37,
			0xfff7eee000000000, 43, 6364136223846793005>;
#endif

		template<typename UIntType, typename BaseRng>
		class PacketRandomEngineAdaptor
		{
			static_assert(IsPacketRandomEngine<BaseRng>::value, "BaseRNG must be a kind of PacketRandomEngine.");
		public:
			using result_type = UIntType;

			PacketRandomEngineAdaptor(const BaseRng& _rng)
				: rng{ _rng }
			{
			}
			
			PacketRandomEngineAdaptor(BaseRng&& _rng)
				: rng{ _rng }
			{
			}

			PacketRandomEngineAdaptor(const PacketRandomEngineAdaptor&) = default;
			PacketRandomEngineAdaptor(PacketRandomEngineAdaptor&&) = default;

			static constexpr result_type min()
			{
				return std::numeric_limits<result_type>::min();
			}

			static constexpr result_type max()
			{
				return std::numeric_limits<result_type>::max();
			}

			result_type operator()()
			{
				if (cnt >= buf_size)
				{
					refill_buffer();
				}
				return buf[cnt++];
			}

		private:
			static constexpr size_t buf_size = 64 / sizeof(result_type);

			void refill_buffer()
			{
				cnt = 0;
				const size_t stride = sizeof(typename BaseRng::result_type) / sizeof(result_type);
				for (size_t i = 0; i < buf_size; i += stride)
				{
					*(typename BaseRng::result_type*)&buf[i] = rng();
				}
			}

			BaseRng rng;
			std::array<result_type, buf_size> buf;
			size_t cnt = buf_size;

		};

		template<typename UIntType, typename Rng> 
		typename std::enable_if<
			IsPacketRandomEngine<typename std::remove_reference<Rng>::type>::value, 
			PacketRandomEngineAdaptor<UIntType, typename std::remove_reference<Rng>::type>
		>::type makeScalarRng(Rng&& rng)
		{
			return { std::forward<Rng>(rng) };
		}

		template<typename UIntType, typename Rng>
		typename std::enable_if<
			IsScalarRandomEngine<typename std::remove_reference<Rng>::type>::value,
			typename std::remove_reference<Rng>::type
		>::type makeScalarRng(Rng&& rng)
		{
			return std::forward<Rng>(rng);
		}

#ifdef EIGEN_VECTORIZE_AVX2
		using vmt19937_64 = pmt19937_64<internal::Packet8i>;
#elif defined(EIGEN_VECTORIZE_AVX) || defined(EIGEN_VECTORIZE_SSE2)
		using vmt19937_64 = pmt19937_64<internal::Packet4i>;
#else
		using vmt19937_64 = std::mt19937_64;
#endif

	}
}

#endif
