/**
 * @file PacketRandomEngine.h
 * @author bab2min (bab2min@gmail.com)
 * @brief
 * @version 0.4.1
 * @date 2022-08-13
 *
 * @copyright Copyright (c) 2020-2021
 *
 */

#ifndef EIGENRAND_PACKET_RANDOM_ENGINE_H
#define EIGENRAND_PACKET_RANDOM_ENGINE_H

#include <array>
#include <random>
#include <type_traits>
#include "MorePacketMath.h"
#include <fstream>

namespace Eigen
{
	namespace Rand
	{
		namespace detail
		{
			template<typename T>
			auto test_integral_result_type(int)->std::integral_constant<bool, std::is_integral<typename T::result_type>::value && !(T::min() == 0 && (T::max() & T::max() + 1) == 0)>;

			template<typename T>
			auto test_integral_result_type(...)->std::false_type;

			template<typename T>
			auto test_integral_fullbit_result_type(int)->std::integral_constant<bool, std::is_integral<typename T::result_type>::value&& T::min() == 0 && (T::max() & T::max() + 1) == 0>;

			template<typename T>
			auto test_integral_fullbit_result_type(...)->std::false_type;

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
		struct IsScalarFullBitRandomEngine : decltype(detail::test_integral_fullbit_result_type<Ty>(0))
		{
		};

		template<typename Ty>
		struct IsPacketRandomEngine : decltype(detail::test_intpacket_result_type<Ty>(0))
		{
		};

		enum class RandomEngineType
		{
			none, scalar, scalar_fullbit, packet
		};

		template<typename Ty>
		struct GetRandomEngineType : std::integral_constant <
			RandomEngineType,
			IsPacketRandomEngine<Ty>::value ? RandomEngineType::packet :
			IsScalarFullBitRandomEngine<Ty>::value ? RandomEngineType::scalar_fullbit :
			IsScalarRandomEngine<Ty>::value ? RandomEngineType::scalar : RandomEngineType::none
		>
		{
		};

		template<typename Ty, size_t length, size_t alignment = 64>
		class AlignedArray
		{
		public:
			AlignedArray()
			{
				allocate();
				for (size_t i = 0; i < length; ++i)
				{
					new (&aligned[i]) Ty();
				}
			}

			AlignedArray(const AlignedArray& o)
			{
				allocate();
				for (size_t i = 0; i < length; ++i)
				{
					aligned[i] = o[i];
				}
			}

			AlignedArray(AlignedArray&& o)
			{
				std::swap(memory, o.memory);
				std::swap(aligned, o.aligned);
			}

			AlignedArray& operator=(const AlignedArray& o)
			{
				for (size_t i = 0; i < length; ++i)
				{
					aligned[i] = o[i];
				}
				return *this;
			}

			AlignedArray& operator=(AlignedArray&& o)
			{
				std::swap(memory, o.memory);
				std::swap(aligned, o.aligned);
				return *this;
			}

			~AlignedArray()
			{
				deallocate();
			}

			Ty& operator[](size_t i)
			{
				return aligned[i];
			}

			const Ty& operator[](size_t i) const
			{
				return aligned[i];
			}

			size_t size() const
			{
				return length;
			}

			Ty* data()
			{
				return aligned;
			}

			const Ty* data() const
			{
				return aligned;
			}

		private:
			void allocate()
			{
				memory = std::malloc(sizeof(Ty) * length + alignment);
				aligned = (Ty*)(((size_t)memory + alignment) & ~(alignment - 1));
			}

			void deallocate()
			{
				if (memory)
				{
					for (size_t i = 0; i < length; ++i)
					{
						aligned[i].~Ty();
					}
					std::free(memory);
					memory = nullptr;
					aligned = nullptr;
				}
			}

			void* memory = nullptr;
			Ty* aligned = nullptr;
		};

#ifndef EIGEN_DONT_VECTORIZE
		/**
		 * @brief A vectorized version of Mersenne Twister Engine
		 *
		 * @tparam Packet a type of integer packet being generated from this engine
		 * @tparam _Nx
		 * @tparam _Mx
		 * @tparam _Rx
		 * @tparam _Px
		 * @tparam _Ux
		 * @tparam _Dx
		 * @tparam _Sx
		 * @tparam _Bx
		 * @tparam _Tx
		 * @tparam _Cx
		 * @tparam _Lx
		 * @tparam _Fx
		 *
		 * @note It is recommended to use the alias, Eigen::Rand::Vmt19937_64 rather than using raw MersenneTwister template class
		 * because the definition of Eigen::Rand::Vmt19937_64 is changed to use the appropriate PacketType depending on compile options and the architecture of machines.
		 */
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

			/**
			 * @brief Construct a new Mersenne Twister engine with a scalar seed
			 *
			 * @param x0 scalar seed for the engine
			 *
			 * @note The seed for the first element of packet is initialized to `x0`,
			 * for the second element to `x0 + 1`, and the n-th element to is `x0 + n - 1`.
			 */
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

			/**
			 * @brief Construct a new Mersenne Twister engine with a packet seed
			 *
			 * @param x0 packet seed for the engine
			 */
			MersenneTwister(Packet x0)
			{
				seed(x0);
			}

			/**
			 * @brief initialize the engine with a given seed
			 *
			 * @param x0 packet seed for the engine
			 */
			void seed(Packet x0)
			{
				using namespace Eigen::internal;
				Packet prev = state[0] = x0;
				for (int i = 1; i < _Nx; ++i)
				{
					prev = state[i] = pmuluadd64(pxor(prev, psrl64<word_size - 2>(prev)), _Fx, i);
				}
				stateIdx = _Nx;
			}

			/**
			 * @brief minimum value of the result
			 *
			 * @return uint64_t
			 */
			static constexpr uint64_t min()
			{
				return 0;
			}

			/**
			 * @brief maximum value of the result
			 *
			 * @return uint64_t
			 */
			static constexpr uint64_t max()
			{
				return _wMask;
			}

			/**
			 * @brief Generates one random packet and advance the internal state.
			 *
			 * @return result_type
			 *
			 * @note A value generated from this engine is not scalar, but packet type.
			 * If you need to extract scalar values, use Eigen::Rand::makeScalarRng or Eigen::Rand::PacketRandomEngineAdaptor.
			 */
			result_type operator()()
			{
				if (stateIdx == _Nx)
					refill_upper();
				else if (2 * _Nx <= stateIdx)
					refill_lower();

				using namespace Eigen::internal;

				Packet res = state[stateIdx++];
				res = pxor(res, pand(psrl64<_Ux>(res), pseti64<Packet>(_Dx)));
				res = pxor(res, pand(psll64<_Sx>(res), pseti64<Packet>(_Bx)));
				res = pxor(res, pand(psll64<_Tx>(res), pseti64<Packet>(_Cx)));
				res = pxor(res, psrl64<_Lx>(res));
				return res;
			}

			/**
			 * @brief Discards `num` items being generated
			 *
			 * @param num the number of items being discarded
			 */
			void discard(unsigned long long num)
			{
				for (; 0 < num; --num)
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
						psrl64<1>(tmp),
						pand(pcmpeq64(pand(tmp, one), one), px)),
						state[i + _Nx + _Mx]
					);
				}

				for (; i < _Nx - 1; ++i)
				{
					Packet tmp = por(pand(state[i + _Nx], hmask),
						pand(state[i + _Nx + 1], lmask));

					state[i] = pxor(pxor(
						psrl64<1>(tmp),
						pand(pcmpeq64(pand(tmp, one), one), px)),
						state[i - _Nx + _Mx]
					);
				}

				Packet tmp = por(pand(state[i + _Nx], hmask),
					pand(state[0], lmask));
				state[i] = pxor(pxor(
					psrl64<1>(tmp),
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
						psrl64<1>(tmp),
						pand(pcmpeq64(pand(tmp, one), one), px)),
						state[i - _Nx + _Mx]
					);
				}
			}

			AlignedArray<Packet, _Nx * 2> state;
			size_t stateIdx = 0;
			typename internal::HalfPacket<Packet>::type cache;
			bool valid = false;

			static constexpr uint64_t _wMask = (uint64_t)-1;
			static constexpr uint64_t _hMask = (_wMask << _Rx) & _wMask;
			static constexpr uint64_t _lMask = ~_hMask & _wMask;
		};

		/**
		 * @brief Alias of Eigen::Rand::MersenneTwister, equivalent to std::mt19937_64
		 *
		 * @tparam Packet
		 */
		template<typename Packet>
		using Pmt19937_64 = MersenneTwister<Packet, 312, 156, 31,
			0xb5026f5aa96619e9, 29,
			0x5555555555555555, 17,
			0x71d67fffeda60000, 37,
			0xfff7eee000000000, 43, 6364136223846793005>;
#endif

		template<typename UIntType, typename BaseRng, int numU64>
		class ParallelRandomEngineAdaptor
		{
			static_assert(GetRandomEngineType<BaseRng>::value != RandomEngineType::none, "BaseRng must be a kind of Random Engine.");
			static_assert(GetRandomEngineType<BaseRng>::value != RandomEngineType::scalar, "BaseRng must be a kind of mersenne_twister_engine.");
		public:
			using result_type = UIntType;

			ParallelRandomEngineAdaptor(size_t seed = BaseRng::default_seed)
			{
				for (int i = 0; i < num_parallel; ++i)
				{
					rngs[i].~BaseRng();
					new (&rngs[i]) BaseRng{ seed + i * u64_stride };
				}
			}

			ParallelRandomEngineAdaptor(const BaseRng& o)
			{
				for (int i = 0; i < num_parallel; ++i)
				{
					rngs[i].~BaseRng();
					new (&rngs[i]) BaseRng{ o };
				}
			}

			ParallelRandomEngineAdaptor(const ParallelRandomEngineAdaptor&) = default;
			ParallelRandomEngineAdaptor(ParallelRandomEngineAdaptor&&) = default;

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

			float uniform_real()
			{
				if (fcnt >= fbuf_size)
				{
					refill_fbuffer();
				}
				return fbuf[fcnt++];
			}
		private:

			void refill_buffer()
			{
				cnt = 0;
				for (size_t i = 0; i < num_parallel; ++i)
				{
					reinterpret_cast<typename BaseRng::result_type&>(buf[i * result_type_stride]) = rngs[i]();
				}
			}

			void refill_fbuffer()
			{
				fcnt = 0;
				for (size_t i = 0; i < num_parallel; ++i)
				{
					auto urf = internal::bit_to_ur_float(rngs[i]());
					reinterpret_cast<decltype(urf)&>(fbuf[i * u64_stride * 2]) = urf;
				}
			}

			static constexpr int u64_stride = sizeof(typename BaseRng::result_type) / sizeof(uint64_t);
			static constexpr int result_type_stride = sizeof(typename BaseRng::result_type) / sizeof(result_type);
			static constexpr int num_parallel = numU64 / u64_stride;
			static constexpr int byte_size = sizeof(uint64_t) * numU64;
			static constexpr size_t buf_size = byte_size / sizeof(result_type);
			static constexpr size_t fbuf_size = byte_size / sizeof(float);

			std::array<BaseRng, num_parallel> rngs;
			AlignedArray<result_type, buf_size> buf;
			AlignedArray<float, fbuf_size> fbuf;
			size_t cnt = buf_size, fcnt = fbuf_size;
		};

		/**
		 * @brief Scalar adaptor for random engines which generates packet
		 *
		 * @tparam UIntType scalar integer type for `result_type` of an adapted random number engine
		 * @tparam BaseRng
		 */
		template<typename UIntType, typename BaseRng>
		using PacketRandomEngineAdaptor = ParallelRandomEngineAdaptor<UIntType, BaseRng,
			sizeof(typename BaseRng::result_type) / sizeof(uint64_t)>;

		template<typename BaseRng>
		class RandomEngineWrapper : public BaseRng
		{
		public:
			using BaseRng::BaseRng;

			RandomEngineWrapper(const BaseRng& o) : BaseRng{ o }
			{
			}

			RandomEngineWrapper(BaseRng&& o) : BaseRng{ o }
			{
			}

			RandomEngineWrapper(size_t seed) : BaseRng{ seed }
			{
			}

			RandomEngineWrapper() = default;
			RandomEngineWrapper(const RandomEngineWrapper&) = default;
			RandomEngineWrapper(RandomEngineWrapper&&) = default;

			float uniform_real()
			{
				internal::BitScalar<float> bs;
				return bs.to_ur(this->operator()());
			}
		};

		template<typename UIntType, typename Rng>
		using UniversalRandomEngine = typename std::conditional<
			IsPacketRandomEngine<typename std::remove_reference<Rng>::type>::value,
			PacketRandomEngineAdaptor<UIntType, typename std::remove_reference<Rng>::type>,
			typename std::conditional<
			IsScalarFullBitRandomEngine<typename std::remove_reference<Rng>::type>::value,
			RandomEngineWrapper<typename std::remove_reference<Rng>::type>,
			void
			>::type
		>::type;

		/**
		 * @brief Helper function for making a UniversalRandomEngine
		 *
		 * @tparam UIntType
		 * @tparam Rng
		 * @param rng any random number engine for either packet or scalar type
		 * @return an instance of PacketRandomEngineAdaptor for UIntType
		 */
		template<typename UIntType, typename Rng>
		UniversalRandomEngine<UIntType, Rng> makeUniversalRng(Rng&& rng)
		{
			static_assert(IsPacketRandomEngine<typename std::remove_reference<Rng>::type>::value || IsScalarFullBitRandomEngine<typename std::remove_reference<Rng>::type>::value,
				"`Rng` must be a kind of RandomPacketEngine like std::mersenne_twister_engine");
			return { std::forward<Rng>(rng) };
		}

#ifdef EIGEN_VECTORIZE_AVX2
		using Vmt19937_64 = Pmt19937_64<internal::Packet8i>;
#elif defined(EIGEN_VECTORIZE_AVX) || defined(EIGEN_VECTORIZE_SSE2) || defined(EIGEN_VECTORIZE_NEON)
		using Vmt19937_64 = Pmt19937_64<internal::Packet4i>;
#else
		/**
		 * @brief same as std::mt19937_64 when EIGEN_DONT_VECTORIZE,
		 * Pmt19937_64<internal::Packet4i> when SSE2 enabled
		 * and Pmt19937_64<internal::Packet8i> when AVX2 enabled
		 *
		 * @note It yields the same random sequence only within the same seed and the same SIMD ISA.
		 * If you want to keep the same random sequence across different SIMD ISAs, use P8_mt19937_64.
		 */
		using Vmt19937_64 = std::mt19937_64;
#endif
		template<typename UIntType = uint64_t>
		using P8_mt19937 = ParallelRandomEngineAdaptor<UIntType, Vmt19937_64, 8>;

		/**
		 * @brief a vectorized mt19937_64 which generates 8 integers of 64bit simultaneously.
		 * It always yields the same value regardless of SIMD ISA.
		 */
		using P8_mt19937_64 = P8_mt19937<uint64_t>;

		using P8_mt19937_64_32 = P8_mt19937<uint32_t>;
	}
}

#endif
