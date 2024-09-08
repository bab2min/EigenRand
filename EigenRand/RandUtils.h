/**
 * @file RandUtils.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.5.1
 * @date 2024-09-08
 *
 * @copyright Copyright (c) 2020-2024
 * 
 */

#ifndef EIGENRAND_RAND_UTILS_H
#define EIGENRAND_RAND_UTILS_H

#include "MorePacketMath.h"
#include "PacketFilter.h"
#include "PacketRandomEngine.h"

namespace Eigen
{
	namespace internal
	{
		template<typename Packet, typename Rng,
			typename RngResult = typename std::remove_reference<Rng>::type::result_type,
			Rand::RandomEngineType reType = Rand::GetRandomEngineType<
			typename std::remove_reference<Rng>::type
		>::value>
			struct RawbitsMaker;

		template<typename PacketType, typename Rng>
		struct UniformRealUtils;

		template<typename PacketType, typename Rng>
		struct RandUtils : public UniformRealUtils<PacketType, Rng>
		{
			EIGEN_STRONG_INLINE PacketType balanced(Rng& rng)
			{
				return psub(pmul(this->zero_to_one(rng), pset1<PacketType>(2)), pset1<PacketType>(1));
			}

			template<typename Scalar>
			EIGEN_STRONG_INLINE PacketType balanced(Rng& rng, Scalar slope, Scalar bias)
			{
				return padd(pmul(this->zero_to_one(rng), pset1<PacketType>(slope)), pset1<PacketType>(bias));
			}

			EIGEN_STRONG_INLINE PacketType nonzero_uniform_real(Rng& rng)
			{
				constexpr auto epsilon = std::numeric_limits<typename unpacket_traits<PacketType>::type>::epsilon() / 8;
				return padd(this->uniform_real(rng), pset1<PacketType>(epsilon));
			}
		};

		template<typename Gen, typename _Scalar, typename Rng, bool _mutable = false>
		struct scalar_rng_adaptor
		{
			static_assert(
				Rand::IsScalarFullBitRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value,
				"Rng must satisfy RandomNumberEngine"
				);

			Gen gen;
			Rng rng;

			scalar_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_rng_adaptor(const scalar_rng_adaptor& o) = default;
			scalar_rng_adaptor(scalar_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() () const
			{
				return gen(rng);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				return gen.template packetOp<Packet>(rng);
			}
		};

		template<typename Gen, typename _Scalar, typename Rng>
		struct scalar_rng_adaptor<Gen, _Scalar, Rng, true>
		{
			static_assert(
				Rand::IsScalarFullBitRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value,
				"Rng must satisfy RandomNumberEngine"
				);

			mutable Gen gen;
			Rng rng;

			scalar_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_rng_adaptor(const scalar_rng_adaptor& o) = default;
			scalar_rng_adaptor(scalar_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() () const
			{
				return gen(rng);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				return gen.template packetOp<Packet>(rng);
			}
		};

		template<typename Gen, typename _Scalar, typename Urng, bool _mutable>
		struct functor_traits<scalar_rng_adaptor<Gen, _Scalar, Urng, _mutable> >
		{
			enum { Cost = HugeCost, PacketAccess = std::is_floating_point<_Scalar>::value ? packet_traits<_Scalar>::HasExp : packet_traits<_Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Gen, typename _Scalar, typename _ScalarA, typename Rng, bool _mutable = false>
		struct scalar_unary_rng_adaptor
		{
			static_assert(
				Rand::IsScalarFullBitRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value,
				"Rng must satisfy RandomNumberEngine"
				);

			Gen gen;
			Rng rng;

			scalar_unary_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_unary_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_unary_rng_adaptor(const scalar_unary_rng_adaptor& o) = default;
			scalar_unary_rng_adaptor(scalar_unary_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() (_ScalarA a) const
			{
				return gen(rng, a);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
			{
				return gen.template packetOp<Packet>(rng, a);
			}
		};

		template<typename Gen, typename _Scalar, typename _ScalarA, typename Rng>
		struct scalar_unary_rng_adaptor<Gen, _Scalar, _ScalarA, Rng, true>
		{
			static_assert(
				Rand::IsScalarFullBitRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value,
				"Rng must satisfy RandomNumberEngine"
				);

			mutable Gen gen;
			Rng rng;

			scalar_unary_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_unary_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_unary_rng_adaptor(const scalar_unary_rng_adaptor& o) = default;
			scalar_unary_rng_adaptor(scalar_unary_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() (_ScalarA a) const
			{
				return gen(rng, a);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a) const
			{
				return gen.template packetOp<Packet>(rng, a);
			}
		};

		template<typename Gen, typename _Scalar, typename _ScalarA, typename Urng, bool _mutable>
		struct functor_traits<scalar_unary_rng_adaptor<Gen, _Scalar, _ScalarA, Urng, _mutable> >
		{
			enum { Cost = HugeCost, PacketAccess = std::is_floating_point<_Scalar>::value ? packet_traits<_Scalar>::HasExp : packet_traits<_Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Gen, typename _Scalar, typename _ScalarA, typename _ScalarB, typename Rng, bool _mutable = false>
		struct scalar_binary_rng_adaptor
		{
			static_assert(
				Rand::IsScalarFullBitRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value,
				"Rng must satisfy RandomNumberEngine"
				);

			Gen gen;
			Rng rng;

			scalar_binary_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_binary_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_binary_rng_adaptor(const scalar_binary_rng_adaptor& o) = default;
			scalar_binary_rng_adaptor(scalar_binary_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() (_ScalarA a, _ScalarB b) const
			{
				return gen(rng, a, b);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(const Packet& a, const Packet& b) const
			{
				return gen.template packetOp<Packet>(rng, a, b);
			}
		};

		template<typename Gen, typename _Scalar, typename _ScalarA, typename _ScalarB, typename Rng>
		struct scalar_binary_rng_adaptor<Gen, _Scalar, _ScalarA, _ScalarB, Rng, true>
		{
			static_assert(
				Rand::IsScalarFullBitRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value,
				"Rng must satisfy RandomNumberEngine"
				);

			mutable Gen gen;
			Rng rng;

			scalar_binary_rng_adaptor(const Rng& _rng) : rng{ _rng }
			{
			}

			template<typename _Gen>
			scalar_binary_rng_adaptor(const Rng& _rng, _Gen&& _gen) : gen{ _gen }, rng{ _rng }
			{
			}

			scalar_binary_rng_adaptor(const scalar_binary_rng_adaptor& o) = default;
			scalar_binary_rng_adaptor(scalar_binary_rng_adaptor&& o) = default;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const _Scalar operator() (_ScalarA a, _ScalarB b) const
			{
				return gen(rng, a, b);
			}

			template<typename PacketA, typename PacketB>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const PacketA packetOp(const PacketA& a, const PacketB& b) const
			{
				return gen.packetOp(rng, a, b);
			}
		};

		template<typename Gen, typename _Scalar, typename _ScalarA, typename _ScalarB, typename Urng, bool _mutable>
		struct functor_traits<scalar_binary_rng_adaptor<Gen, _Scalar, _ScalarA, _ScalarB, Urng, _mutable> >
		{
			enum { Cost = HugeCost, PacketAccess = std::is_floating_point<_Scalar>::value ? packet_traits<_Scalar>::HasExp : packet_traits<_Scalar>::Vectorizable, IsRepeatable = false };
		};
	}
}

#ifdef EIGEN_VECTORIZE_AVX512
#include "arch/AVX512/RandUtils.h"
#endif

#ifdef EIGEN_VECTORIZE_AVX
#include "arch/AVX/RandUtils.h"
#endif

#ifdef EIGEN_VECTORIZE_SSE2
#include "arch/SSE/RandUtils.h"
#endif

#ifdef EIGEN_VECTORIZE_NEON
#include "arch/NEON/RandUtils.h"
#endif


namespace Eigen
{
	namespace internal
	{
		EIGEN_STRONG_INLINE uint32_t collect_upper8bits(uint32_t a, uint32_t b, uint32_t c)
		{
			return ((a & 0xFF000000) >> 24) | ((b & 0xFF000000) >> 16) | ((c & 0xFF000000) >> 8);
		}
	}
}

#if defined(DEBUG) || defined(_DEBUG)
	#define EIGENRAND_CHECK_INFINITY_LOOP() do { assert(_i < 100); } while(0)
#else
	#define EIGENRAND_CHECK_INFINITY_LOOP() 
#endif

#endif
