/**
 * @file Basic.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.2.0
 * @date 2020-06-22
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef EIGENRAND_DISTS_BASIC_H
#define EIGENRAND_DISTS_BASIC_H

namespace Eigen
{
	namespace internal
	{
		namespace constant
		{
			static constexpr double pi = 3.1415926535897932;
			static constexpr double e = 2.7182818284590452;
		}

		template<typename Scalar, typename Rng>
		struct scalar_randbits_op : public scalar_base_rng<Scalar, Rng>
		{
			static_assert(std::is_integral<Scalar>::value, "randBits needs integral types.");

			using scalar_base_rng<Scalar, Rng>::scalar_base_rng;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return pfirst(this->rng());
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RawbitsMaker<Packet, Rng>;
				return RUtils{}.rawbits(this->rng);
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_randbits_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};


		template<typename Scalar, typename Rng>
		struct scalar_uniform_real_op : public scalar_base_rng<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "uniformReal needs floating point types.");

			using scalar_base_rng<Scalar, Rng>::scalar_base_rng;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return bit_scalar<Scalar>{}.to_ur(pfirst(this->rng()));
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar nzur_scalar() const
			{
				return bit_scalar<Scalar>{}.to_nzur(pfirst(this->rng()));
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RandUtils<Packet, Rng>;
				return RUtils{}.uniform_real(this->rng);
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_uniform_real_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Rng>
		struct scalar_balanced_op : public scalar_base_rng<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "balanced needs floating point types.");

			using scalar_base_rng<Scalar, Rng>::scalar_base_rng;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return ((Scalar)((int32_t)pfirst(this->rng()) & 0x7FFFFFFF) / 0x7FFFFFFF) * 2 - 1;
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RandUtils<Packet, Rng>;
				return RUtils{}.balanced(this->rng);
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_balanced_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};
	}
}

#endif