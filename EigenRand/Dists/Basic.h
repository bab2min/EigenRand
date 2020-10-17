/**
 * @file Basic.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.3.0
 * @date 2020-10-07
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef EIGENRAND_DISTS_BASIC_H
#define EIGENRAND_DISTS_BASIC_H

namespace Eigen
{
	namespace Rand
	{
		namespace constant
		{
			static constexpr double pi = 3.1415926535897932;
			static constexpr double e = 2.7182818284590452;
		}

		template<typename DerivedGen, typename Scalar>
		class GenBase
		{
		public:
			template<typename Derived, typename Urng>
			inline const CwiseNullaryOp<internal::scalar_rng_adaptor<DerivedGen&, Scalar, Urng>, const Derived>
				generate(Index rows, Index cols, Urng&& urng)
			{
				return {
					rows, cols, { std::forward<Urng>(urng), static_cast<DerivedGen&>(*this) }
				};
			}

			template<typename Derived, typename Urng>
			inline const CwiseNullaryOp<internal::scalar_rng_adaptor<DerivedGen&, Scalar, Urng>, const Derived>
				generateLike(const Derived& o, Urng&& urng)
			{
				return {
					o.rows(), o.cols(), { std::forward<Urng>(urng), static_cast<DerivedGen&>(*this) }
				};
			}
		};

		template<Index _alignment=0>
		class CacheStore
		{
		protected:
			enum { max_size = sizeof(internal::find_best_packet<float, -1>::type) };
			int8_t raw_data[max_size + _alignment - 1] = { 0, };
			void* aligned_ptr;

		public:
			CacheStore()
			{
				aligned_ptr = (void*)((((size_t)raw_data + _alignment - 1) / _alignment) * _alignment);
			}

			CacheStore(const CacheStore& c)
			{
				std::copy(c.raw_data, c.raw_data + max_size, raw_data);
				aligned_ptr = (void*)((((size_t)raw_data + _alignment - 1) / _alignment) * _alignment);
			}

			CacheStore(CacheStore&& c)
			{
				std::copy(c.raw_data, c.raw_data + max_size, raw_data);
				aligned_ptr = (void*)((((size_t)raw_data + _alignment - 1) / _alignment) * _alignment);
			}

			template<typename Ty>
			Ty& get()
			{
				return *(Ty*)aligned_ptr;
			}

			template<typename Ty>
			const Ty& get() const
			{
				return *(const Ty*)aligned_ptr;
			}
		};

		template<>
		class CacheStore<0>
		{
		protected:
			enum { max_size = sizeof(internal::find_best_packet<float, -1>::type) };
			int8_t raw_data[max_size] = { 0, };

		public:
			CacheStore()
			{
			}

			CacheStore(const CacheStore& c)
			{
				std::copy(c.raw_data, c.raw_data + max_size, raw_data);
			}

			CacheStore(CacheStore&& c)
			{
				std::copy(c.raw_data, c.raw_data + max_size, raw_data);
			}

			template<typename Ty>
			Ty& get()
			{
				return *(Ty*)raw_data;
			}

			template<typename Ty>
			const Ty& get() const
			{
				return *(const Ty*)raw_data;
			}
		};

		using OptCacheStore = CacheStore<EIGEN_MAX_ALIGN_BYTES>;

		/**
		 * @brief Generator of random bits for integral scalars
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class RandbitsGen : public GenBase<RandbitsGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_integral<_Scalar>::value, "randBits needs integral types.");
		
		public:
			using Scalar = _Scalar;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return pfirst(std::forward<Rng>(rng)());
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RUtils = RawbitsMaker<Packet, Rng>;
				return RUtils{}.rawbits(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Generator of reals in a range `[-1, 1]`
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class BalancedGen : public GenBase<BalancedGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "balanced needs floating point types.");

		public:
			using Scalar = _Scalar;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return ((_Scalar)((int32_t)pfirst(std::forward<Rng>(rng)()) & 0x7FFFFFFF) / 0x7FFFFFFF) * 2 - 1;
			}

			template<typename Packet, typename Rng>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RUtils = RandUtils<Packet, Rng>;
				return RUtils{}.balanced(std::forward<Rng>(rng));
			}
		};

		/**
		 * @brief Generator of reals in a range `[0, 1)`
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class UniformRealGen : public GenBase<UniformRealGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "uniformReal needs floating point types.");

		public:
			using Scalar = _Scalar;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return bit_scalar<_Scalar>{}.to_ur(pfirst(std::forward<Rng>(rng)()));
			}

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar nzur_scalar(Rng&& rng)
			{
				using namespace Eigen::internal;
				return bit_scalar<_Scalar>{}.to_nzur(pfirst(std::forward<Rng>(rng)()));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RUtils = RandUtils<Packet, Rng>;
				return RUtils{}.uniform_real(std::forward<Rng>(rng));
			}
		};


		template<typename Derived, typename Urng>
		using RandBitsType = CwiseNullaryOp<internal::scalar_rng_adaptor<RandbitsGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates integers with random bits
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const RandBitsType<Derived, Urng>
			randBits(Index rows, Index cols, Urng&& urng)
		{
			return {
				rows, cols, { std::forward<Urng>(urng) }
			};
		}

		/**
		 * @brief generates integers with random bits
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const RandBitsType<Derived, Urng>
			randBitsLike(Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng) }
			};
		}


		template<typename Derived, typename Urng>
		using BalancedType = CwiseNullaryOp<internal::scalar_rng_adaptor<BalancedGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals in a range `[-1, 1]`
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const BalancedType<Derived, Urng>
			balanced(Index rows, Index cols, Urng&& urng)
		{
			return {
				rows, cols, { std::forward<Urng>(urng) }
			};
		}

		/**
		 * @brief generates reals in a range `[-1, 1]`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const BalancedType<Derived, Urng>
			balancedLike(const Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng) }
			};
		}

		template<typename Derived, typename Urng>
		using UniformRealType = CwiseNullaryOp<internal::scalar_rng_adaptor<UniformRealGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals in a range `[0, 1)`
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const UniformRealType<Derived, Urng>
			uniformReal(Index rows, Index cols, Urng&& urng)
		{
			return {
				rows, cols, { std::forward<Urng>(urng) }
			};
		}

		/**
		 * @brief generates reals in a range `[0, 1)`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const UniformRealType<Derived, Urng>
			uniformRealLike(Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng) }
			};
		}
	}
}

#endif