/**
 * @file Basic.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.3.3
 * @date 2021-03-31
 * 
 * @copyright Copyright (c) 2020-2021
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

		/**
		 * @brief Base class of all univariate random generators
		 * 
		 * @tparam DerivedGen 
		 * @tparam Scalar 
		 */
		template<typename DerivedGen, typename Scalar>
		class GenBase
		{
		public:
			/**
			 * @brief generate random values from its distribution
			 * 
			 * @tparam Derived 
			 * @tparam Urng 
			 * @param rows the number of rows being generated
			 * @param cols the number of columns being generated
			 * @param urng c++11-style random number generator
			 * @return 
			 * a random matrix expression with a shape `(rows, cols)`
			 */
			template<typename Derived, typename Urng>
			inline const CwiseNullaryOp<internal::scalar_rng_adaptor<DerivedGen&, Scalar, Urng>, const Derived>
				generate(Index rows, Index cols, Urng&& urng)
			{
				return {
					rows, cols, { std::forward<Urng>(urng), static_cast<DerivedGen&>(*this) }
				};
			}

			/**
			 * @brief generate random values from its distribution
			 * 
			 * @tparam Derived 
			 * @tparam Urng 
			 * @param o an instance of any type of Eigen::DenseBase
			 * @param urng c++11-style random number generator
			 * @return 
			 * a random matrix expression of the same shape as `o`
			 */
			template<typename Derived, typename Urng>
			inline const CwiseNullaryOp<internal::scalar_rng_adaptor<DerivedGen&, Scalar, Urng>, const Derived>
				generateLike(const Derived& o, Urng&& urng)
			{
				return {
					o.rows(), o.cols(), { std::forward<Urng>(urng), static_cast<DerivedGen&>(*this) }
				};
			}
		};

		/**
		 * @brief Base class of all multivariate random vector generators
		 * 
		 * @tparam DerivedGen 
		 * @tparam _Scalar 
		 * @tparam Dim 
		 */
		template<typename DerivedGen, typename _Scalar, Index Dim>
		class MvVecGenBase
		{
		public:
			/**
			 * @brief returns the dimensions of vectors to be generated
			 */
			Index dims() const { return static_cast<DerivedGen&>(*this).dims(); }

			/**
			 * @brief generates multiple samples at once
			 * 
			 * @tparam Urng 
			 * @param urng c++11-style random number generator
			 * @param samples the number of samples to be generated
			 * @return
			 * a random matrix with a shape `(dim, samples)` which is consist of `samples` random vector columns
			 */
			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng, Index samples)
			{
				return static_cast<DerivedGen&>(*this).generatr(std::forward<Urng>(urng), samples);
			}

			/**
			 * @brief generates one sample
			 * 
			 * @tparam Urng 
			 * @param urng c++11-style random number generator
			 * @return a random vector with a shape `(dim,)`
			 */
			template<typename Urng>
			inline Matrix<_Scalar, Dim, 1> generate(Urng&& urng)
			{
				return static_cast<DerivedGen&>(*this).generatr(std::forward<Urng>(urng));
			}
		};

		/**
		 * @brief Base class of all multivariate random matrix generators
		 * 
		 * @tparam DerivedGen 
		 * @tparam _Scalar 
		 * @tparam Dim 
		 */
		template<typename DerivedGen, typename _Scalar, Index Dim>
		class MvMatGenBase
		{
		public:
			/**
			 * @brief returns the dimensions of matrices to be generated
			 */
			Index dims() const { return static_cast<DerivedGen&>(*this).dims(); }

			/**
			 * @brief generates multiple samples at once
			 * 
			 * @tparam Urng 
			 * @param urng c++11-style random number generator
			 * @param samples the number of samples to be generated
			 * @return
			 * a random matrix with a shape `(dim, dim * samples)` which is `samples` random matrices concatenated along the column axis
			 */
			template<typename Urng>
			inline Matrix<_Scalar, Dim, -1> generate(Urng&& urng, Index samples)
			{
				return static_cast<DerivedGen&>(*this).generate(std::forward<Urng>(urng), samples);
			}

			/**
			 * @brief generates one sample
			 * 
			 * @tparam Urng 
			 * @param urng c++11-style random number generator
			 * @return a random matrix with a shape `(dim, dim)`
			 */
			template<typename Urng>
			inline Matrix<_Scalar, Dim, Dim> generate(Urng&& urng)
			{
				return static_cast<DerivedGen&>(*this).generate(std::forward<Urng>(urng));
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
		 * @tparam _Scalar any integral type
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
		 * @tparam _Scalar any real type
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
		 * @tparam _Scalar any real type
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


		/**
		 * @brief Generator of Bernoulli distribution
		 *
		 * @tparam _Scalar
		 */
		template<typename _Scalar>
		class BernoulliGen : public GenBase<BernoulliGen<_Scalar>, _Scalar>
		{
			uint32_t p;
		public:
			using Scalar = _Scalar;

			BernoulliGen(double _p = 0.5)
			{
				eigen_assert(0 <= _p && _p <= 1 );
				p = (uint32_t)(_p * 0x80000000);
			}

			BernoulliGen(const BernoulliGen&) = default;
			BernoulliGen(BernoulliGen&&) = default;

			BernoulliGen& operator=(const BernoulliGen&) = default;
			BernoulliGen& operator=(BernoulliGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return (((uint32_t)pfirst(std::forward<Rng>(rng)()) & 0x7FFFFFFF) < p) ? 1 : 0;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using IPacket = decltype(reinterpret_to_int(std::declval<Packet>()));
				using RUtils = RawbitsMaker<IPacket, Rng>;
				auto one = pset1<Packet>(1);
				auto zero = pset1<Packet>(0);
				auto r = RUtils{}.rawbits(std::forward<Rng>(rng));
				r = pand(r, pset1<IPacket>(0x7FFFFFFF));
				return pblendv(pcmplt(r, pset1<IPacket>(p)), one, zero);
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
		 * 
		 * @see Eigen::Rand::RandbitsGen
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
		 * 
		 * @see Eigen::Rand::RandbitsGen
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
		 * 
		 * @see Eigen::Rand::BalancedGen
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
		 * 
		 * @see Eigen::Rand::BalancedGen
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
		 * 
		 * @see Eigen::Rand::UniformRealGen
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
		 * 
		 * @see Eigen::Rand::UniformRealGen
		 */
		template<typename Derived, typename Urng>
		inline const UniformRealType<Derived, Urng>
			uniformRealLike(Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng) }
			};
		}

		template<typename Derived, typename Urng>
		using BernoulliType = CwiseNullaryOp<internal::scalar_rng_adaptor<BernoulliGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates 1 with probability `p` and 0 with probability `1 - p`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param p a probability of generating 1
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 */
		template<typename Derived, typename Urng>
		inline const BernoulliType<Derived, Urng>
			bernoulli(Index rows, Index cols, Urng&& urng, double p = 0.5)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), BernoulliGen<typename Derived::Scalar>{ p } }
			};
		}

		/**
		 * @brief generates 1 with probability `p` and 0 with probability `1 - p`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param p a probability of generating 1
		 * @return a random matrix expression of the same shape as `o`
		 */
		template<typename Derived, typename Urng>
		inline const BernoulliType<Derived, Urng>
			bernoulli(Derived& o, Urng&& urng, double p = 0.5)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), BernoulliGen<typename Derived::Scalar>{ p } }
			};
		}
	}
}

#endif