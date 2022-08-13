/**
 * @file Basic.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.4.1
 * @date 2022-08-13
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

		template<typename _Scalar>
		struct ExtractFirstUint;

		template<>
		struct ExtractFirstUint<float>
		{
			template<typename Packet>
			auto operator()(Packet v) -> decltype(Eigen::internal::pfirst(v))
			{
				return Eigen::internal::pfirst(v);
			}
		};

		template<>
		struct ExtractFirstUint<double>
		{
			template<typename Packet>
			auto operator()(Packet v) -> uint64_t
			{
				uint64_t arr[sizeof(Packet) / 8];
				Eigen::internal::pstoreu((Packet*)arr, v);
				return arr[0];
			}
		};

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
		 * @brief Generator of reals in a range `[a, b]`
		 *
		 * @tparam _Scalar any real type
		 */
		template<typename _Scalar>
		class Balanced2Gen : public GenBase<Balanced2Gen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "balanced needs floating point types.");
			_Scalar slope = 2, bias = -1;
		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new balanced generator
			 *
			 * @param _a,_b left and right boundary
			 */
			Balanced2Gen(_Scalar _a = -1, _Scalar _b = 1)
				: slope{ _b - _a }, bias{ _a }
			{
			}

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return ((_Scalar)((int32_t)pfirst(std::forward<Rng>(rng)()) & 0x7FFFFFFF) / 0x7FFFFFFF) * slope + bias;
			}

			template<typename Packet, typename Rng>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RUtils = RandUtils<Packet, Rng>;
				return RUtils{}.balanced(std::forward<Rng>(rng), slope, bias);
			}
		};

		namespace detail
		{
			template<size_t v>
			struct BitWidth
			{
				static constexpr size_t value = BitWidth<v / 2>::value + 1;
			};

			template<>
			struct BitWidth<0>
			{
				static constexpr size_t value = 0;
			};

			template<class Rng>
			struct RngBitSize
			{
				static constexpr size_t _min = Rng::min();
				static constexpr size_t _max = Rng::max();

				static constexpr bool _fullbit_rng = _min == 0 && (_max & (_max + 1)) == 0;
				static constexpr size_t value = IsPacketRandomEngine<Rng>::value ? sizeof(typename Rng::result_type) * 8 : (_fullbit_rng ? BitWidth<_max>::value : 0);
			};
		}

		/**
		 * @brief Generator of reals in a range `[0, 1)`
		 * 
		 * @tparam _Scalar any real type
		 */
		template<typename _Scalar>
		class StdUniformRealGen : public GenBase<StdUniformRealGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "uniformReal needs floating point types.");

		public:
			using Scalar = _Scalar;

			template<typename Rng, 
				typename std::enable_if<sizeof(Scalar) * 8 <= detail::RngBitSize<typename std::remove_const<typename std::remove_reference<Rng>::type>::type>::value, int>::type = 0
			>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return BitScalar<_Scalar>{}.to_ur(ExtractFirstUint<_Scalar>{}(std::forward<Rng>(rng)()));
			}

			template<typename Rng,
				typename std::enable_if<detail::RngBitSize<typename std::remove_const<typename std::remove_reference<Rng>::type>::type>::value < sizeof(Scalar) * 8, int>::type = 0
			>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using RRng = typename std::remove_const<typename std::remove_reference<Rng>::type>::type;
				static_assert(detail::RngBitSize<RRng>::value > 0,
					"BaseRng must be a kind of mersenne_twister_engine.");
				using ResultType = typename std::conditional<detail::RngBitSize<RRng>::value == 32, uint32_t, uint64_t>::type;
				using namespace Eigen::internal;
				ResultType arr[sizeof(Scalar) / sizeof(ResultType)];
				for (size_t i = 0; i < sizeof(Scalar) / sizeof(ResultType); ++i)
				{
					arr[i] = rng();
				}
				return BitScalar<_Scalar>{}.to_ur(*(uint64_t*)arr);
			}

			template<typename Rng,
				typename std::enable_if<sizeof(Scalar) <= sizeof(typename std::remove_const<typename std::remove_reference<Rng>::type>::type::result_type), int>::type = 0
			>
			EIGEN_STRONG_INLINE const _Scalar nzur_scalar(Rng&& rng)
			{
				using namespace Eigen::internal;
				return BitScalar<_Scalar>{}.to_nzur(ExtractFirstUint<_Scalar>{}(std::forward<Rng>(rng)()));
			}

			template<typename Rng,
				typename std::enable_if<sizeof(typename std::remove_const<typename std::remove_reference<Rng>::type>::type::result_type) < sizeof(Scalar), int > ::type = 0
			>
			EIGEN_STRONG_INLINE const _Scalar nzur_scalar(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RngResult = typename std::remove_const<typename std::remove_reference<Rng>::type>::type::result_type;
				RngResult arr[sizeof(Scalar) / sizeof(RngResult)];
				for (size_t i = 0; i < sizeof(Scalar) / sizeof(RngResult); ++i)
				{
					arr[i] = rng();
				}
				return BitScalar<_Scalar>{}.to_nzur(*(Scalar*)arr);
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
		 * @brief Generator of reals in a range `[a, b)`
		 * 
		 * @tparam _Scalar any real type
		 */
		template<typename _Scalar>
		class UniformRealGen : public GenBase<UniformRealGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "uniformReal needs floating point types.");
			_Scalar bias, slope;

		public:
			using Scalar = _Scalar;

			UniformRealGen(_Scalar _min = 0, _Scalar _max = 1)
				: bias{ _min }, slope{ _max - _min }
			{
			}

			UniformRealGen(const UniformRealGen&) = default;
			UniformRealGen(UniformRealGen&&) = default;

			UniformRealGen& operator=(const UniformRealGen&) = default;
			UniformRealGen& operator=(UniformRealGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return bias + BitScalar<_Scalar>{}.to_ur(pfirst(std::forward<Rng>(rng)())) * slope;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RUtils = RandUtils<Packet, Rng>;
				return padd(pmul(
					RUtils{}.uniform_real(std::forward<Rng>(rng)), pset1<Packet>(slope)
				), pset1<Packet>(bias));
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
		using Balanced2Type = CwiseNullaryOp<internal::scalar_rng_adaptor<Balanced2Gen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals in a range `[a, b]`
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a,b left and right boundary
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 *
		 * @see Eigen::Rand::BalancedGen
		 */
		template<typename Derived, typename Urng>
		inline const Balanced2Type<Derived, Urng>
			balanced(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a, typename Derived::Scalar b)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), Balanced2Gen<typename Derived::Scalar>{a, b} }
			};
		}

		/**
		 * @brief generates reals in a range `[a, b]`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a,b left and right boundary
		 * @return a random matrix expression of the same shape as `o`
		 *
		 * @see Eigen::Rand::BalancedGen
		 */
		template<typename Derived, typename Urng>
		inline const Balanced2Type<Derived, Urng>
			balancedLike(const Derived& o, Urng&& urng, typename Derived::Scalar a, typename Derived::Scalar b)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), Balanced2Gen<typename Derived::Scalar>{a, b} }
			};
		}

		template<typename Derived, typename Urng>
		using StdUniformRealType = CwiseNullaryOp<internal::scalar_rng_adaptor<StdUniformRealGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

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
		inline const StdUniformRealType<Derived, Urng>
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
		inline const StdUniformRealType<Derived, Urng>
			uniformRealLike(Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng) }
			};
		}

		template<typename Derived, typename Urng>
		using UniformRealType = CwiseNullaryOp<internal::scalar_rng_adaptor<UniformRealGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals in a range `[min, max)`
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param min, max the range of reals being generated
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 *
		 * @see Eigen::Rand::UniformRealGen
		 */
		template<typename Derived, typename Urng>
		inline const UniformRealType<Derived, Urng>
			uniformReal(Index rows, Index cols, Urng&& urng, typename Derived::Scalar min, typename Derived::Scalar max)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), UniformRealGen<typename Derived::Scalar>{ min, max } }
			};
		}

		/**
		 * @brief generates reals in a range `[min, max)`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param min, max the range of reals being generated
		 * @return a random matrix expression of the same shape as `o`
		 *
		 * @see Eigen::Rand::UniformRealGen
		 */
		template<typename Derived, typename Urng>
		inline const UniformRealType<Derived, Urng>
			uniformRealLike(Derived& o, Urng&& urng, typename Derived::Scalar min, typename Derived::Scalar max)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), UniformRealGen<typename Derived::Scalar>{ min, max } }
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