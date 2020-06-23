#ifndef EIGENRAND_CORE_H
#define EIGENRAND_CORE_H

#include <EigenRand/MorePacketMath.h>
#include <EigenRand/PacketFilter.h>
#include <EigenRand/PacketRandomEngine.h>

namespace Eigen
{
	namespace internal
	{
		template<typename IntPacket>
		inline auto bit_to_ur_float(const IntPacket& x) -> decltype(reinterpret_to_float(x))
		{
			using FloatPacket = decltype(reinterpret_to_float(x));

			const IntPacket lower = pset1<IntPacket>(0x7FFFFF),
				upper = pset1<IntPacket>(127 << 23);
			const FloatPacket one = pset1<FloatPacket>(1);

			return psub(reinterpret_to_float(por(pand(x, lower), upper)), one);
		}

		template<typename IntPacket>
		inline auto bit_to_ur_double(const IntPacket& x) -> decltype(reinterpret_to_double(x))
		{
			using DoublePacket = decltype(reinterpret_to_double(x));

			const IntPacket lower = pseti64<IntPacket>(0xFFFFFFFFFFFFFull),
				upper = pseti64<IntPacket>(1023ull << 52);
			const DoublePacket one = pset1<DoublePacket>(1);

			return psub(reinterpret_to_double(por(pand(x, lower), upper)), one);
		}

		template<typename Scalar>
		struct bit_scalar;

		template<>
		struct bit_scalar<float>
		{
			float to_ur(uint32_t x)
			{
				union
				{
					uint32_t u;
					float f;
				};
				u = (x & 0x7FFFFF) | (127 << 23);
				return f - 1.f;
			}

			float to_nzur(uint32_t x)
			{
				return to_ur(x) + std::numeric_limits<float>::epsilon() / 8;
			}
		};

		template<>
		struct bit_scalar<double>
		{
			double to_ur(uint64_t x)
			{
				union
				{
					uint64_t u;
					double f;
				};
				u = (x & 0xFFFFFFFFFFFFFull) | (1023ull << 52);
				return f - 1.;
			}

			double to_nzur(uint64_t x)
			{
				return to_ur(x) + std::numeric_limits<double>::epsilon() / 8;
			}
		};

		template<typename Packet, typename Rng,
			typename RngResult = typename std::remove_reference<Rng>::type::result_type,
			Rand::RandomEngineType reType = Rand::GetRandomEngieType<
			typename std::remove_reference<Rng>::type
		>::value>
			struct RawbitsMaker;

		template<typename PacketType, typename Rng>
		struct UniformRealUtils;

		template<typename PacketType, typename Rng>
		struct RandUtils : public UniformRealUtils<PacketType, Rng>
		{
			PacketType balanced(Rng& rng)
			{
				return psub(pmul(this->zero_to_one(rng), pset1<PacketType>(2)), pset1<PacketType>(1));
			}

			PacketType nonzero_uniform_real(Rng& rng)
			{
				constexpr auto epsilon = std::numeric_limits<typename unpacket_traits<PacketType>::type>::epsilon() / 8;
				return padd(this->uniform_real(rng), pset1<PacketType>(epsilon));
			}
		};

		inline uint32_t collect_upper8bits(uint32_t a, uint32_t b, uint32_t c)
		{
			return ((a & 0xFF000000) >> 24) | ((b & 0xFF000000) >> 16) | ((c & 0xFF000000) >> 8);
		}
	}
}

#ifdef EIGEN_VECTORIZE_AVX
#include <immintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Rng, typename RngResult>
		struct RawbitsMaker<Packet4i, Rng, RngResult, Rand::RandomEngineType::scalar>
		{
			Packet4i rawbits(Rng& rng)
			{
				if (sizeof(RngResult) == 8)
				{
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), rng(), rng());
				}
			}

			Packet4i rawbits_34(Rng& rng)
			{
				if (sizeof(RngResult) == 8)
				{
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
					Packet4i p = _mm_setr_epi32(rng(), rng(), rng(), 0);
					return _mm_shuffle_epi8(p, _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11));
				}
			}

			Packet4i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm_set_epi64x(rng(), 0);
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), 0, 0);
				}
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet4i, Rng, Packet8i, Rand::RandomEngineType::packet>
		{
			Packet4i rawbits(Rng& rng)
			{
				return rng.half();
			}

			Packet4i rawbits_34(Rng& rng)
			{
				return rng.half();
			}

			Packet4i rawbits_half(Rng& rng)
			{
				return rng.half();
			}
		};

		template<typename Rng, typename RngResult>
		struct RawbitsMaker<Packet8i, Rng, RngResult, Rand::RandomEngineType::scalar>
		{
			Packet8i rawbits(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm256_set_epi64x(rng(), rng(), rng(), rng());
				}
				else
				{
					return _mm256_set_epi32(rng(), rng(), rng(), rng(),
						rng(), rng(), rng(), rng());
				}
			}

			Packet8i rawbits_34(Rng& rng)
			{
				Packet8i p;
				if (sizeof(decltype(rng())) == 8)
				{
#ifdef EIGEN_VECTORIZE_AVX2
					p = _mm256_setr_epi64x(rng(), rng(), rng(), 0);
					p = _mm256_permutevar8x32_epi32(p, _mm256_setr_epi32(0, 1, 2, 7, 3, 4, 5, 7));
					p = _mm256_shuffle_epi8(p, _mm256_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11,
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11
					));

#else
					auto v = rng();
					p = _mm256_setr_epi64x(rng(), v, rng(), v >> 32);
					Packet4i p1, p2, o = _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11);
					split_two(p, p1, p2);
					p = combine_two(_mm_shuffle_epi8(p1, o), _mm_shuffle_epi8(p2, o));
#endif
				}
				else
				{
					p = _mm256_setr_epi32(rng(), rng(), rng(), 0, rng(), rng(), rng(), 0);
#ifdef EIGEN_VECTORIZE_AVX2
					p = _mm256_shuffle_epi8(p, _mm256_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11,
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11
					));
#else
					Packet4i p1, p2, o = _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11);
					split_two(p, p1, p2);
					p = combine_two(_mm_shuffle_epi8(p1, o), _mm_shuffle_epi8(p2, o));
#endif
				}
				return p;
			}

			Packet4i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), rng(), rng());
				}
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet8i, Rng, Packet8i, Rand::RandomEngineType::packet>
		{
			Packet8i rawbits(Rng& rng)
			{
				return rng();
			}

			Packet8i rawbits_34(Rng& rng)
			{
				return rng();
			}

			Packet4i rawbits_half(Rng& rng)
			{
				return rng.half();
			}
		};

#ifndef EIGEN_VECTORIZE_AVX2
		template<>
		inline Packet8f bit_to_ur_float<Packet8i>(const Packet8i& x)
		{
			const Packet4i lower = pset1<Packet4i>(0x7FFFFF),
				upper = pset1<Packet4i>(127 << 23);
			const Packet8f one = pset1<Packet8f>(1);

			Packet4i x1, x2;
			split_two(x, x1, x2);

			return psub(reinterpret_to_float(
				combine_two(por(pand(x1, lower), upper), por(pand(x2, lower), upper)
				)), one);
		}

		template<>
		inline Packet4d bit_to_ur_double<Packet8i>(const Packet8i& x)
		{
			const Packet4i lower = pseti64<Packet4i>(0xFFFFFFFFFFFFFull),
				upper = pseti64<Packet4i>(1023ull << 52);
			const Packet4d one = pset1<Packet4d>(1);

			Packet4i x1, x2;
			split_two(x, x1, x2);

			return psub(reinterpret_to_double(
				combine_two(por(pand(x1, lower), upper), por(pand(x2, lower), upper)
				)), one);
		}
#endif

		template<typename Rng>
		struct UniformRealUtils<Packet8f, Rng> : public RawbitsMaker<Packet8i, Rng>
		{
			Packet8f zero_to_one(Rng& rng)
			{
				return pdiv(_mm256_cvtepi32_ps(pand(this->rawbits(rng), pset1<Packet8i>(0x7FFFFFFF))),
					pset1<Packet8f>(0x7FFFFFFF));
			}

			Packet8f uniform_real(Rng& rng)
			{
				return bit_to_ur_float(this->rawbits_34(rng));
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet4d, Rng> : public RawbitsMaker<Packet8i, Rng>
		{
			Packet4d zero_to_one(Rng& rng)
			{
				return pdiv(_mm256_cvtepi32_pd(pand(this->rawbits_half(rng), pset1<Packet4i>(0x7FFFFFFF))),
					pset1<Packet4d>(0x7FFFFFFF));
			}

			Packet4d uniform_real(Rng& rng)
			{
				return bit_to_ur_double(this->rawbits(rng));
			}
		};
	}
}

#elif defined(EIGEN_VECTORIZE_SSE2)
#include <xmmintrin.h>

namespace Eigen
{
	namespace internal
	{
		template<typename Rng, typename RngResult>
		struct RawbitsMaker<Packet4i, Rng, RngResult, Rand::RandomEngineType::scalar>
		{
			Packet4i rawbits(Rng& rng)
			{
				if (sizeof(RngResult) == 8)
				{
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), rng(), rng());
				}
			}

			Packet4i rawbits_34(Rng& rng)
			{
				if (sizeof(RngResult) == 8)
				{
					return _mm_set_epi64x(rng(), rng());
				}
				else
				{
#ifdef EIGEN_VECTORIZE_SSSE3
					Packet4i p = _mm_setr_epi32(rng(), rng(), rng(), 0);
					return _mm_shuffle_epi8(p, _mm_setr_epi8(
						0, 1, 2, 3,
						4, 5, 6, 7,
						8, 9, 10, 11,
						3, 7, 11, 11));
#else
					return _mm_set_epi32(rng(), rng(), rng(), rng());
#endif
				}
			}

			Packet4i rawbits_half(Rng& rng)
			{
				if (sizeof(decltype(rng())) == 8)
				{
					return _mm_set_epi64x(rng(), 0);
				}
				else
				{
					return _mm_set_epi32(rng(), rng(), 0, 0);
				}
			}
		};

		template<typename Rng>
		struct RawbitsMaker<Packet4i, Rng, Packet4i, Rand::RandomEngineType::packet>
		{
			Packet4i rawbits(Rng& rng)
			{
				return rng();
			}

			Packet4i rawbits_34(Rng& rng)
			{
				return rng();
			}

			Packet4i rawbits_half(Rng& rng)
			{
				return rng();
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet4f, Rng> : public RawbitsMaker<Packet4i, Rng>
		{
			Packet4f zero_to_one(Rng& rng)
			{
				return pdiv(_mm_cvtepi32_ps(pand(this->rawbits(rng), pset1<Packet4i>(0x7FFFFFFF))),
					pset1<Packet4f>(0x7FFFFFFF));
			}

			Packet4f uniform_real(Rng& rng)
			{
				return bit_to_ur_float(this->rawbits_34(rng));
			}
		};

		template<typename Rng>
		struct UniformRealUtils<Packet2d, Rng> : public RawbitsMaker<Packet4i, Rng>
		{
			Packet2d zero_to_one(Rng& rng)
			{
				return pdiv(_mm_cvtepi32_pd(pand(this->rawbits_half(rng), pset1<Packet4i>(0x7FFFFFFF))),
					pset1<Packet2d>(0x7FFFFFFF));
			}

			Packet2d uniform_real(Rng& rng)
			{
				return bit_to_ur_double(this->rawbits(rng));
			}
		};
	}
}
#endif

namespace Eigen
{
	namespace internal
	{
		template<typename Scalar, typename Rng>
		struct scalar_base_rng
		{
			static_assert(
				Rand::IsScalarRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value ||
				Rand::IsPacketRandomEngine<
				typename std::remove_reference<Rng>::type
				>::value, "Rng must satisfy RandomNumberEngine");

			Rng rng;

			scalar_base_rng(const Rng& _rng) : rng{ _rng }
			{
			}

			scalar_base_rng(const scalar_base_rng& o)
				: rng{ o.rng }
			{
			}

			scalar_base_rng(scalar_base_rng&& o)
				: rng{ std::move(o.rng) }
			{
			}
		};

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

		template<typename Scalar, typename Rng>
		struct scalar_norm_dist_op : public scalar_uniform_real_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "normalDist needs floating point types.");

			using scalar_uniform_real_op<Scalar, Rng>::scalar_uniform_real_op;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				using ur_base = scalar_uniform_real_op<Scalar, Rng>;

				thread_local Scalar cache;
				thread_local bool valid = false;
				bit_scalar<Scalar> bs;
				if (valid)
				{
					valid = false;
					return cache;
				}

				Scalar v1, v2, sx;
				while (1)
				{
					v1 = 2 * ur_base::operator()() - 1;
					v2 = 2 * ur_base::operator()() - 1;
					sx = v1 * v1 + v2 * v2;
					if (sx && sx < 1) break;
				}
				Scalar fx = std::sqrt((Scalar)-2.0 * std::log(sx) / sx);
				cache = fx * v2;
				valid = true;
				return fx * v1;
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using ur_base = scalar_uniform_real_op<Scalar, Rng>;

				thread_local Packet cache;
				thread_local bool valid = false;
				if (valid)
				{
					valid = false;
					return cache;
				}
				valid = true;
				Packet u1 = ur_base::template packetOp<Packet>(),
					u2 = ur_base::template packetOp<Packet>();
				const auto twopi = pset1<Packet>(2 * 3.14159265358979323846);
				const auto one = pset1<Packet>(1);
				const auto minustwo = pset1<Packet>(-2);

				u1 = psub(one, u1);

				auto radius = psqrt(pmul(minustwo, plog(u1)));
				auto theta = pmul(twopi, u2);
				Packet sintheta, costheta;

				psincos(theta, sintheta, costheta);
				cache = pmul(radius, costheta);
				return pmul(radius, sintheta);
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_norm_dist2_op : public scalar_norm_dist_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "normalDist needs floating point types.");

			Scalar mean = 0, stdev = 1;

			scalar_norm_dist2_op(const Rng& _rng,
				Scalar _mean = 0, Scalar _stdev = 1)
				: scalar_norm_dist_op<Scalar, Rng>{ _rng },
				mean{ _mean }, stdev{ _stdev }
			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return scalar_norm_dist_op<Scalar, Rng>::operator()() * stdev + mean;
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				return padd(pmul(
					scalar_norm_dist_op<Scalar, Rng>::template packetOp<Packet>(),
					pset1<Packet>(stdev)
				), pset1<Packet>(mean));
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_lognorm_dist_op : public scalar_norm_dist2_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "lognormalDist needs floating point types.");

			using scalar_norm_dist2_op<Scalar, Rng>::scalar_norm_dist2_op;

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return std::exp(scalar_norm_dist2_op<Scalar, Rng>::operator()());
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				return pexp(scalar_norm_dist2_op<Scalar, Rng>::template packetOp<Packet>());
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_exp_dist_op : public scalar_uniform_real_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "expDist needs floating point types.");

			Scalar lambda = 1;

			scalar_exp_dist_op(const Rng& _rng, Scalar _lambda = 1)
				: scalar_uniform_real_op<Scalar, Rng>{ _rng }, lambda{ _lambda }
			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return -std::log(1 - scalar_uniform_real_op<Scalar, Rng>::operator()()) / lambda;
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RandUtils<Packet, Rng>;

				return pnegate(pdiv(plog(
					psub(pset1<Packet>(1), scalar_uniform_real_op<Scalar, Rng>::template packetOp<Packet>())
				), pset1<Packet>(lambda)));
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_gamma_dist_op : public scalar_exp_dist_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "gammaDist needs floating point types.");

			Scalar alpha, beta, px, sqrt;

			scalar_gamma_dist_op(const Rng& _rng, Scalar _alpha = 1, Scalar _beta = 1)
				: scalar_exp_dist_op<Scalar, Rng>{ _rng }, alpha{ _alpha }, beta{ _beta }
			{
				px = 2.718281828459 / (alpha + 2.718281828459);
				sqrt = std::sqrt(2 * alpha - 1);
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				using ur_base = scalar_uniform_real_op<Scalar, Rng>;
				if (alpha < 1)
				{
					Scalar ux, vx, xx, qx;
					while (1)
					{
						ux = ur_base::operator()();
						vx = this->nzur_scalar();

						if (ux < px)
						{
							xx = std::pow(vx, 1 / alpha);
							qx = std::exp(-xx);
						}
						else
						{
							xx = 1 - std::log(vx);
							qx = std::pow(xx, alpha - 1);
						}

						if (ur_base::operator()() < qx)
						{
							return beta * xx;
						}
					}
				}
				if (alpha == 1)
				{
					return beta * scalar_exp_dist_op<Scalar, Rng>::operator()();
				}
				int count;
				if ((count = alpha) == alpha && count < 20)
				{
					Scalar yx;
					yx = this->nzur_scalar();
					while (--count)
					{
						yx *= this->nzur_scalar();
					}
					return -beta * std::log(yx);
				}

				while (1)
				{
					Scalar yx, xx;
					yx = std::tan(3.141592653589793 * ur_base::operator()());
					xx = sqrt * yx + alpha - 1;
					if (xx <= 0) continue;
					if (ur_base::operator()() <= (1 + yx * yx)
						* std::exp((alpha - 1) * std::log(xx / (alpha - 1)) - sqrt * yx))
					{
						return beta * xx;
					}
				}
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RandUtils<Packet, Rng>;
				auto& cm = Rand::detail::CompressMask<Packet>::get_inst();

				RUtils ru;
				thread_local Packet cache_rest;
				thread_local int cache_rest_cnt;
				thread_local const scalar_gamma_dist_op* cache_ptr = nullptr;
				if (cache_ptr != this)
				{
					cache_ptr = this;
					cache_rest = pset1<Packet>(0);
					cache_rest_cnt = 0;
				}

				if (alpha < 1)
				{
					while (1)
					{
						Packet ux = ru.uniform_real(this->rng);
						Packet vx = ru.nonzero_uniform_real(this->rng);

						Packet xx = pexp(pmul(pset1<Packet>(1 / alpha), plog(vx)));
						Packet qx = pexp(pnegate(xx));

						Packet xx2 = psub(pset1<Packet>(1), plog(vx));
						Packet qx2 = pexp(pmul(plog(xx2), pset1<Packet>(alpha - 1)));

						auto c = pcmplt(ux, pset1<Packet>(px));
						xx = pblendv(c, xx, xx2);
						qx = pblendv(c, qx, qx2);

						ux = ru.uniform_real(this->rng);
						Packet cands = pmul(pset1<Packet>(beta), xx);
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pcmplt(ux, qx),
							cache_rest, cache_rest_cnt, full);
						if (full) return cands;
					}
				}
				if (alpha == 1)
				{
					return pmul(pset1<Packet>(beta),
						scalar_exp_dist_op<Scalar, Rng>::template packetOp<Packet>()
					);
				}
				int count;
				if ((count = alpha) == alpha && count < 20)
				{
					RUtils ru;
					Packet ux, yx;
					yx = ru.nonzero_uniform_real(this->rng);
					while (--count)
					{
						yx = pmul(yx, ru.nonzero_uniform_real(this->rng));
					}
					return pnegate(pmul(pset1<Packet>(beta), plog(yx)));
				}
				else
				{
					while (1)
					{
						Packet alpha_1 = pset1<Packet>(alpha - 1);
						Packet ys, yc;
						psincos(pmul(pset1<Packet>(3.141592653589793), ru.uniform_real(this->rng)), ys, yc);
						Packet yx = pdiv(ys, yc);
						Packet xx = padd(pmul(pset1<Packet>(sqrt), yx), alpha_1);
						auto c = pcmplt(pset1<Packet>(0), xx);
						Packet ux = ru.uniform_real(this->rng);
						Packet ub = pmul(padd(pmul(yx, yx), pset1<Packet>(1)),
							pexp(psub(
								pmul(alpha_1, plog(pdiv(xx, alpha_1))),
								pmul(yx, pset1<Packet>(sqrt))
							))
						);
						c = pand(c, pcmple(ux, ub));
						Packet cands = pmul(pset1<Packet>(beta), xx);
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, c,
							cache_rest, cache_rest_cnt, full);
						if (full) return cands;
					}
				}
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_weibull_dist_op : public scalar_uniform_real_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "weilbullDist needs floating point types.");

			Scalar a = 1, b = 1;

			scalar_weibull_dist_op(const Rng& _rng, Scalar _a = 1, Scalar _b = 1)
				: scalar_uniform_real_op<Scalar, Rng>{ _rng }, a{ _a }, b{ _b }
			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return std::pow(-std::log(1 - scalar_uniform_real_op<Scalar, Rng>::operator()()), 1 / a) * b;
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RandUtils<Packet, Rng>;

				return pmul(pexp(pmul(plog(pnegate(plog(
					psub(pset1<Packet>(1), scalar_uniform_real_op<Scalar, Rng>::template packetOp<Packet>())
				))), pset1<Packet>(1 / a))), pset1<Packet>(b));
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_extreme_value_dist_op : public scalar_uniform_real_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "extremeValueDist needs floating point types.");

			Scalar a = 0, b = 1;

			scalar_extreme_value_dist_op(const Rng& _rng, Scalar _a = 0, Scalar _b = 1)
				: scalar_uniform_real_op<Scalar, Rng>{ _rng }, a{ _a }, b{ _b }
			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return (a - b * std::log(-std::log(this->nzur_scalar())));
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RandUtils<Packet, Rng>;
				return psub(pset1<Packet>(a),
					pmul(plog(pnegate(plog(RUtils{}.nonzero_uniform_real(this->rng)))), pset1<Packet>(b))
				);
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_chi_squared_dist_op : public scalar_gamma_dist_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "chiSquaredDist needs floating point types.");

			scalar_chi_squared_dist_op(const Rng& _rng, Scalar n = 1)
				: scalar_gamma_dist_op<Scalar, Rng>{ _rng, n * Scalar(0.5), 2 }
			{
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_cauchy_dist_op : public scalar_uniform_real_op<Scalar, Rng>
		{
			static_assert(std::is_floating_point<Scalar>::value, "cauchyDist needs floating point types.");

			Scalar a = 0, b = 1;

			scalar_cauchy_dist_op(const Rng& _rng, Scalar _a = 0, Scalar _b = 1)
				: scalar_uniform_real_op<Scalar, Rng>{ _rng }, a{ _a }, b{ _b }
			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return a + b * std::tan(3.141592653589793 * (scalar_uniform_real_op<Scalar, Rng>::operator()() - 0.5));
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using RUtils = RandUtils<Packet, Rng>;
				Packet s, c;
				psincos(pmul(pset1<Packet>(3.141592653589793),
					psub(scalar_uniform_real_op<Scalar, Rng>::template packetOp<Packet>(), pset1<Packet>(0.5))
				), s, c);
				return padd(pset1<Packet>(a),
					pmul(pset1<Packet>(b), pdiv(s, c))
				);
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_randbits_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_balanced_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_uniform_real_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_norm_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_lognorm_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_exp_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_gamma_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_weibull_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_extreme_value_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_chi_squared_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_cauchy_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};
	}

	namespace Rand
	{
		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>
			randBits(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_randbits_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>
			randBitsLike(Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_randbits_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_randbits_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>
			balanced(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_balanced_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>
			balancedLike(const Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_balanced_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_balanced_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>
			uniformReal(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>
			uniformRealLike(Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_uniform_real_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			normalDist(Index rows, Index cols, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			normalDistLike(Derived& o, Urng&& urng)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_norm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng))
				);
		}


		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>
			normalDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>
			normalDistLike(Derived& o, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_norm_dist2_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			lognormalDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>
			lognormalDistLike(Derived& o, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return CwiseNullaryOp<internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_lognorm_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), mean, stdev)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>
			expDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), lambda)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>
			expDistLike(Derived& o, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return CwiseNullaryOp<internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_exp_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), lambda)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>
			gammaDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), alpha, beta)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>
			gammaDistLike(Derived& o, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return CwiseNullaryOp<internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_gamma_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), alpha, beta)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>
			weibullDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>
			weibullDistLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_weibull_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>
			extremeValueDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>
			extremeValueDistLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return CwiseNullaryOp<internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_extreme_value_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), a, b)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>
			chiSquaredDist(Index rows, Index cols, Urng&& urng, typename Derived::Scalar n = 0)
		{
			return CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				rows, cols, internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
				);
		}

		template<typename Derived, typename Urng>
		inline const CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>
			chiSquaredDistLike(Derived& o, Urng&& urng, typename Derived::Scalar n = 0)
		{
			return CwiseNullaryOp<internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>, const Derived>(
				o.rows(), o.cols(), internal::scalar_chi_squared_dist_op<typename Derived::Scalar, Urng>(std::forward<Urng>(urng), n)
				);
		}
	}
}

#endif