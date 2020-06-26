/**
 * @file NormalExp.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.1.0
 * @date 2020-06-22
 * 
 * @copyright Copyright (c) 2020
 * 
 */


#ifndef EIGENRAND_DISTS_NORMAL_EXP_H
#define EIGENRAND_DISTS_NORMAL_EXP_H

namespace Eigen
{
	namespace internal
	{
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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_norm_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_norm_dist2_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_lognorm_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_exp_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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
				auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();

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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_gamma_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_weibull_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_extreme_value_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_chi_squared_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
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
		struct functor_traits<scalar_cauchy_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};
	}
}

#endif