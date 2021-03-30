/**
 * @file NormalExp.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.3.3
 * @date 2021-03-31
 * 
 * @copyright Copyright (c) 2020-2021
 * 
 */


#ifndef EIGENRAND_DISTS_NORMAL_EXP_H
#define EIGENRAND_DISTS_NORMAL_EXP_H

namespace Eigen
{
	namespace Rand
	{
		/**
		 * @brief Generator of reals on the standard normal distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class StdNormalGen : OptCacheStore, public GenBase<StdNormalGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "normalDist needs floating point types.");
			bool valid = false;
			UniformRealGen<_Scalar> ur;
			
		public:
			using Scalar = _Scalar;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				if (valid)
				{
					valid = false;
					return OptCacheStore::get<_Scalar>();
				}
				valid = true;

				_Scalar v1, v2, sx;
				while (1)
				{
					v1 = 2 * ur(rng) - 1;
					v2 = 2 * ur(rng) - 1;
					sx = v1 * v1 + v2 * v2;
					if (sx && sx < 1) break;
				}
				_Scalar fx = std::sqrt((_Scalar)-2.0 * std::log(sx) / sx);
				OptCacheStore::get<_Scalar>() = fx * v2;
				return fx * v1;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				if (valid)
				{
					valid = false;
					return OptCacheStore::template get<Packet>();
				}
				valid = true;
				Packet u1 = ur.template packetOp<Packet>(rng),
					u2 = ur.template packetOp<Packet>(rng);

				u1 = psub(pset1<Packet>(1), u1);

				auto radius = psqrt(pmul(pset1<Packet>(-2), plog(u1)));
				auto theta = pmul(pset1<Packet>(2 * constant::pi), u2);
				Packet sintheta, costheta;

				psincos(theta, sintheta, costheta);
				OptCacheStore::template get<Packet>() = pmul(radius, costheta);
				return pmul(radius, sintheta);
			}
		};

		/**
		 * @brief Generator of reals on a normal distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class NormalGen : public GenBase<NormalGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "normalDist needs floating point types.");
			StdNormalGen<_Scalar> stdnorm;
			_Scalar mean = 0, stdev = 1;
		
		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new normal generator
			 * 
			 * @param _mean mean of the distribution
			 * @param _stdev standard deviation of the distribution
			 */
			NormalGen(_Scalar _mean = 0, _Scalar _stdev = 1)
				: mean{ _mean }, stdev{ _stdev }
			{
			}

			NormalGen(const NormalGen&) = default;
			NormalGen(NormalGen&&) = default;

			NormalGen& operator=(const NormalGen&) = default;
			NormalGen& operator=(NormalGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return stdnorm(std::forward<Rng>(rng)) * stdev + mean;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				return padd(pmul(
					stdnorm.template packetOp<Packet>(std::forward<Rng>(rng)),
					pset1<Packet>(stdev)
				), pset1<Packet>(mean));
			}
		};

		/**
		 * @brief Generator of reals on a lognormal distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class LognormalGen : public GenBase<LognormalGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "lognormalDist needs floating point types.");
			NormalGen<_Scalar> norm;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new lognormal generator
			 * 
			 * @param _mean mean of the log distribution
			 * @param _stdev standard deviation of the log distribution
			 */
			LognormalGen(_Scalar _mean = 0, _Scalar _stdev = 1)
				: norm{ _mean, _stdev }
			{
			}

			LognormalGen(const LognormalGen&) = default;
			LognormalGen(LognormalGen&&) = default;

			LognormalGen& operator=(const LognormalGen&) = default;
			LognormalGen& operator=(LognormalGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return std::exp(norm(std::forward<Rng>(rng)));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				return pexp(norm.template packetOp<Packet>(std::forward<Rng>(rng)));
			}
		};

		/**
		 * @brief Generator of reals on a Student's t distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class StudentTGen : public GenBase<StudentTGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "studentT needs floating point types.");
			UniformRealGen<_Scalar> ur;
			_Scalar n;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Student's t generator
			 * 
			 * @param _n degrees of freedom
			 */
			StudentTGen(_Scalar _n = 1)
				: n{ _n }
			{
			}

			StudentTGen(const StudentTGen&) = default;
			StudentTGen(StudentTGen&&) = default;

			StudentTGen& operator=(const StudentTGen&) = default;
			StudentTGen& operator=(StudentTGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				_Scalar v1, v2, sx;
				while (1)
				{
					v1 = 2 * ur(rng) - 1;
					v2 = 2 * ur(rng) - 1;
					sx = v1 * v1 + v2 * v2;
					if (sx && sx < 1) break;
				}

				_Scalar fx = std::sqrt(n * (std::pow(sx, -2 / n) - 1) / sx);
				return fx * v1;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				Packet u1 = ur.template packetOp<Packet>(rng),
					u2 = ur.template packetOp<Packet>(rng);

				u1 = psub(pset1<Packet>(1), u1);
				auto pn = pset1<Packet>(n);
				auto radius = psqrt(pmul(pn,
					psub(pexp(pmul(plog(u1), pset1<Packet>(-2 / n))), pset1<Packet>(1))
				));
				auto theta = pmul(pset1<Packet>(2 * constant::pi), u2);
				//Packet sintheta, costheta;
				//psincos(theta, sintheta, costheta);
				return pmul(radius, psin(theta));
			}
		};

		template<typename> class GammaGen;

		/**
		 * @brief Generator of reals on an exponential distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class ExponentialGen : public GenBase<ExponentialGen<_Scalar>, _Scalar>
		{
			friend GammaGen<_Scalar>;
			static_assert(std::is_floating_point<_Scalar>::value, "expDist needs floating point types.");
			UniformRealGen<_Scalar> ur;
			_Scalar lambda = 1;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new exponential generator
			 * 
			 * @param _lambda scale parameter of the distribution
			 */
			ExponentialGen(_Scalar _lambda = 1)
				: lambda{ _lambda }
			{
			}
			
			ExponentialGen(const ExponentialGen&) = default;
			ExponentialGen(ExponentialGen&&) = default;

			ExponentialGen& operator=(const ExponentialGen&) = default;
			ExponentialGen& operator=(ExponentialGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return -std::log(1 - ur(std::forward<Rng>(rng))) / lambda;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				return pnegate(pdiv(plog(
					psub(pset1<Packet>(1), ur.template packetOp<Packet>(std::forward<Rng>(rng)))
				), pset1<Packet>(lambda)));
			}
		};

		template<typename> class NegativeBinomialGen;

		/**
		 * @brief Generator of reals on a gamma distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class GammaGen : OptCacheStore, public GenBase<GammaGen<_Scalar>, _Scalar>
		{
			template<typename _Ty>
			friend class NegativeBinomialGen;
			static_assert(std::is_floating_point<_Scalar>::value, "gammaDist needs floating point types.");
			int cache_rest_cnt = 0;
			ExponentialGen<_Scalar> expon;
			_Scalar alpha, beta, px, sqrt;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new gamma generator
			 * 
			 * @param _alpha shape parameter of the distribution
			 * @param _beta scale parameter of the distribution
			 */
			GammaGen(_Scalar _alpha = 1, _Scalar _beta = 1)
				: alpha{ _alpha }, beta{ _beta }
			{
				px = constant::e / (alpha + constant::e);
				sqrt = std::sqrt(2 * alpha - 1);
			}

			GammaGen(const GammaGen&) = default;
			GammaGen(GammaGen&&) = default;

			GammaGen& operator=(const GammaGen&) = default;
			GammaGen& operator=(GammaGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				if (alpha < 1)
				{
					_Scalar ux, vx, xx, qx;
					while (1)
					{
						ux = expon.ur(rng);
						vx = expon.ur.nzur_scalar(rng);

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

						if (expon.ur(rng) < qx)
						{
							return beta * xx;
						}
					}
				}
				if (alpha == 1)
				{
					return beta * expon(rng);
				}
				int count;
				if ((count = alpha) == alpha && count < 20)
				{
					_Scalar yx;
					yx = expon.ur.nzur_scalar(rng);
					while (--count)
					{
						yx *= expon.ur.nzur_scalar(rng);
					}
					return -beta * std::log(yx);
				}

				while (1)
				{
					_Scalar yx, xx;
					yx = std::tan(constant::pi * expon.ur(rng));
					xx = sqrt * yx + alpha - 1;
					if (xx <= 0) continue;
					if (expon.ur(rng) <= (1 + yx * yx)
						* std::exp((alpha - 1) * std::log(xx / (alpha - 1)) - sqrt * yx))
					{
						return beta * xx;
					}
				}
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RUtils = RandUtils<Packet, Rng>;
				auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();

				RUtils ru;
				if (alpha < 1)
				{
					while (1)
					{
						Packet ux = ru.uniform_real(rng);
						Packet vx = ru.nonzero_uniform_real(rng);

						Packet xx = pexp(pmul(pset1<Packet>(1 / alpha), plog(vx)));
						Packet qx = pexp(pnegate(xx));

						Packet xx2 = psub(pset1<Packet>(1), plog(vx));
						Packet qx2 = pexp(pmul(plog(xx2), pset1<Packet>(alpha - 1)));

						auto c = pcmplt(ux, pset1<Packet>(px));
						xx = pblendv(c, xx, xx2);
						qx = pblendv(c, qx, qx2);

						ux = ru.uniform_real(rng);
						Packet cands = pmul(pset1<Packet>(beta), xx);
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pcmplt(ux, qx),
							OptCacheStore::template get<Packet>(), cache_rest_cnt, full);
						if (full) return cands;
					}
				}
				if (alpha == 1)
				{
					return pmul(pset1<Packet>(beta),
						expon.template packetOp<Packet>(rng)
					);
				}
				int count;
				if ((count = alpha) == alpha && count < 20)
				{
					RUtils ru;
					Packet ux, yx;
					yx = ru.nonzero_uniform_real(rng);
					while (--count)
					{
						yx = pmul(yx, ru.nonzero_uniform_real(rng));
					}
					return pnegate(pmul(pset1<Packet>(beta), plog(yx)));
				}
				else
				{
					while (1)
					{
						Packet alpha_1 = pset1<Packet>(alpha - 1);
						Packet ys, yc;
						psincos(pmul(pset1<Packet>(constant::pi), ru.uniform_real(rng)), ys, yc);
						Packet yx = pdiv(ys, yc);
						Packet xx = padd(pmul(pset1<Packet>(sqrt), yx), alpha_1);
						auto c = pcmplt(pset1<Packet>(0), xx);
						Packet ux = ru.uniform_real(rng);
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
							OptCacheStore::template get<Packet>(), cache_rest_cnt, full);
						if (full) return cands;
					}
				}
			}
		};

		/**
		 * @brief Generator of reals on a Weibull distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class WeibullGen : public GenBase<WeibullGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "weilbullDist needs floating point types.");
			UniformRealGen<_Scalar> ur;
			_Scalar a = 1, b = 1;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Weibull generator
			 * 
			 * @param _a shape parameter of the distribution
			 * @param _b scale parameter of the distribution
			 */
			WeibullGen(_Scalar _a = 1, _Scalar _b = 1)
				: a{ _a }, b{ _b }
			{
			}
			
			WeibullGen(const WeibullGen&) = default;
			WeibullGen(WeibullGen&&) = default;

			WeibullGen& operator=(const WeibullGen&) = default;
			WeibullGen& operator=(WeibullGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return std::pow(-std::log(1 - ur(std::forward<Rng>(rng))), 1 / a) * b;
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				return pmul(pexp(pmul(plog(pnegate(plog(
					psub(pset1<Packet>(1), ur.template packetOp<Packet>(std::forward<Rng>(rng)))
				))), pset1<Packet>(1 / a))), pset1<Packet>(b));
			}
		};

		/**
		 * @brief Generator of reals on an extreme value distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class ExtremeValueGen : public GenBase<ExtremeValueGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "extremeValueDist needs floating point types.");
			UniformRealGen<_Scalar> ur;
			_Scalar a = 0, b = 1;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new extreme value generator
			 * 
			 * @param _a location parameter of the distribution
			 * @param _b scale parameter of the distribution
			 */
			ExtremeValueGen(_Scalar _a = 0, _Scalar _b = 1)
				: a{ _a }, b{ _b }
			{
			}

			ExtremeValueGen(const ExtremeValueGen&) = default;
			ExtremeValueGen(ExtremeValueGen&&) = default;

			ExtremeValueGen& operator=(const ExtremeValueGen&) = default;
			ExtremeValueGen& operator=(ExtremeValueGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return (a - b * std::log(-std::log(ur.nzur_scalar(std::forward<Rng>(rng)))));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using RUtils = RandUtils<Packet, Rng>;
				return psub(pset1<Packet>(a),
					pmul(plog(pnegate(plog(RUtils{}.nonzero_uniform_real(std::forward<Rng>(rng))))), pset1<Packet>(b))
				);
			}
		};

		/**
		 * @brief Generator of reals on a chi-squared distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class ChiSquaredGen : public GenBase<ChiSquaredGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "chiSquaredDist needs floating point types.");
			GammaGen<_Scalar> gamma;
		public:
			using Scalar = _Scalar;
			
			/**
			 * @brief Construct a new chi-squared generator
			 * 
			 * @param n degrees of freedom
			 */
			ChiSquaredGen(_Scalar n = 1)
				: gamma{ n * _Scalar(0.5), 2 }
			{
			}

			ChiSquaredGen(const ChiSquaredGen&) = default;
			ChiSquaredGen(ChiSquaredGen&&) = default;

			ChiSquaredGen& operator=(const ChiSquaredGen&) = default;
			ChiSquaredGen& operator=(ChiSquaredGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				return gamma(rng);
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				return gamma.template packetOp<Packet>(rng);
			}
		};
		
		/**
		 * @brief Generator of reals on a Cauchy distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class CauchyGen : public GenBase<CauchyGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "cauchyDist needs floating point types.");
			UniformRealGen<_Scalar> ur; 
			_Scalar a = 0, b = 1;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Cauchy generator
			 * 
			 * @param _a location parameter of the distribution
			 * @param _b scale parameter of the distribution
			 */
			CauchyGen(_Scalar _a = 0, _Scalar _b = 1)
				: a{ _a }, b{ _b }
			{
			}

			CauchyGen(const CauchyGen&) = default;
			CauchyGen(CauchyGen&&) = default;

			CauchyGen& operator=(const CauchyGen&) = default;
			CauchyGen& operator=(CauchyGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return a + b * std::tan(constant::pi * (ur(std::forward<Rng>(rng)) - 0.5));
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				Packet s, c;
				psincos(pmul(pset1<Packet>(constant::pi),
					psub(ur.template packetOp<Packet>(std::forward<Rng>(rng)), pset1<Packet>(0.5))
				), s, c);
				return padd(pset1<Packet>(a),
					pmul(pset1<Packet>(b), pdiv(s, c))
				);
			}
		};

		template<typename> class FisherFGen;

		/**
		 * @brief Generator of reals on a beta distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class BetaGen : OptCacheStore, public GenBase<BetaGen<_Scalar>, _Scalar>
		{
			friend FisherFGen<_Scalar>;
			static_assert(std::is_floating_point<_Scalar>::value, "betaDist needs floating point types.");
			int cache_rest_cnt = 0;
			UniformRealGen<_Scalar> ur;
			_Scalar a, b;
			GammaGen<_Scalar> gd1, gd2;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new beta generator
			 * 
			 * @param _a, _b shape parameter
			 */
			BetaGen(_Scalar _a = 1, _Scalar _b = 1)
				: a{ _a }, b{ _b },
				gd1{ _a }, gd2{ _b }
			{
			}

			BetaGen(const BetaGen&) = default;
			BetaGen(BetaGen&&) = default;

			BetaGen& operator=(const BetaGen&) = default;
			BetaGen& operator=(BetaGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				if (a < 1 && b < 1)
				{
					_Scalar x, p1, p2;
					while (1)
					{
						p1 = std::pow(ur(rng), 1 / a);
						p2 = std::pow(ur(rng), 1 / b);
						x = p1 + p2;
						if (x <= 1) break;
					}
					return p1 / x;
				}
				else
				{
					_Scalar p1 = gd1(rng), p2 = gd2(rng);
					return p1 / (p1 + p2);
				}
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				if (a < 1 && b < 1)
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					Packet x, p1, p2;
					while (1)
					{
						p1 = pexp(pmul(plog(ur.template packetOp<Packet>(rng)), pset1<Packet>(1 / a)));
						p2 = pexp(pmul(plog(ur.template packetOp<Packet>(rng)), pset1<Packet>(1 / b)));
						x = padd(p1, p2);
						Packet cands = pdiv(p1, x);
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pcmple(x, pset1<Packet>(1)),
							OptCacheStore::template get<Packet>(), cache_rest_cnt, full);
						if (full) return cands;
					}
				}
				else
				{
					auto p1 = gd1.template packetOp<Packet>(rng),
						p2 = gd2.template packetOp<Packet>(rng);
					return pdiv(p1, padd(p1, p2));
				}
			}
		};

		/**
		 * @brief Generator of reals on a Fisher's f distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class FisherFGen : public GenBase<FisherFGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_floating_point<_Scalar>::value, "fisherF needs floating point types.");
			BetaGen<_Scalar> beta;
		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Fisher's f generator
			 * 
			 * @param m, n degrees of freedom
			 */
			FisherFGen(_Scalar m = 1, _Scalar n = 1)
				: beta{ m * _Scalar(0.5), n * _Scalar(0.5) }
			{
			}
			
			FisherFGen(const FisherFGen&) = default;
			FisherFGen(FisherFGen&&) = default;

			FisherFGen& operator=(const FisherFGen&) = default;
			FisherFGen& operator=(FisherFGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				auto x = beta(std::forward<Rng>(rng));
				return beta.b / beta.a * x / (1 - x);
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				auto x = beta.template packetOp<Packet>(std::forward<Rng>(rng));
				return pdiv(pmul(pset1<Packet>(beta.b / beta.a), x), psub(pset1<Packet>(1), x));
			}
		};


		template<typename Derived, typename Urng>
		using BetaType = CwiseNullaryOp<internal::scalar_rng_adaptor<BetaGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the beta distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a,b shape parameter
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::BetaGen
		 */
		template<typename Derived, typename Urng>
		inline const BetaType<Derived, Urng>
			beta(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), BetaGen<typename Derived::Scalar>{a, b} }
			};
		}

		/**
		 * @brief generates reals on the beta distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a,b shape parameter
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::BetaGen
		 */
		template<typename Derived, typename Urng>
		inline const BetaType<Derived, Urng>
			betaLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), BetaGen<typename Derived::Scalar>{a, b} }
			};
		}

		template<typename Derived, typename Urng>
		using CauchyType = CwiseNullaryOp<internal::scalar_rng_adaptor<CauchyGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the Cauchy distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::CauchyGen
		 */
		template<typename Derived, typename Urng>
		inline const CauchyType<Derived, Urng>
			cauchy(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), CauchyGen<typename Derived::Scalar>{a, b} }
			};
		}

		/**
		 * @brief generates reals on the Cauchy distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::CauchyGen
		 */
		template<typename Derived, typename Urng>
		inline const CauchyType<Derived, Urng>
			cauchyLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), CauchyGen<typename Derived::Scalar>{a, b} }
			};
		}

		template<typename Derived, typename Urng>
		using NormalType = CwiseNullaryOp<internal::scalar_rng_adaptor<StdNormalGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on a standard normal distribution (`mean` = 0, `stdev`=1)
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::StdNormalGen
		 */
		template<typename Derived, typename Urng>
		inline const NormalType<Derived, Urng>
			normal(Index rows, Index cols, Urng&& urng)
		{
			return {
				rows, cols, { std::forward<Urng>(urng) }
			};
		}

		/**
		 * @brief generates reals on a standard normal distribution (`mean` = 0, `stdev`=1)
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::StdNormalGen
		 */
		template<typename Derived, typename Urng>
		inline const NormalType<Derived, Urng>
			normalLike(Derived& o, Urng&& urng)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng) }
			};
		}

		template<typename Derived, typename Urng>
		using Normal2Type = CwiseNullaryOp<internal::scalar_rng_adaptor<NormalGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on a normal distribution with arbitrary `mean` and `stdev`.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::NormalGen
		 */
		template<typename Derived, typename Urng>
		inline const Normal2Type<Derived, Urng>
			normal(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), NormalGen<typename Derived::Scalar>{mean, stdev} }
			};
		}

		/**
		 * @brief generates reals on a normal distribution with arbitrary `mean` and `stdev`.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::NormalGen
		 */
		template<typename Derived, typename Urng>
		inline const Normal2Type<Derived, Urng>
			normalLike(Derived& o, Urng&& urng, typename Derived::Scalar mean, typename Derived::Scalar stdev = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), NormalGen<typename Derived::Scalar>{mean, stdev} }
			};
		}

		template<typename Derived, typename Urng>
		using LognormalType = CwiseNullaryOp<internal::scalar_rng_adaptor<LognormalGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on a lognormal distribution with arbitrary `mean` and `stdev`.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::LognormalGen
		 */
		template<typename Derived, typename Urng>
		inline const LognormalType<Derived, Urng>
			lognormal(Index rows, Index cols, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), LognormalGen<typename Derived::Scalar>{mean, stdev} }
			};
		}

		/**
		 * @brief generates reals on a lognormal distribution with arbitrary `mean` and `stdev`.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param mean a mean value of the distribution
		 * @param stdev a standard deviation value of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::LognormalGen
		 */
		template<typename Derived, typename Urng>
		inline const LognormalType<Derived, Urng>
			lognormalLike(Derived& o, Urng&& urng, typename Derived::Scalar mean = 0, typename Derived::Scalar stdev = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), LognormalGen<typename Derived::Scalar>{mean, stdev} }
			};
		}

		template<typename Derived, typename Urng>
		using StudentTType = CwiseNullaryOp<internal::scalar_rng_adaptor<StudentTGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the Student's t distribution with arbirtrary degress of freedom.
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param n degrees of freedom
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::StudentTGen
		 */
		template<typename Derived, typename Urng>
		inline const StudentTType<Derived, Urng>
			studentT(Index rows, Index cols, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), StudentTGen<typename Derived::Scalar>{n} }
			};
		}

		/**
		 * @brief generates reals on the Student's t distribution with arbirtrary degress of freedom.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param n degrees of freedom
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::StudentTGen
		 */
		template<typename Derived, typename Urng>
		inline const StudentTType<Derived, Urng>
			studentTLike(Derived& o, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), StudentTGen<typename Derived::Scalar>{n} }
			};
		}

		template<typename Derived, typename Urng>
		using ExponentialType = CwiseNullaryOp<internal::scalar_rng_adaptor<ExponentialGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on an exponential distribution with arbitrary scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param lambda a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::ExponentialGen
		 */
		template<typename Derived, typename Urng>
		inline const ExponentialType<Derived, Urng>
			exponential(Index rows, Index cols, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), ExponentialGen<typename Derived::Scalar>{lambda} }
			};
		}

		/**
		 * @brief generates reals on an exponential distribution with arbitrary scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param lambda a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::ExponentialGen
		 */
		template<typename Derived, typename Urng>
		inline const ExponentialType<Derived, Urng>
			exponentialLike(Derived& o, Urng&& urng, typename Derived::Scalar lambda = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), ExponentialGen<typename Derived::Scalar>{lambda} }
			};
		}

		template<typename Derived, typename Urng>
		using GammaType = CwiseNullaryOp<internal::scalar_rng_adaptor<GammaGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on a gamma distribution with arbitrary shape and scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param alpha a shape parameter of the distribution
		 * @param beta a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::GammaGen
		 */
		template<typename Derived, typename Urng>
		inline const GammaType<Derived, Urng>
			gamma(Index rows, Index cols, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), GammaGen<typename Derived::Scalar>{alpha, beta} }
			};
		}

		/**
		 * @brief generates reals on a gamma distribution with arbitrary shape and scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param alpha a shape parameter of the distribution
		 * @param beta a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::GammaGen
		 */
		template<typename Derived, typename Urng>
		inline const GammaType<Derived, Urng>
			gammaLike(Derived& o, Urng&& urng, typename Derived::Scalar alpha = 1, typename Derived::Scalar beta = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), GammaGen<typename Derived::Scalar>{alpha, beta} }
			};
		}

		template<typename Derived, typename Urng>
		using WeibullType = CwiseNullaryOp<internal::scalar_rng_adaptor<WeibullGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on a Weibull distribution with arbitrary shape and scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a a shape parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::WeibullGen
		 */
		template<typename Derived, typename Urng>
		inline const WeibullType<Derived, Urng>
			weibull(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), WeibullGen<typename Derived::Scalar>{a, b} }
			};
		}

		/**
		 * @brief generates reals on a Weibull distribution with arbitrary shape and scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a a shape parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::WeibullGen
		 */
		template<typename Derived, typename Urng>
		inline const WeibullType<Derived, Urng>
			weibullLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 1, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), WeibullGen<typename Derived::Scalar>{a, b} }
			};
		}

		template<typename Derived, typename Urng>
		using ExtremeValueType = CwiseNullaryOp<internal::scalar_rng_adaptor<ExtremeValueGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on an extreme value distribution
		 * (a.k.a Gumbel Type I, log-Weibull, Fisher-Tippett Type I) with arbitrary shape and scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::ExtremeValueGen
		 */
		template<typename Derived, typename Urng>
		inline const ExtremeValueType<Derived, Urng>
			extremeValue(Index rows, Index cols, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), ExtremeValueGen<typename Derived::Scalar>{a, b} }
			};
		}

		/**
		 * @brief generates reals on an extreme value distribution
		 * (a.k.a Gumbel Type I, log-Weibull, Fisher-Tippett Type I) with arbitrary shape and scale parameter.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param a a location parameter of the distribution
		 * @param b a scale parameter of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::ExtremeValueGen
		 */
		template<typename Derived, typename Urng>
		inline const ExtremeValueType<Derived, Urng>
			extremeValueLike(Derived& o, Urng&& urng, typename Derived::Scalar a = 0, typename Derived::Scalar b = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), ExtremeValueGen<typename Derived::Scalar>{a, b} }
			};
		}

		template<typename Derived, typename Urng>
		using ChiSquaredType = CwiseNullaryOp<internal::scalar_rng_adaptor<ChiSquaredGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the Chi-squared distribution with arbitrary degrees of freedom.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param n the degrees of freedom of the distribution
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::ChiSquaredGen
		 */
		template<typename Derived, typename Urng>
		inline const ChiSquaredType<Derived, Urng>
			chiSquared(Index rows, Index cols, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), ChiSquaredGen<typename Derived::Scalar>{n} }
			};
		}

		/**
		 * @brief generates reals on the Chi-squared distribution with arbitrary degrees of freedom.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param n the degrees of freedom of the distribution
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::ChiSquaredGen
		 */
		template<typename Derived, typename Urng>
		inline const ChiSquaredType<Derived, Urng>
			chiSquaredLike(Derived& o, Urng&& urng, typename Derived::Scalar n = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), ChiSquaredGen<typename Derived::Scalar>{n} }
			};
		}

		template<typename Derived, typename Urng>
		using FisherFType = CwiseNullaryOp<internal::scalar_rng_adaptor<FisherFGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the Fisher's F distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param m degrees of freedom
		 * @param n degrees of freedom
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::FisherFGen 
		 */
		template<typename Derived, typename Urng>
		inline const FisherFType<Derived, Urng>
			fisherF(Index rows, Index cols, Urng&& urng, typename Derived::Scalar m = 1, typename Derived::Scalar n = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), FisherFGen<typename Derived::Scalar>{m, n} }
			};
		}

		/**
		 * @brief generates reals on the Fisher's F distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param m degrees of freedom
		 * @param n degrees of freedom
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::FisherFGen
		 */
		template<typename Derived, typename Urng>
		inline const FisherFType<Derived, Urng>
			fisherFLike(Derived& o, Urng&& urng, typename Derived::Scalar m = 1, typename Derived::Scalar n = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), FisherFGen<typename Derived::Scalar>{m, n} }
			};
		}
	}
}

#endif