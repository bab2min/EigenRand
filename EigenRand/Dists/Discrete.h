/**
 * @file Discrete.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.2.0
 * @date 2020-06-22
 * 
 * @copyright Copyright (c) 2020
 * 
 */

#ifndef EIGENRAND_DISTS_DISCRETE_H
#define EIGENRAND_DISTS_DISCRETE_H

#include <memory>
#include <iterator>
#include <limits>

namespace Eigen
{
	namespace internal
	{
		template<typename Scalar, typename Rng>
		struct scalar_uniform_int_op : public scalar_randbits_op<Scalar, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "uniformInt needs integral types.");
			using ur_base = scalar_randbits_op<Scalar, Rng>;

			Scalar pmin;
			size_t pdiff, bitsize, bitmask;
			
			scalar_uniform_int_op(const Rng& _rng, Scalar _min, Scalar _max)
				: ur_base{ _rng }, pmin{ _min }, pdiff{ (size_t)(_max - _min) }
			{
				if ((pdiff + 1) > pdiff)
				{
					bitsize = (size_t)std::ceil(std::log2(pdiff + 1));
				}
				else
				{
					bitsize = (size_t)std::ceil(std::log2(pdiff));
				}
				bitmask = (Scalar)(((size_t)-1) >> (sizeof(size_t) * 8 - bitsize));
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				auto rx = ur_base::operator()();
				if (pdiff == bitmask)
				{
					return (Scalar)(rx & bitmask) + pmin;
				}
				else
				{
					size_t bitcnt = bitsize;
					while (1)
					{
						Scalar cands = (Scalar)(rx & bitmask);
						if (cands <= pdiff) return cands;
						if (bitcnt + bitsize < 32)
						{
							rx >>= bitsize;
							bitcnt += bitsize;
						}
						else
						{
							rx = ur_base::operator()();
							bitcnt = bitsize;
						}
					}
				}
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				auto rx = ur_base::template packetOp<Packet>();
				auto pbitmask = pset1<Packet>(bitmask);
				if (pdiff == bitmask)
				{
					return padd(pand(rx, pbitmask), pset1<Packet>(pmin));
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					thread_local Packet cache_rest;
					thread_local int cache_rest_cnt;
					thread_local const scalar_uniform_int_op* cache_ptr = nullptr;
					if (cache_ptr != this)
					{
						cache_ptr = this;
						cache_rest = pset1<Packet>(0);
						cache_rest_cnt = 0;
					}

					auto plen = pset1<Packet>(pdiff + 1);
					size_t bitcnt = bitsize;
					while (1)
					{
						// accept cands that only < plen
						auto cands = pand(rx, pbitmask);
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pcmplt(cands, plen),
							cache_rest, cache_rest_cnt, full);
						if (full) return padd(cands, pset1<Packet>(pmin));

						if (bitcnt + bitsize < 32)
						{
							rx = psrl(rx, bitsize);
							bitcnt += bitsize;
						}
						else
						{
							rx = ur_base::template packetOp<Packet>();
							bitcnt = bitsize;
						}
					}
				}
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_uniform_int_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename _Precision = uint32_t, typename _Size = uint32_t>
		class AliasMethod
		{
			std::unique_ptr<_Precision[]> arr;
			std::unique_ptr<_Size[]> alias;
			size_t msize = 0, bitsize = 0, bitmask = 0;

		public:
			AliasMethod()
			{
			}

			AliasMethod(const AliasMethod& o)
			{
				operator=(o);
			}

			AliasMethod(AliasMethod&& o)
			{
				operator=(o);
			}

			AliasMethod& operator=(const AliasMethod& o)
			{
				msize = o.msize;
				bitsize = o.bitsize;
				bitmask = o.bitmask;
				if (msize)
				{
					arr = std::unique_ptr<_Precision[]>(new _Precision[1 << bitsize]);
					alias = std::unique_ptr<_Size[]>(new _Size[1 << bitsize]);

					std::copy(o.arr.get(), o.arr.get() + (1 << bitsize), arr.get());
					std::copy(o.alias.get(), o.alias.get() + (1 << bitsize), alias.get());
				}
				return *this;
			}

			AliasMethod& operator=(AliasMethod&& o)
			{
				msize = o.msize;
				bitsize = o.bitsize;
				bitmask = o.bitmask;
				std::swap(arr, o.arr);
				std::swap(alias, o.alias);
				return *this;
			}

			template<typename _Iter>
			AliasMethod(_Iter first, _Iter last)
			{
				buildTable(first, last);
			}

			template<typename _Iter>
			void buildTable(_Iter first, _Iter last)
			{
				size_t psize, nbsize;
				msize = 0;
				double sum = 0;
				for (auto it = first; it != last; ++it, ++msize)
				{
					sum += *it;
				}

				if (!std::isfinite(sum)) throw std::invalid_argument{ "cannot build NaN value distribution" };

				// ceil to power of 2
				nbsize = (size_t)std::ceil(std::log2(msize));
				psize = (size_t)1 << nbsize;

				if (nbsize != bitsize)
				{
					arr = std::unique_ptr<_Precision[]>(new _Precision[psize]);
					std::fill(arr.get(), arr.get() + psize, 0);
					alias = std::unique_ptr<_Size[]>(new _Size[psize]);
					bitsize = nbsize;
					bitmask = ((size_t)(-1)) >> (sizeof(size_t) * 8 - bitsize);
				}

				sum /= psize;

				auto f = std::unique_ptr<double[]>(new double[psize]);
				auto pf = f.get();
				for (auto it = first; it != last; ++it, ++pf)
				{
					*pf = *it / sum;
				}
				std::fill(pf, pf + psize - msize, 0);

				size_t over = 0, under = 0, mm;
				while (over < psize && f[over] < 1) ++over;
				while (under < psize && f[under] >= 1) ++under;
				mm = under + 1;

				while (over < psize && under < psize)
				{
					if (std::is_integral<_Precision>::value)
					{
						arr[under] = (_Precision)(f[under] * (std::numeric_limits<_Precision>::max() + 1.0));
					}
					else
					{
						arr[under] = (_Precision)f[under];
					}
					alias[under] = over;
					f[over] += f[under] - 1;
					if (f[over] >= 1 || mm <= over)
					{
						for (under = mm; under < psize && f[under] >= 1; ++under);
						mm = under + 1;
					}
					else
					{
						under = over;
					}

					while (over < psize && f[over] < 1) ++over;
				}

				for (; over < psize; ++over)
				{
					if (f[over] >= 1)
					{
						if (std::is_integral<_Precision>::value)
						{
							arr[over] = std::numeric_limits<_Precision>::max();
						}
						else
						{
							arr[over] = 1;
						}
						alias[over] = over;
					}
				}

				if (under < psize)
				{
					if (std::is_integral<_Precision>::value)
					{
						arr[under] = std::numeric_limits<_Precision>::max();
					}
					else
					{
						arr[under] = 1;
					}
					alias[under] = under;
					for (under = mm; under < msize; ++under)
					{
						if (f[under] < 1)
						{
							if (std::is_integral<_Precision>::value)
							{
								arr[under] = std::numeric_limits<_Precision>::max();
							}
							else
							{
								arr[under] = 1;
							}
							alias[under] = under;
						}
					}
				}
			}

			size_t get_bitsize() const
			{
				return bitsize;
			}

			size_t get_bitmask() const
			{
				return bitmask;
			}

			const _Precision* get_prob() const
			{
				return arr.get();
			}

			const _Size* get_alias() const
			{
				return alias.get();
			}
		};

		template<typename Scalar, typename Rng, typename Precision = float>
		struct scalar_discrete_dist_op;

		template<typename Scalar, typename Rng>
		struct scalar_discrete_dist_op<Scalar, Rng, int32_t> : public scalar_randbits_op<Scalar, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "discreteDist needs integral types.");
			using ur_base = scalar_randbits_op<Scalar, Rng>;

			std::vector<uint32_t> cdf;
			AliasMethod<int32_t, Scalar> alias_table;

			template<typename RealIter>
			scalar_discrete_dist_op(const Rng& _rng, RealIter first, RealIter last)
				: ur_base{ _rng }
			{
				if (std::distance(first, last) < 16)
				{
					// use linear or binary search
					std::vector<double> _cdf;
					double acc = 0;
					for (; first != last; ++first)
					{
						_cdf.emplace_back(acc += *first);
					}

					for (auto& p : _cdf)
					{
						cdf.emplace_back((uint32_t)(p / _cdf.back() * 0x80000000));
					}
				}
				else
				{
					// use alias table
					alias_table = AliasMethod<int32_t, Scalar>{ first, last };
				}
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				if (!cdf.empty())
				{
					auto rx = ur_base::operator()() & 0x7FFFFFFF;
					return (Scalar)(std::lower_bound(cdf.begin(), cdf.end() - 1, rx) - cdf.begin());
				}
				else
				{
					auto rx = ur_base::operator()();
					auto albit = rx & alias_table.get_bitmask();
					uint32_t alx = (uint32_t)(rx >> (sizeof(rx) * 8 - 31));
					if (alx < alias_table.get_prob()[albit]) return albit;
					return alias_table.get_alias()[albit];
				}
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
#ifdef EIGEN_VECTORIZE_AVX2
				thread_local Packet4i cache;
				thread_local const scalar_discrete_dist_op* cache_ptr = nullptr;
				if (cache_ptr == this)
				{
					cache_ptr = nullptr;
					return cache;
				}

				using PacketType = Packet8i;
#else
				using PacketType = Packet;
#endif

				PacketType ret;
				if (!cdf.empty())
				{
					ret = pset1<PacketType>(cdf.size() - 1);
					auto rx = pand(ur_base::template packetOp<PacketType>(), pset1<PacketType>(0x7FFFFFFF));
					for (size_t i = 0; i < cdf.size() - 1; ++i)
					{
						ret = padd(ret, pcmplt(rx, pset1<PacketType>(cdf[i])));
					}
				}
				else
				{
					auto rx = ur_base::template packetOp<PacketType>();
					auto albit = pand(rx, pset1<PacketType>(alias_table.get_bitmask()));
					auto c = pcmplt(psrl(rx, 1), pgather(alias_table.get_prob(), albit));
					ret = pblendv(c, albit, pgather(alias_table.get_alias(), albit));
				}

#ifdef EIGEN_VECTORIZE_AVX2
				cache = _mm256_extractf128_si256(ret, 1);
				cache_ptr = this;
				return _mm256_extractf128_si256(ret, 0);
#else
				return ret;
#endif
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_discrete_dist_op<Scalar, Rng, float> : public scalar_uniform_real_op<float, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "discreteDist needs integral types.");
			using ur_base = scalar_uniform_real_op<float, Rng>;

			std::vector<float> cdf;
			AliasMethod<float, Scalar> alias_table;

			template<typename RealIter>
			scalar_discrete_dist_op(const Rng& _rng, RealIter first, RealIter last)
				: ur_base{ _rng }
			{
				if (std::distance(first, last) < 16)
				{
					// use linear or binary search
					std::vector<double> _cdf;
					double acc = 0;
					for (; first != last; ++first)
					{
						_cdf.emplace_back(acc += *first);
					}

					for (auto& p : _cdf)
					{
						cdf.emplace_back(p / _cdf.back());
					}
				}
				else
				{
					// use alias table
					alias_table = AliasMethod<float, Scalar>{ first, last };
				}
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				if (!cdf.empty())
				{
					auto rx = ur_base::operator()();
					return (Scalar)(std::lower_bound(cdf.begin(), cdf.end() - 1, rx) - cdf.begin());
				}
				else
				{
					auto albit = pfirst(this->rng()) & alias_table.get_bitmask();
					auto alx = ur_base::operator()();
					if (alx < alias_table.get_prob()[albit]) return albit;
					return alias_table.get_alias()[albit];
				}
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));

				if (!cdf.empty())
				{
					auto ret = pset1<Packet>(cdf.size());
					auto rx = ur_base::template packetOp<PacketType>();
					for (auto& p : cdf)
					{
						ret = padd(ret, reinterpret_to_int(pcmplt(rx, pset1<PacketType>(p))));
					}
					return ret;
				}
				else
				{
					using RUtils = RawbitsMaker<Packet, Rng>;
					auto albit = pand(RUtils{}.rawbits(this->rng), pset1<Packet>(alias_table.get_bitmask()));
					auto c = reinterpret_to_int(pcmplt(ur_base::template packetOp<PacketType>(), pgather(alias_table.get_prob(), albit)));
					return pblendv(c, albit, pgather(alias_table.get_alias(), albit));
				}
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_discrete_dist_op<Scalar, Rng, double> : public scalar_uniform_real_op<double, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "discreteDist needs integral types.");
			using ur_base = scalar_uniform_real_op<double, Rng>;

			std::vector<double> cdf;
			AliasMethod<double, Scalar> alias_table;

			template<typename RealIter>
			scalar_discrete_dist_op(const Rng& _rng, RealIter first, RealIter last)
				: ur_base{ _rng }
			{
				if (std::distance(first, last) < 16)
				{
					// use linear or binary search
					std::vector<double> _cdf;
					double acc = 0;
					for (; first != last; ++first)
					{
						_cdf.emplace_back(acc += *first);
					}

					for (auto& p : _cdf)
					{
						cdf.emplace_back(p / _cdf.back());
					}
				}
				else
				{
					// use alias table
					alias_table = AliasMethod<double, Scalar>{ first, last };
				}
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				if (!cdf.empty())
				{
					auto rx = ur_base::operator()();
					return (Scalar)(std::lower_bound(cdf.begin(), cdf.end() - 1, rx) - cdf.begin());
				}
				else
				{
					auto albit = pfirst(this->rng()) & alias_table.get_bitmask();
					auto alx = ur_base::operator()();
					if (alx < alias_table.get_prob()[albit]) return albit;
					return alias_table.get_alias()[albit];
				}
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using DPacket = decltype(reinterpret_to_double(std::declval<Packet>()));
				if (!cdf.empty())
				{
					auto ret = pset1<Packet>(cdf.size());
#ifdef EIGEN_VECTORIZE_AVX
					auto rx = ur_base::template packetOp<Packet4d>();
					for (auto& p : cdf)
					{
						auto c = reinterpret_to_int(pcmplt(rx, pset1<decltype(rx)>(p)));
						auto r = combine_low32(c);
						ret = padd(ret, r);
					}
#else
					auto rx1 = ur_base::template packetOp<DPacket>(),
						rx2 = ur_base::template packetOp<DPacket>();
					for (auto& p : cdf)
					{
						auto pp = pset1<decltype(rx1)>(p);
						ret = padd(ret, combine_low32(reinterpret_to_int(pcmplt(rx1, pp)), reinterpret_to_int(pcmplt(rx2, pp))));
					}
#endif
					return ret;
				}
				else
				{
#ifdef EIGEN_VECTORIZE_AVX
					using RUtils = RawbitsMaker<Packet, Rng>;
					auto albit = pand(RUtils{}.rawbits(this->rng), pset1<Packet>(alias_table.get_bitmask()));
					auto c = reinterpret_to_int(pcmplt(ur_base::template packetOp<Packet4d>(), pgather(alias_table.get_prob(), _mm256_castsi128_si256(albit))));
					return pblendv(combine_low32(c), albit, pgather(alias_table.get_alias(), albit));
#else
					using RUtils = RawbitsMaker<Packet, Rng>;
					auto albit = pand(RUtils{}.rawbits(this->rng), pset1<Packet>(alias_table.get_bitmask()));
					auto c1 = reinterpret_to_int(pcmplt(ur_base::template packetOp<DPacket>(), pgather(alias_table.get_prob(), albit)));
					auto c2 = reinterpret_to_int(pcmplt(ur_base::template packetOp<DPacket>(), pgather(alias_table.get_prob(), albit, true)));
					return pblendv(combine_low32(c1, c2), albit, pgather(alias_table.get_alias(), albit));
#endif
				}
			}
		};

		template<typename Scalar, typename Urng, typename Precision>
		struct functor_traits<scalar_discrete_dist_op<Scalar, Urng, Precision> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Rng>
		struct scalar_poisson_dist_op : public scalar_uniform_real_op<float, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "poisson needs integral types.");
			using ur_base = scalar_uniform_real_op<float, Rng>;

			double mean, ne_mean, sqrt_tmean, log_mean, g1;

			scalar_poisson_dist_op(const Rng& _rng, double _mean)
				: ur_base{ _rng }, mean{ _mean }, ne_mean{ std::exp(-_mean) }
			{
				sqrt_tmean = std::sqrt(2 * mean);
				log_mean = std::log(mean);
				g1 = mean * log_mean - std::lgamma(mean + 1);
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				if (mean < 12)
				{
					Scalar res = 0;
					double val = 1;
					for (; ; ++res)
					{
						val *= ur_base::operator()();
						if (val <= ne_mean) break;
					}
					return res;
				}
				else
				{
					Scalar res;
					double yx;
					while(1)
					{
						yx = std::tan(constant::pi * ur_base::operator()());
						res = (Scalar)(sqrt_tmean * yx + mean);
						if (res >= 0 && ur_base::operator()() <= 0.9 * (1.0 + yx * yx)
							* std::exp(res * log_mean - std::lgamma(res + 1.0) - g1))
						{
							return res;
						}
					}
				}
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));

				if (mean < 12)
				{
					Packet res = pset1<Packet>(0);
					PacketType val = pset1<PacketType>(1), pne_mean = pset1<PacketType>(ne_mean);
					while (1)
					{
						val = pmul(val, ur_base::template packetOp<PacketType>());
						auto c = reinterpret_to_int(pcmplt(pne_mean, val));
						if (pmovemask(c) == 0) break;
						res = padd(res, pnegate(c));
					}
					return res;
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					thread_local PacketType cache_rest;
					thread_local int cache_rest_cnt;
					thread_local const scalar_poisson_dist_op* cache_ptr = nullptr;
					if (cache_ptr != this)
					{
						cache_ptr = this;
						cache_rest = pset1<PacketType>(0);
						cache_rest_cnt = 0;
					}

					const PacketType ppi = pset1<PacketType>(constant::pi),
						psqrt_tmean = pset1<PacketType>(sqrt_tmean),
						pmean = pset1<PacketType>(mean),
						plog_mean = pset1<PacketType>(log_mean),
						pg1 = pset1<PacketType>(g1);
					while (1)
					{
						PacketType fres, yx, psin, pcos;
						psincos(pmul(ppi, ur_base::template packetOp<PacketType>()), psin, pcos);
						yx = pdiv(psin, pcos);
						fres = ptruncate(padd(pmul(psqrt_tmean, yx), pmean));

						auto p1 = pmul(padd(pmul(yx, yx), pset1<PacketType>(1)), pset1<PacketType>(0.9));
						auto p2 = pexp(psub(psub(pmul(fres, plog_mean), plgamma(padd(fres, pset1<PacketType>(1)))), pg1));

						auto c1 = pcmple(pset1<PacketType>(0), fres);
						auto c2 = pcmple(ur_base::template packetOp<PacketType>(), pmul(p1, p2));

						auto cands = fres;
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pand(c1, c2),
							cache_rest, cache_rest_cnt, full);
						if (full) return pcast<PacketType, Packet>(cands);
					}
				}
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_poisson_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Rng>
		struct scalar_binomial_dist_op : public scalar_poisson_dist_op<Scalar, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "binomial needs integral types.");
			using ur_base = scalar_uniform_real_op<float, Rng>;

			Scalar trials;
			double p, small_p, g1, sqrt_v, log_small_p, log_small_q;

			scalar_binomial_dist_op(const Rng& _rng, Scalar _trials = 1, double _p = 0.5)
				: scalar_poisson_dist_op<Scalar, Rng>{ _rng, _trials * std::min(_p, 1 - _p) }, 
				trials{ _trials }, p{ _p }, small_p{ std::min(p, 1 - p) }
				
			{
				g1 = std::lgamma(trials + 1);
				sqrt_v = std::sqrt(2 * this->mean * (1 - small_p));
				log_small_p = std::log(small_p);
				log_small_q = std::log(1 - small_p);
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				Scalar res;
				if (trials < 25)
				{
					res = 0;
					for (int i = 0; i < trials; ++i)
					{
						if (ur_base::operator()() < p) ++res;
					}
					return res;
				}
				else if (this->mean < 1.0)
				{
					res = scalar_poisson_dist_op<Scalar, Rng>::operator()();
				}
				else
				{
					while(1)
					{
						double ys;
						ys = std::tan(constant::pi * ur_base::operator()());
						res = (Scalar)(sqrt_v * ys + this->mean);
						if (0 <= res && res <= trials && ur_base::operator()() <= 1.2 * sqrt_v
							* (1.0 + ys * ys)
							* std::exp(g1 - std::lgamma(res + 1)
								- std::lgamma(trials - res + 1.0)
								+ res * log_small_p
								+ (trials - res) * log_small_q)
							)
						{
							break;
						}
					}
				}
				return p == small_p ? res : trials - res;
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));
				Packet res;
				if (trials < 25)
				{
					PacketType pp = pset1<PacketType>(p);
					res = pset1<Packet>(trials);
					for (int i = 0; i < trials; ++i)
					{
						auto c = reinterpret_to_int(pcmple(pp, ur_base::template packetOp<PacketType>()));
						res = padd(res, c);
					}
					return res;
				}
				else if (this->mean < 1.0)
				{
					res = scalar_poisson_dist_op<Scalar, Rng>::template packetOp<Packet>();
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					thread_local PacketType cache_rest;
					thread_local int cache_rest_cnt;
					thread_local const scalar_binomial_dist_op* cache_ptr = nullptr;
					if (cache_ptr != this)
					{
						cache_ptr = this;
						cache_rest = pset1<PacketType>(0);
						cache_rest_cnt = 0;
					}

					const PacketType ppi = pset1<PacketType>(constant::pi),
						ptrials = pset1<PacketType>(trials),
						psqrt_v = pset1<PacketType>(sqrt_v),
						pmean = pset1<PacketType>(this->mean),
						plog_small_p = pset1<PacketType>(log_small_p),
						plog_small_q = pset1<PacketType>(log_small_q),
						pg1 = pset1<PacketType>(g1);
					while (1)
					{
						PacketType fres, ys, psin, pcos;
						psincos(pmul(ppi, ur_base::template packetOp<PacketType>()), psin, pcos);
						ys = pdiv(psin, pcos);
						fres = ptruncate(padd(pmul(psqrt_v, ys), pmean));
						
						auto p1 = pmul(pmul(pset1<PacketType>(1.2), psqrt_v), padd(pset1<PacketType>(1), pmul(ys, ys)));
						auto p2 = pexp(
							padd(padd(psub(
								psub(pg1, plgamma(padd(fres, pset1<PacketType>(1)))),
								plgamma(psub(padd(ptrials, pset1<PacketType>(1)), fres))
							), pmul(fres, plog_small_p)), pmul(psub(ptrials, fres), plog_small_q))
						);

						auto c1 = pand(pcmple(pset1<PacketType>(0), fres), pcmple(fres, ptrials));
						auto c2 = pcmple(ur_base::template packetOp<PacketType>(), pmul(p1, p2));

						auto cands = fres;
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pand(c1, c2),
							cache_rest, cache_rest_cnt, full);
						if (full)
						{
							res = pcast<PacketType, Packet>(cands);
							break;
						}
					}
				}
				return p == small_p ? res : psub(pset1<Packet>(trials), res);
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_binomial_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

		template<typename Scalar, typename Rng>
		struct scalar_geometric_dist_op : public scalar_uniform_real_op<float, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "geomtric needs integral types.");
			using ur_base = scalar_uniform_real_op<float, Rng>;

			double p, rlog_q;

			scalar_geometric_dist_op(const Rng& _rng, double _p)
				: ur_base{ _rng }, p{ _p }, rlog_q{ 1 / std::log(1 - p) }
			{
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				return (Scalar)(std::log(1 - ur_base::operator()()) * rlog_q);
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));

				return pcast<PacketType, Packet>(ptruncate(pmul(plog(
					psub(pset1<PacketType>(1), ur_base::template packetOp<PacketType>())
				), pset1<PacketType>(rlog_q))));
			}
		};

		template<typename Scalar, typename Urng>
		struct functor_traits<scalar_geometric_dist_op<Scalar, Urng> >
		{
			enum { Cost = HugeCost, PacketAccess = packet_traits<Scalar>::Vectorizable, IsRepeatable = false };
		};

	}
}

#endif
