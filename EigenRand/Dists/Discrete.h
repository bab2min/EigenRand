/**
 * @file Discrete.h
 * @author bab2min (bab2min@gmail.com)
 * @brief 
 * @version 0.4.1
 * @date 2022-08-13
 *
 * @copyright Copyright (c) 2020-2021
 * 
 */

#ifndef EIGENRAND_DISTS_DISCRETE_H
#define EIGENRAND_DISTS_DISCRETE_H

#include <memory>
#include <iterator>
#include <limits>

namespace Eigen
{
	namespace Rand
	{
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
					arr = std::unique_ptr<_Precision[]>(new _Precision[(size_t)1 << bitsize]);
					alias = std::unique_ptr<_Size[]>(new _Size[(size_t)1 << bitsize]);

					std::copy(o.arr.get(), o.arr.get() + ((size_t)1 << bitsize), arr.get());
					std::copy(o.alias.get(), o.alias.get() + ((size_t)1 << bitsize), alias.get());
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
					alias[under] = (_Size)over;
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
						alias[over] = (_Size)over;
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
					alias[under] = (_Size)under;
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
							alias[under] = (_Size)under;
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
		
		/**
		 * @brief Generator of integers with a given range `[min, max]`
		 * 
		 * @tparam _Scalar any integral type
		 */
		template<typename _Scalar>
		class UniformIntGen : OptCacheStore, public GenBase<UniformIntGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "uniformInt needs integral types.");
			int cache_rest_cnt = 0;
			RandbitsGen<_Scalar> randbits;
			_Scalar pmin;
			size_t pdiff, bitsize, bitmask;
		
		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new UniformInt Generator
			 * 
			 * @param _min, _max the range of integers being generated
			 */
			UniformIntGen(_Scalar _min = 0, _Scalar _max = 0)
				: pmin{ _min }, pdiff{ (size_t)(_max - _min) }
			{
				if ((pdiff + 1) > pdiff)
				{
					bitsize = (size_t)std::ceil(std::log2(pdiff + 1));
				}
				else
				{
					bitsize = (size_t)std::ceil(std::log2(pdiff));
				}
				bitmask = (_Scalar)(((size_t)-1) >> (sizeof(size_t) * 8 - bitsize));
			}

			UniformIntGen(const UniformIntGen&) = default;
			UniformIntGen(UniformIntGen&&) = default;

			UniformIntGen& operator=(const UniformIntGen&) = default;
			UniformIntGen& operator=(UniformIntGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				auto rx = randbits(rng);
				if (pdiff == bitmask)
				{
					return (_Scalar)(rx & bitmask) + pmin;
				}
				else
				{
					size_t bitcnt = bitsize;
					for (int _i = 0; ; ++_i)
					{
						EIGENRAND_CHECK_INFINITY_LOOP();
						_Scalar cands = (_Scalar)(rx & bitmask);
						if (cands <= pdiff) return cands + pmin;
						if (bitcnt + bitsize < 32)
						{
							rx >>= bitsize;
							bitcnt += bitsize;
						}
						else
						{
							rx = randbits(rng);
							bitcnt = bitsize;
						}
					}
				}
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				auto rx = randbits.template packetOp<Packet>(rng);
				auto pbitmask = pset1<Packet>(bitmask);
				if (pdiff == bitmask)
				{
					return padd(pand(rx, pbitmask), pset1<Packet>(pmin));
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					auto plen = pset1<Packet>(pdiff + 1);
					size_t bitcnt = bitsize;
					for (int _i = 0; ; ++_i)
					{
						EIGENRAND_CHECK_INFINITY_LOOP();
						// accept cands that only < plen
						auto cands = pand(rx, pbitmask);
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pcmplt(cands, plen),
							OptCacheStore::template get<Packet>(), cache_rest_cnt, full);
						if (full) return padd(cands, pset1<Packet>(pmin));

						if (bitcnt + bitsize < 32)
						{
							rx = psrl<-1>(rx, bitsize);
							bitcnt += bitsize;
						}
						else
						{
							rx = randbits.template packetOp<Packet>(rng);
							bitcnt = bitsize;
						}
					}
				}
			}
		};

		/**
		 * @brief Generator of integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`
		 * 
		 * @tparam _Scalar any integral type
		 * @tparam Precision internal precision type
		 */
		template<typename _Scalar, typename Precision = float>
		class DiscreteGen;

		/**
		 * @brief `DiscreteGen` with `int32_t` precision
		 * 
		 * @tparam _Scalar any intergral type
		 */
		template<typename _Scalar>
		class DiscreteGen<_Scalar, int32_t> : public GenBase<DiscreteGen<_Scalar, int32_t>, _Scalar>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "discreteDist needs integral types.");
#ifdef EIGEN_VECTORIZE_AVX2
			OptCacheStore cache;
			bool valid = false;
#endif
			RandbitsGen<int32_t> randbits;
			std::vector<uint32_t> cdf;
			AliasMethod<int32_t, _Scalar> alias_table;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Discrete Generator
			 * 
			 * @tparam RealIter 
			 * @param first, last the range of elements defining the numbers to use as weights. 
			 * The type of the elements referred by it must be convertible to `double`.
			 */
			template<typename RealIter>
			DiscreteGen(RealIter first, RealIter last)
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
					alias_table = AliasMethod<int32_t, _Scalar>{ first, last };
				}
			}

			/**
			 * @brief Construct a new Discrete Generator
			 * 
			 * @tparam Real 
			 * @param il an instance of initializer_list containing the numbers to use as weights. 
			 * The type of the elements referred by it must be convertible to `double`.
			 */
			template<typename Real,
				typename std::enable_if<std::is_arithmetic<Real>::value, int>::type = 0>
			DiscreteGen(const std::initializer_list<Real>& il)
				: DiscreteGen(il.begin(), il.end())
			{
			}

			DiscreteGen()
				: DiscreteGen({ 1 })
			{
			}

			DiscreteGen(const DiscreteGen&) = default;
			DiscreteGen(DiscreteGen&&) = default;

			DiscreteGen& operator=(const DiscreteGen&) = default;
			DiscreteGen& operator=(DiscreteGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				if (!cdf.empty())
				{
					auto rx = randbits(std::forward<Rng>(rng)) & 0x7FFFFFFF;
					return (_Scalar)(std::lower_bound(cdf.begin(), cdf.end() - 1, rx) - cdf.begin());
				}
				else
				{
					auto rx = randbits(std::forward<Rng>(rng));
					auto albit = rx & alias_table.get_bitmask();
					uint32_t alx = (uint32_t)(rx >> (sizeof(rx) * 8 - 31));
					if (alx < alias_table.get_prob()[albit]) return (_Scalar)albit;
					return alias_table.get_alias()[albit];
				}
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
#ifdef EIGEN_VECTORIZE_AVX2
				if (valid)
				{
					valid = false;
					return cache.template get<Packet>();
				}
				using PacketType = Packet8i;
#else
				using PacketType = Packet;
#endif

				PacketType ret;
				if (!cdf.empty())
				{
					ret = pset1<PacketType>(cdf.size() - 1);
					auto rx = pand(randbits.template packetOp<PacketType>(std::forward<Rng>(rng)), pset1<PacketType>(0x7FFFFFFF));
					for (size_t i = 0; i < cdf.size() - 1; ++i)
					{
						ret = padd(ret, pcmplt(rx, pset1<PacketType>(cdf[i])));
					}
				}
				else
				{
					auto rx = randbits.template packetOp<PacketType>(std::forward<Rng>(rng));
					auto albit = pand(rx, pset1<PacketType>(alias_table.get_bitmask()));
					auto c = pcmplt(psrl<1>(rx), pgather(alias_table.get_prob(), albit));
					ret = pblendv(c, albit, pgather(alias_table.get_alias(), albit));
				}

#ifdef EIGEN_VECTORIZE_AVX2
				valid = true;
				cache.template get<Packet>() = _mm256_extractf128_si256(ret, 1);
				return _mm256_extractf128_si256(ret, 0);
#else
				return ret;
#endif
			}
		};

		/**
		 * @brief `DiscreteGen` with `float` precision
		 * 
		 * @tparam _Scalar any intergral type
		 */
		template<typename _Scalar>
		class DiscreteGen<_Scalar, float> : public GenBase<DiscreteGen<_Scalar, float>, _Scalar>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "discreteDist needs integral types.");
			UniformRealGen<float> ur;
			std::vector<float> cdf;
			AliasMethod<float, _Scalar> alias_table;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Discrete Generator
			 * 
			 * @tparam RealIter 
			 * @param first, last the range of elements defining the numbers to use as weights. 
			 * The type of the elements referred by it must be convertible to `double`.
			 */
			template<typename RealIter>
			DiscreteGen(RealIter first, RealIter last)
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
					alias_table = AliasMethod<float, _Scalar>{ first, last };
				}
			}

			/**
			 * @brief Construct a new Discrete Generator
			 * 
			 * @tparam Real 
			 * @param il an instance of initializer_list containing the numbers to use as weights. 
			 * The type of the elements referred by it must be convertible to `double`.
			 */
			template<typename Real, 
				typename std::enable_if<std::is_arithmetic<Real>::value, int>::type = 0>
			DiscreteGen(const std::initializer_list<Real>& il)
				: DiscreteGen(il.begin(), il.end())
			{
			}

			DiscreteGen()
				: DiscreteGen({ 1 })
			{
			}

			DiscreteGen(const DiscreteGen&) = default;
			DiscreteGen(DiscreteGen&&) = default;

			DiscreteGen& operator=(const DiscreteGen&) = default;
			DiscreteGen& operator=(DiscreteGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				if (!cdf.empty())
				{
					auto rx = ur(std::forward<Rng>(rng));
					return (_Scalar)(std::lower_bound(cdf.begin(), cdf.end() - 1, rx) - cdf.begin());
				}
				else
				{
					auto albit = pfirst(rng()) & alias_table.get_bitmask();
					auto alx = ur(rng);
					if (alx < alias_table.get_prob()[albit]) return albit;
					return alias_table.get_alias()[albit];
				}
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));

				if (!cdf.empty())
				{
					auto ret = pset1<Packet>(cdf.size());
					auto rx = ur.template packetOp<PacketType>(std::forward<Rng>(rng));
					for (auto& p : cdf)
					{
						ret = padd(ret, reinterpret_to_int(pcmplt(rx, pset1<PacketType>(p))));
					}
					return ret;
				}
				else
				{
					using RUtils = RawbitsMaker<Packet, Rng>;
					auto albit = pand(RUtils{}.rawbits(rng), pset1<Packet>(alias_table.get_bitmask()));
					auto c = reinterpret_to_int(pcmplt(ur.template packetOp<PacketType>(rng), pgather(alias_table.get_prob(), albit)));
					return pblendv(c, albit, pgather(alias_table.get_alias(), albit));
				}
			}
		};

		/**
		 * @brief `DiscreteGen` with `double` precision
		 * 
		 * @tparam _Scalar any intergral type
		 */
		template<typename _Scalar>
		class DiscreteGen<_Scalar, double> : public GenBase<DiscreteGen<_Scalar, double>, _Scalar>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "discreteDist needs integral types.");
			UniformRealGen<double> ur;
			std::vector<double> cdf;
			AliasMethod<double, _Scalar> alias_table;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Discrete Generator
			 * 
			 * @tparam RealIter 
			 * @param first, last the range of elements defining the numbers to use as weights. 
			 * The type of the elements referred by it must be convertible to `double`.
			 */
			template<typename RealIter>
			DiscreteGen(RealIter first, RealIter last)
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
					alias_table = AliasMethod<double, _Scalar>{ first, last };
				}
			}

			/**
			 * @brief Construct a new Discrete Generator
			 * 
			 * @tparam Real 
			 * @param il an instance of initializer_list containing the numbers to use as weights. 
			 * The type of the elements referred by it must be convertible to `double`.
			 */
			template<typename Real,
				typename std::enable_if<std::is_arithmetic<Real>::value, int>::type = 0>
			DiscreteGen(const std::initializer_list<Real>& il)
				: DiscreteGen(il.begin(), il.end())
			{
			}

			DiscreteGen()
				: DiscreteGen({ 1 })
			{
			}

			DiscreteGen(const DiscreteGen&) = default;
			DiscreteGen(DiscreteGen&&) = default;

			DiscreteGen& operator=(const DiscreteGen&) = default;
			DiscreteGen& operator=(DiscreteGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				if (!cdf.empty())
				{
					auto rx = ur(std::forward<Rng>(rng));
					return (_Scalar)(std::lower_bound(cdf.begin(), cdf.end() - 1, rx) - cdf.begin());
				}
				else
				{
					auto albit = pfirst(rng()) & alias_table.get_bitmask();
					auto alx = ur(rng);
					if (alx < alias_table.get_prob()[albit]) return albit;
					return alias_table.get_alias()[albit];
				}
			}

#ifdef EIGEN_VECTORIZE_NEON
#else
			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using DPacket = decltype(reinterpret_to_double(std::declval<Packet>()));
				if (!cdf.empty())
				{
					auto ret = pset1<Packet>(cdf.size());
	#ifdef EIGEN_VECTORIZE_AVX
					auto rx = ur.template packetOp<Packet4d>(std::forward<Rng>(rng));
					for (auto& p : cdf)
					{
						auto c = reinterpret_to_int(pcmplt(rx, pset1<decltype(rx)>(p)));
						auto r = combine_low32(c);
						ret = padd(ret, r);
					}
	#else
					auto rx1 = ur.template packetOp<DPacket>(rng),
						rx2 = ur.template packetOp<DPacket>(rng);
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
					auto albit = pand(RUtils{}.rawbits(rng), pset1<Packet>(alias_table.get_bitmask()));
					auto c = reinterpret_to_int(pcmplt(ur.template packetOp<Packet4d>(rng), pgather(alias_table.get_prob(), _mm256_castsi128_si256(albit))));
					return pblendv(combine_low32(c), albit, pgather(alias_table.get_alias(), albit));
	#else
					using RUtils = RawbitsMaker<Packet, Rng>;
					auto albit = pand(RUtils{}.rawbits(rng), pset1<Packet>(alias_table.get_bitmask()));
					auto c1 = reinterpret_to_int(pcmplt(ur.template packetOp<DPacket>(rng), pgather(alias_table.get_prob(), albit)));
					auto c2 = reinterpret_to_int(pcmplt(ur.template packetOp<DPacket>(rng), pgather(alias_table.get_prob(), albit, true)));
					return pblendv(combine_low32(c1, c2), albit, pgather(alias_table.get_alias(), albit));
	#endif
				}
			}
#endif
		};

		template<typename> class BinomialGen;

		/**
		 * @brief Generator of integers on a Poisson distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class PoissonGen : OptCacheStore, public GenBase<PoissonGen<_Scalar>, _Scalar>
		{
			friend BinomialGen<_Scalar>;
			static_assert(std::is_same<_Scalar, int32_t>::value, "poisson needs integral types.");
			int cache_rest_cnt = 0;
			UniformRealGen<float> ur;

		protected:
			double mean, ne_mean, sqrt_tmean, log_mean, g1;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Poisson Generator
			 * 
			 * @param _mean mean of the distribution
			 */
			PoissonGen(double _mean = 1)
				: mean{ _mean }, ne_mean{ std::exp(-_mean) }
			{
				sqrt_tmean = std::sqrt(2 * mean);
				log_mean = std::log(mean);
				g1 = mean * log_mean - std::lgamma(mean + 1);
			}

			PoissonGen(const PoissonGen&) = default;
			PoissonGen(PoissonGen&&) = default;

			PoissonGen& operator=(const PoissonGen&) = default;
			PoissonGen& operator=(PoissonGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				if (mean < 12)
				{
					_Scalar res = 0;
					double val = 1;
					for (; ; ++res)
					{
						val *= ur(rng);
						if (val <= ne_mean) break;
					}
					return res;
				}
				else
				{
					_Scalar res;
					double yx;
					for (int _i = 0; ; ++_i)
					{
						EIGENRAND_CHECK_INFINITY_LOOP();
						yx = std::tan(constant::pi * ur(rng));
						res = (_Scalar)(sqrt_tmean * yx + mean);
						if (res >= 0 && ur(rng) <= 0.9 * (1.0 + yx * yx)
							* std::exp(res * log_mean - std::lgamma(res + 1.0) - g1))
						{
							return res;
						}
					}
				}
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));

				if (mean < 12)
				{
					Packet res = pset1<Packet>(0);
					PacketType val = pset1<PacketType>(1), pne_mean = pset1<PacketType>(ne_mean);
					for (int _i = 0; ; ++_i)
					{
						EIGENRAND_CHECK_INFINITY_LOOP();
						val = pmul(val, ur.template packetOp<PacketType>(rng));
						auto c = reinterpret_to_int(pcmplt(pne_mean, val));
						if (pmovemask(c) == 0) break;
						res = padd(res, pnegate(c));
					}
					return res;
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					const PacketType ppi = pset1<PacketType>(constant::pi),
						psqrt_tmean = pset1<PacketType>(sqrt_tmean),
						pmean = pset1<PacketType>(mean),
						plog_mean = pset1<PacketType>(log_mean),
						pg1 = pset1<PacketType>(g1);
					for (int _i = 0; ; ++_i)
					{
						EIGENRAND_CHECK_INFINITY_LOOP();
						PacketType fres, yx, psin, pcos;
						psincos(pmul(ppi, ur.template packetOp<PacketType>(rng)), psin, pcos);
						yx = pdiv(psin, pcos);
						fres = ptruncate(padd(pmul(psqrt_tmean, yx), pmean));

						auto p1 = pmul(padd(pmul(yx, yx), pset1<PacketType>(1)), pset1<PacketType>(0.9));
						auto p2 = pexp(psub(psub(pmul(fres, plog_mean), plgamma_approx(padd(fres, pset1<PacketType>(1)))), pg1));

						auto c1 = pcmple(pset1<PacketType>(0), fres);
						auto c2 = pcmple(ur.template packetOp<PacketType>(rng), pmul(p1, p2));

						auto cands = fres;
						bool full = false;
						cache_rest_cnt = cm.compress_append(cands, pand(c1, c2),
							OptCacheStore::template get<PacketType>(), cache_rest_cnt, full);
						if (full) return pcast<PacketType, Packet>(cands);
					}
				}
			}
		};

		/**
		 * @brief Generator of integers on a binomial distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class BinomialGen : public GenBase<BinomialGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "binomial needs integral types.");

			PoissonGen<_Scalar> poisson;
			_Scalar trials;
			double p = 0, small_p = 0, g1 = 0, sqrt_v = 0, log_small_p = 0, log_small_q = 0;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Binomial Generator
			 * 
			 * @param _trials the number of trials
			 * @param _p probability of a trial generating true
			 */
			BinomialGen(_Scalar _trials = 1, double _p = 0.5)
				: poisson{ _trials * std::min(_p, 1 - _p) },
				trials{ _trials }, p{ _p }, small_p{ std::min(p, 1 - p) }

			{
				if (!(trials < 25 || poisson.mean < 1.0))
				{
					g1 = std::lgamma(trials + 1);
					sqrt_v = std::sqrt(2 * poisson.mean * (1 - small_p));
					log_small_p = std::log(small_p);
					log_small_q = std::log(1 - small_p);
				}
			}

			BinomialGen(const BinomialGen&) = default;
			BinomialGen(BinomialGen&&) = default;

			BinomialGen& operator=(const BinomialGen&) = default;
			BinomialGen& operator=(BinomialGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				_Scalar res;
				if (trials < 25)
				{
					res = 0;
					for (int i = 0; i < trials; ++i)
					{
						if (poisson.ur(rng) < p) ++res;
					}
					return res;
				}
				else if (poisson.mean < 1.0)
				{
					res = poisson(rng);
				}
				else
				{
					for (int _i = 0; ; ++_i)
					{
						EIGENRAND_CHECK_INFINITY_LOOP();
						double ys;
						ys = std::tan(constant::pi * poisson.ur(rng));
						res = (_Scalar)(sqrt_v * ys + poisson.mean);
						if (0 <= res && res <= trials && poisson.ur(rng) <= 1.2 * sqrt_v
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

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));
				Packet res;
				if (trials < 25)
				{
					PacketType pp = pset1<PacketType>(p);
					res = pset1<Packet>(trials);
					for (int i = 0; i < trials; ++i)
					{
						auto c = reinterpret_to_int(pcmple(pp, poisson.ur.template packetOp<PacketType>(rng)));
						res = padd(res, c);
					}
					return res;
				}
				else if (poisson.mean < 1.0)
				{
					res = poisson.template packetOp<Packet>(rng);
				}
				else
				{
					auto& cm = Rand::detail::CompressMask<sizeof(Packet)>::get_inst();
					const PacketType ppi = pset1<PacketType>(constant::pi),
						ptrials = pset1<PacketType>(trials),
						psqrt_v = pset1<PacketType>(sqrt_v),
						pmean = pset1<PacketType>(poisson.mean),
						plog_small_p = pset1<PacketType>(log_small_p),
						plog_small_q = pset1<PacketType>(log_small_q),
						pg1 = pset1<PacketType>(g1);
					for (int _i = 0; ; ++_i)
					{
						EIGENRAND_CHECK_INFINITY_LOOP();
						PacketType fres, ys, psin, pcos;
						psincos(pmul(ppi, poisson.ur.template packetOp<PacketType>(rng)), psin, pcos);
						ys = pdiv(psin, pcos);
						fres = ptruncate(padd(pmul(psqrt_v, ys), pmean));

						auto p1 = pmul(pmul(pset1<PacketType>(1.2), psqrt_v), padd(pset1<PacketType>(1), pmul(ys, ys)));
						auto p2 = pexp(
							padd(padd(psub(
								psub(pg1, plgamma_approx(padd(fres, pset1<PacketType>(1)))),
								plgamma_approx(psub(padd(ptrials, pset1<PacketType>(1)), fres))
							), pmul(fres, plog_small_p)), pmul(psub(ptrials, fres), plog_small_q))
						);

						auto c1 = pand(pcmple(pset1<PacketType>(0), fres), pcmple(fres, ptrials));
						auto c2 = pcmple(poisson.ur.template packetOp<PacketType>(rng), pmul(p1, p2));

						auto cands = fres;
						bool full = false;
						poisson.cache_rest_cnt = cm.compress_append(cands, pand(c1, c2),
							poisson.template get<PacketType>(), poisson.cache_rest_cnt, full);
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

		/**
		 * @brief Generator of integers on a geometric distribution
		 * 
		 * @tparam _Scalar 
		 */
		template<typename _Scalar>
		class GeometricGen : public GenBase<GeometricGen<_Scalar>, _Scalar>
		{
			static_assert(std::is_same<_Scalar, int32_t>::value, "geomtric needs integral types.");
			UniformRealGen<float> ur;
			double p, rlog_q;

		public:
			using Scalar = _Scalar;

			/**
			 * @brief Construct a new Geometric Generator
			 * 
			 * @param _p probability of a trial generating true
			 */
			GeometricGen(double _p = 0.5)
				: p{ _p }, rlog_q{ 1 / std::log(1 - p) }
			{
			}

			GeometricGen(const GeometricGen&) = default;
			GeometricGen(GeometricGen&&) = default;

			GeometricGen& operator=(const GeometricGen&) = default;
			GeometricGen& operator=(GeometricGen&&) = default;

			template<typename Rng>
			EIGEN_STRONG_INLINE const _Scalar operator() (Rng&& rng)
			{
				using namespace Eigen::internal;
				return (_Scalar)(std::log(1 - ur(std::forward<Rng>(rng))) * rlog_q);
			}

			template<typename Packet, typename Rng>
			EIGEN_STRONG_INLINE const Packet packetOp(Rng&& rng)
			{
				using namespace Eigen::internal;
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));

				return pcast<PacketType, Packet>(ptruncate(pmul(plog(
					psub(pset1<PacketType>(1), ur.template packetOp<PacketType>(std::forward<Rng>(rng)))
				), pset1<PacketType>(rlog_q))));
			}
		};


		template<typename Derived, typename Urng>
		using UniformIntType = CwiseNullaryOp<internal::scalar_rng_adaptor<UniformIntGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates integers with a given range `[min, max]`
		 *
		 * @tparam Derived a type of Eigen::DenseBase
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param min, max the range of integers being generated
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::UniformIntGen
		 */
		template<typename Derived, typename Urng>
		inline const UniformIntType<Derived, Urng>
			uniformInt(Index rows, Index cols, Urng&& urng, typename Derived::Scalar min, typename Derived::Scalar max)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), UniformIntGen<typename Derived::Scalar>{ min, max } }
			};
		}

		/**
		 * @brief generates integers with a given range `[min, max]`
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param min, max the range of integers being generated
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::UniformIntGen
		 */
		template<typename Derived, typename Urng>
		inline const UniformIntType<Derived, Urng>
			uniformIntLike(Derived& o, Urng&& urng, typename Derived::Scalar min, typename Derived::Scalar max)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), UniformIntGen<typename Derived::Scalar>{ min, max } }
			};
		}

		template<typename Derived, typename Urng>
		using DiscreteFType = CwiseNullaryOp<internal::scalar_rng_adaptor<DiscreteGen<typename Derived::Scalar, float>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteFType<Derived, Urng>
			discreteF(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, float>{first, last} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteFType<Derived, Urng>
			discreteFLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, float>(first, last) }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by it must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteFType<Derived, Urng>
			discreteF(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, float>{il.begin(), il.end()} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is float(23bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by it must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteFType<Derived, Urng>
			discreteFLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, float>(il.begin(), il.end()) }
			};
		}

		template<typename Derived, typename Urng>
		using DiscreteDType = CwiseNullaryOp<internal::scalar_rng_adaptor<DiscreteGen<typename Derived::Scalar, double>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteDType<Derived, Urng>
			discreteD(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, double>{first, last} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteDType<Derived, Urng>
			discreteDLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, double>{first, last} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by it must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteDType<Derived, Urng>
			discreteD(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, double>{il.begin(), il.end()} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is double(52bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by it must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteDType<Derived, Urng>
			discreteDLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, double>{il.begin(), il.end()} }
			};
		}

		template<typename Derived, typename Urng>
		using DiscreteType = CwiseNullaryOp<internal::scalar_rng_adaptor<DiscreteGen<typename Derived::Scalar, int32_t>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteType<Derived, Urng>
			discrete(Index rows, Index cols, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, int32_t>{first, last} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param first, last the range of elements defining the numbers to use as weights. The type of the elements referred by `RealIter` must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename RealIter>
		inline const DiscreteType<Derived, Urng>
			discreteLike(Derived& o, Urng&& urng, RealIter first, RealIter last)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, int32_t>{first, last} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by it must be convertible to `double`.
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteType<Derived, Urng>
			discrete(Index rows, Index cols, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, int32_t>{il.begin(), il.end()} }
			};
		}

		/**
		 * @brief generates random integers on the interval `[0, n)`, where the probability of each individual integer `i` is proportional to `w(i)`.
		 * The data type used for calculation of probabilities is int32(32bit precision).
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param il an instance of `initializer_list` containing the numbers to use as weights. The type of the elements referred by it must be convertible to `double`.
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::DiscreteGen
		 */
		template<typename Derived, typename Urng, typename Real>
		inline const DiscreteType<Derived, Urng>
			discreteLike(Derived& o, Urng&& urng, const std::initializer_list<Real>& il)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), DiscreteGen<typename Derived::Scalar, int32_t>{il.begin(), il.end()} }
			};
		}

		template<typename Derived, typename Urng>
		using PoissonType = CwiseNullaryOp<internal::scalar_rng_adaptor<PoissonGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the Poisson distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param mean rate parameter
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::PoissonGen
		 */
		template<typename Derived, typename Urng>
		inline const PoissonType<Derived, Urng>
			poisson(Index rows, Index cols, Urng&& urng, double mean = 1)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), PoissonGen<typename Derived::Scalar>{mean} }
			};
		}

		/**
		 * @brief generates reals on the Poisson distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param mean rate parameter
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::PoissonGen
		 */
		template<typename Derived, typename Urng>
		inline const PoissonType<Derived, Urng>
			poissonLike(Derived& o, Urng&& urng, double mean = 1)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), PoissonGen<typename Derived::Scalar>{mean} }
			};
		}

		template<typename Derived, typename Urng>
		using BinomialType = CwiseNullaryOp<internal::scalar_rng_adaptor<BinomialGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the binomial distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param trials the number of trials
		 * @param p probability of a trial generating true
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::BinomialGen
		 */
		template<typename Derived, typename Urng>
		inline const BinomialType<Derived, Urng>
			binomial(Index rows, Index cols, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), BinomialGen<typename Derived::Scalar>{trials, p} }
			};
		}

		/**
		 * @brief generates reals on the binomial distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param trials the number of trials
		 * @param p probability of a trial generating true
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::BinomialGen
		 */
		template<typename Derived, typename Urng>
		inline const BinomialType<Derived, Urng>
			binomialLike(Derived& o, Urng&& urng, typename Derived::Scalar trials = 1, double p = 0.5)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), BinomialGen<typename Derived::Scalar>{trials, p} }
			};
		}

		template<typename Derived, typename Urng>
		using GeometricType = CwiseNullaryOp<internal::scalar_rng_adaptor<GeometricGen<typename Derived::Scalar>, typename Derived::Scalar, Urng, true>, const Derived>;

		/**
		 * @brief generates reals on the geometric distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param rows the number of rows being generated
		 * @param cols the number of columns being generated
		 * @param urng c++11-style random number generator
		 * @param p probability of a trial generating true
		 * @return a random matrix expression with a shape (`rows`, `cols`)
		 * 
		 * @see Eigen::Rand::GeometricGen
		 */
		template<typename Derived, typename Urng>
		inline const GeometricType<Derived, Urng>
			geometric(Index rows, Index cols, Urng&& urng, double p = 0.5)
		{
			return {
				rows, cols, { std::forward<Urng>(urng), GeometricGen<typename Derived::Scalar>{p} }
			};
		}

		/**
		 * @brief generates reals on the geometric distribution.
		 *
		 * @tparam Derived
		 * @tparam Urng
		 * @param o an instance of any type of Eigen::DenseBase
		 * @param urng c++11-style random number generator
		 * @param p probability of a trial generating true
		 * @return a random matrix expression of the same shape as `o`
		 * 
		 * @see Eigen::Rand::GeometricGen
		 */
		template<typename Derived, typename Urng>
		inline const GeometricType<Derived, Urng>
			geometricLike(Derived& o, Urng&& urng, double p = 0.5)
		{
			return {
				o.rows(), o.cols(), { std::forward<Urng>(urng), GeometricGen<typename Derived::Scalar>{p} }
			};
		}
	}

#ifdef EIGEN_VECTORIZE_NEON
	namespace internal
	{
		template<typename _Scalar, typename Urng, bool _mutable>
		struct functor_traits<scalar_rng_adaptor<Rand::DiscreteGen<_Scalar, double>, _Scalar, Urng, _mutable> >
		{
			enum { Cost = HugeCost, PacketAccess = 0, IsRepeatable = false };
		};
	}
#endif
}

#endif
