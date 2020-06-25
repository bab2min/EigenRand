/**
* EigenRand
* Author: bab2min@gmail.com
* Date: 2020-06-22
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
					bitmask = ((1 << bitsize) - 1);
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
#ifdef EIGEN_VECTORIZE_AVX
				using PacketType = Packet4f;
#else
				using PacketType = decltype(reinterpret_to_float(std::declval<Packet>()));
#endif
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
	}
}

#endif
