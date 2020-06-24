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
			size_t msize = 0, bitsize = 0;

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
				nbsize = /*log2_ceil(msize)*/0;
				psize = (size_t)1 << nbsize;

				if (nbsize != bitsize)
				{
					arr = std::unique_ptr<_Precision[]>(new _Precision[psize]);
					std::fill(arr.get(), arr.get() + psize, 0);
					alias = std::unique_ptr<_Size[]>(new _Size[psize]);
					bitsize = nbsize;
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
					arr[under] = f[under] * (std::numeric_limits<_Precision>::max() + 1.0);
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
						arr[over] = std::numeric_limits<_Precision>::max();
						alias[over] = over;
					}
				}

				if (under < psize)
				{
					arr[under] = std::numeric_limits<_Precision>::max();
					alias[under] = under;
					for (under = mm; under < msize; ++under)
					{
						if (f[under] < 1)
						{
							arr[under] = std::numeric_limits<_Precision>::max();
							alias[under] = under;
						}
					}
				}
			}

			size_t get_bitsize() const
			{
				return bitsize;
			}

			const _Precision* get_prob() const
			{
				return arr.get();
			}

			const _Size* get_alias() const
			{
				return alias.get();
			}

			template<typename _Rng>
			size_t operator()(_Rng& rng) const
			{
				auto x = rng();
				size_t a;
				if (sizeof(_Precision) < sizeof(typename _Rng::result_type))
				{
					a = x >> (sizeof(x) * 8 - bitsize);
				}
				else
				{
					a = rng() & ((1 << bitsize) - 1);
				}

				_Precision b = (_Precision)x;
				if (b < arr[a])
				{
					assert(a < msize);
					return a;
				}
				assert(alias[a] < msize);
				return alias[a];
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

			template<typename RealIter>
			scalar_discrete_dist_op(const Rng& _rng, RealIter first, RealIter last)
				: ur_base{ _rng }
			{
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

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				auto rx = ur_base::operator()() & 0x7FFFFFFF;
				return (Scalar)(std::lower_bound(cdf.begin(), cdf.end(), rx) - cdf.begin());
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				auto ret = pset1<Packet>(cdf.size() - 1);
				auto rx = pand(ur_base::template packetOp<Packet>(), pset1<Packet>(0x7FFFFFFF));
				for (size_t i = 0; i < cdf.size() - 1; ++i)
				{
					ret = padd(ret, pcmplt(rx, pset1<Packet>(cdf[i])));
				}
				return ret;
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_discrete_dist_op<Scalar, Rng, float> : public scalar_uniform_real_op<float, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "discreteDist needs integral types.");
			using ur_base = scalar_uniform_real_op<float, Rng>;

			std::vector<float> cdf;

			template<typename RealIter>
			scalar_discrete_dist_op(const Rng& _rng, RealIter first, RealIter last)
				: ur_base{ _rng }
			{
				float acc = 0;
				for (; first != last; ++first)
				{
					cdf.emplace_back(acc += *first);
				}

				for (auto& p : cdf)
				{
					p /= cdf.back();
				}
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				auto rx = ur_base::operator()();
				return (Scalar)(std::lower_bound(cdf.begin(), cdf.end(), rx) - cdf.begin());
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				auto ret = pset1<Packet>(cdf.size());
#ifdef EIGEN_VECTORIZE_AVX
				auto rx = _mm256_castps256_ps128(ur_base::template packetOp<Packet8f>());
				for (auto& p : cdf)
				{
					ret = padd(ret, reinterpret_to_int(pcmplt(rx, pset1<decltype(rx)>(p))));
				}
				return ret;
#else
				auto rx = ur_base::template packetOp<decltype(reinterpret_to_float(std::declval<Packet>()))>();
				for (auto& p : cdf)
				{
					ret = padd(ret, reinterpret_to_int(pcmplt(rx, pset1<decltype(rx)>(p))));
				}
				return ret;
#endif
			}
		};

		template<typename Scalar, typename Rng>
		struct scalar_discrete_dist_op<Scalar, Rng, double> : public scalar_uniform_real_op<double, Rng>
		{
			static_assert(std::is_same<Scalar, int32_t>::value, "discreteDist needs integral types.");
			using ur_base = scalar_uniform_real_op<double, Rng>;

			std::vector<double> cdf;

			template<typename RealIter>
			scalar_discrete_dist_op(const Rng& _rng, RealIter first, RealIter last)
				: ur_base{ _rng }
			{
				double acc = 0;
				for (; first != last; ++first)
				{
					cdf.emplace_back(acc += *first);
				}

				for (auto& p : cdf)
				{
					p /= cdf.back();
				}
			}

			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Scalar operator() () const
			{
				auto rx = ur_base::operator()();
				return (Scalar)(std::lower_bound(cdf.begin(), cdf.end(), rx) - cdf.begin());
			}

			template<typename Packet>
			EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Packet packetOp() const
			{
				auto ret = pset1<Packet>(cdf.size());
#ifdef EIGEN_VECTORIZE_AVX
				auto rx = ur_base::template packetOp<Packet4d>();
				for (auto& p : cdf)
				{
					auto c = _mm256_castpd_si256(pcmplt(rx, pset1<decltype(rx)>(p)));
#ifdef EIGEN_VECTORIZE_AVX2
					auto r = _mm256_castsi256_si128(_mm256_permutevar8x32_epi32(c, _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7)));
#else
					auto sc = _mm256_permutevar_ps(_mm256_castsi256_ps(c), _mm256_setr_epi32(0, 2, 1, 3, 1, 3, 0, 2));
					auto r = _mm_castps_si128(_mm_blend_ps(_mm256_extractf128_ps(sc, 0), _mm256_extractf128_ps(sc, 1), 0b1100));
#endif
					ret = padd(ret, r);
				}
				return ret;
#else
				auto rx1 = ur_base::template packetOp<decltype(reinterpret_to_double(std::declval<Packet>()))>(),
					rx2 = ur_base::template packetOp<decltype(reinterpret_to_double(std::declval<Packet>()))>();
				for (auto& p : cdf)
				{
					auto pp = pset1<decltype(rx1)>(p);
					ret = padd(ret, combine_low32(reinterpret_to_int(pcmplt(rx1, pp)), reinterpret_to_int(pcmplt(rx2, pp))));
				}
				return ret;
#endif
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
