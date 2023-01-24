// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2014 Gael Guennebaud <gael.guennebaud@inria.fr>
// Copyright (C) 2006-2008 Benoit Jacob <jacob.benoit.1@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CWISE_HETERO_BINARY_OP_H
#define EIGEN_CWISE_HETERO_BINARY_OP_H

namespace Eigen {
    
    template<typename BinaryOp, typename LhsType, typename RhsType>
    class CwiseHeteroBinaryOp;

    namespace internal {
        template<typename BinaryOp, typename Lhs, typename Rhs>
        struct traits<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> >
        {
            // we must not inherit from traits<Lhs> since it has
            // the potential to cause problems with MSVC
            typedef typename remove_all<Lhs>::type Ancestor;
            typedef typename traits<Ancestor>::XprKind XprKind;
            enum {
                RowsAtCompileTime = traits<Ancestor>::RowsAtCompileTime,
                ColsAtCompileTime = traits<Ancestor>::ColsAtCompileTime,
                MaxRowsAtCompileTime = traits<Ancestor>::MaxRowsAtCompileTime,
                MaxColsAtCompileTime = traits<Ancestor>::MaxColsAtCompileTime
            };

            // even though we require Lhs and Rhs to have the same scalar type (see CwiseHeteroBinaryOp constructor),
            // we still want to handle the case when the result type is different.
            typedef typename result_of<
                BinaryOp(
                    const typename Lhs::Scalar&,
                    const typename Rhs::Scalar&
                )
            >::type Scalar;
            typedef typename cwise_promote_storage_type<typename traits<Lhs>::StorageKind,
                typename traits<Rhs>::StorageKind,
                BinaryOp>::ret StorageKind;
            typedef typename promote_index_type<typename traits<Lhs>::StorageIndex,
                typename traits<Rhs>::StorageIndex>::type StorageIndex;
            typedef typename Lhs::Nested LhsNested;
            typedef typename Rhs::Nested RhsNested;
            typedef typename remove_reference<LhsNested>::type _LhsNested;
            typedef typename remove_reference<RhsNested>::type _RhsNested;
            enum {
                Flags = cwise_promote_storage_order<typename traits<Lhs>::StorageKind, typename traits<Rhs>::StorageKind, _LhsNested::Flags& RowMajorBit, _RhsNested::Flags& RowMajorBit>::value
            };
        };
    } // end namespace internal

    template<typename BinaryOp, typename Lhs, typename Rhs, typename StorageKind>
    class CwiseHeteroBinaryOpImpl;

    /** \class CwiseHeteroBinaryOp
      * \ingroup Core_Module
      *
      * \brief Generic expression where a coefficient-wise binary operator is applied to two expressions
      *
      * \tparam BinaryOp template functor implementing the operator
      * \tparam LhsType the type of the left-hand side
      * \tparam RhsType the type of the right-hand side
      *
      * This class represents an expression  where a coefficient-wise binary operator is applied to two expressions.
      * It is the return type of binary operators, by which we mean only those binary operators where
      * both the left-hand side and the right-hand side are Eigen expressions.
      * For example, the return type of matrix1+matrix2 is a CwiseHeteroBinaryOp.
      *
      * Most of the time, this is the only way that it is used, so you typically don't have to name
      * CwiseHeteroBinaryOp types explicitly.
      *
      * \sa MatrixBase::binaryExpr(const MatrixBase<OtherDerived> &,const CustomBinaryOp &) const, class CwiseUnaryOp, class CwiseNullaryOp
      */
    template<typename BinaryOp, typename LhsType, typename RhsType>
    class CwiseHeteroBinaryOp :
        public CwiseHeteroBinaryOpImpl<
        BinaryOp, LhsType, RhsType,
        typename internal::cwise_promote_storage_type<typename internal::traits<LhsType>::StorageKind,
        typename internal::traits<RhsType>::StorageKind,
        BinaryOp>::ret>,
        internal::no_assignment_operator
    {
    public:

        typedef typename internal::remove_all<BinaryOp>::type Functor;
        typedef typename internal::remove_all<LhsType>::type Lhs;
        typedef typename internal::remove_all<RhsType>::type Rhs;

        typedef typename CwiseHeteroBinaryOpImpl<
            BinaryOp, LhsType, RhsType,
            typename internal::cwise_promote_storage_type<typename internal::traits<LhsType>::StorageKind,
            typename internal::traits<Rhs>::StorageKind,
            BinaryOp>::ret>::Base Base;
        EIGEN_GENERIC_PUBLIC_INTERFACE(CwiseHeteroBinaryOp)

            typedef typename internal::ref_selector<LhsType>::type LhsNested;
        typedef typename internal::ref_selector<RhsType>::type RhsNested;
        typedef typename internal::remove_reference<LhsNested>::type _LhsNested;
        typedef typename internal::remove_reference<RhsNested>::type _RhsNested;

#if EIGEN_COMP_MSVC && EIGEN_HAS_CXX11
        //Required for Visual Studio or the Copy constructor will probably not get inlined!
        EIGEN_STRONG_INLINE
            CwiseHeteroBinaryOp(const CwiseHeteroBinaryOp<BinaryOp, LhsType, RhsType>&) = default;
#endif

        EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            CwiseHeteroBinaryOp(const Lhs& aLhs, const Rhs& aRhs, const BinaryOp& func = BinaryOp())
            : m_lhs(aLhs), m_rhs(aRhs), m_functor(func)
        {
            // require the sizes to match
            EIGEN_STATIC_ASSERT_SAME_MATRIX_SIZE(Lhs, Rhs)
                eigen_assert(aLhs.rows() == aRhs.rows() && aLhs.cols() == aRhs.cols());
        }

        EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr
            Index rows() const EIGEN_NOEXCEPT {
            // return the fixed size type if available to enable compile time optimizations
            return internal::traits<typename internal::remove_all<LhsNested>::type>::RowsAtCompileTime == Dynamic ? m_rhs.rows() : m_lhs.rows();
        }
        EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE constexpr
            Index cols() const EIGEN_NOEXCEPT {
            // return the fixed size type if available to enable compile time optimizations
            return internal::traits<typename internal::remove_all<LhsNested>::type>::ColsAtCompileTime == Dynamic ? m_rhs.cols() : m_lhs.cols();
        }

        /** \returns the left hand side nested expression */
        EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            const _LhsNested& lhs() const { return m_lhs; }
        /** \returns the right hand side nested expression */
        EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            const _RhsNested& rhs() const { return m_rhs; }
        /** \returns the functor representing the binary operation */
        EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
            const BinaryOp& functor() const { return m_functor; }

    protected:
        LhsNested m_lhs;
        RhsNested m_rhs;
        const BinaryOp m_functor;
    };

    // Generic API dispatcher
    template<typename BinaryOp, typename Lhs, typename Rhs, typename StorageKind>
    class CwiseHeteroBinaryOpImpl
        : public internal::generic_xpr_base<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> >::type
    {
    public:
        typedef typename internal::generic_xpr_base<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> >::type Base;
    };


    namespace internal {
        // -------------------- CwiseHeteroBinaryOp --------------------

        // this is a binary expression
        template<typename BinaryOp, typename Lhs, typename Rhs>
        struct evaluator<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> >
            : public binary_evaluator<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> >
        {
            typedef CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> XprType;
            typedef binary_evaluator<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> > Base;

            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
                explicit evaluator(const XprType& xpr) : Base(xpr) {}
        };

        template<typename BinaryOp, typename Lhs, typename Rhs>
        struct binary_evaluator<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs>, IndexBased, IndexBased>
            : evaluator_base<CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> >
        {
            typedef CwiseHeteroBinaryOp<BinaryOp, Lhs, Rhs> XprType;
            using LhsScalar = typename Lhs::Scalar;
            using RhsScalar = typename Rhs::Scalar;

            enum {
                CoeffReadCost = int(evaluator<Lhs>::CoeffReadCost) + int(evaluator<Rhs>::CoeffReadCost) + int(functor_traits<BinaryOp>::Cost),

                LhsFlags = evaluator<Lhs>::Flags,
                RhsFlags = evaluator<Rhs>::Flags,
                SameType = is_same<typename Lhs::Scalar, typename Rhs::Scalar>::value,
                StorageOrdersAgree = (int(LhsFlags) & RowMajorBit) == (int(RhsFlags) & RowMajorBit),
                Flags0 = (int(LhsFlags) | int(RhsFlags)) & (
                    HereditaryBits
                    | (int(LhsFlags) & int(RhsFlags) &
                        ((StorageOrdersAgree ? LinearAccessBit : 0)
                            | (functor_traits<BinaryOp>::PacketAccess && StorageOrdersAgree ? PacketAccessBit : 0)
                            )
                        )
                    ),
                Flags = (Flags0 & ~RowMajorBit) | (LhsFlags & RowMajorBit),
                Alignment = EIGEN_PLAIN_ENUM_MIN(evaluator<Lhs>::Alignment, evaluator<Rhs>::Alignment)
            };

            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
                explicit binary_evaluator(const XprType& xpr) : m_d(xpr)
            {
                EIGEN_INTERNAL_CHECK_COST_VALUE(functor_traits<BinaryOp>::Cost);
                EIGEN_INTERNAL_CHECK_COST_VALUE(CoeffReadCost);
            }

            typedef typename XprType::CoeffReturnType CoeffReturnType;

            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
                CoeffReturnType coeff(Index row, Index col) const
            {
                return m_d.func()(m_d.lhsImpl.coeff(row, col), m_d.rhsImpl.coeff(row, col));
            }

            EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
                CoeffReturnType coeff(Index index) const
            {
                return m_d.func()(m_d.lhsImpl.coeff(index), m_d.rhsImpl.coeff(index));
            }

            template<int LoadMode, typename PacketType>
            EIGEN_STRONG_INLINE
                PacketType packet(Index row, Index col) const
            {
                using IPacketType = decltype(reinterpret_to_float(std::declval<PacketType>()));
                using FPacketType = decltype(reinterpret_to_float(std::declval<PacketType>()));
                using RhsPacket = typename std::conditional<std::is_same<RhsScalar, float>::value, FPacketType, IPacketType>::type;

                return m_d.func().packetOp(m_d.lhsImpl.template packet<LoadMode, PacketType>(row, col),
                    m_d.rhsImpl.template packet<LoadMode, RhsPacket>(row, col));
            }

            template<int LoadMode, typename PacketType>
            EIGEN_STRONG_INLINE
                PacketType packet(Index index) const
            {
                using IPacketType = decltype(reinterpret_to_float(std::declval<PacketType>()));
                using FPacketType = decltype(reinterpret_to_float(std::declval<PacketType>()));
                using RhsPacket = typename std::conditional<std::is_same<RhsScalar, float>::value, FPacketType, IPacketType>::type;

                return m_d.func().packetOp(m_d.lhsImpl.template packet<LoadMode, PacketType>(index),
                    m_d.rhsImpl.template packet<LoadMode, RhsPacket>(index));
            }

        protected:

            // this helper permits to completely eliminate the functor if it is empty
            struct Data
            {
                EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
                    Data(const XprType& xpr) : op(xpr.functor()), lhsImpl(xpr.lhs()), rhsImpl(xpr.rhs()) {}
                EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
                    const BinaryOp& func() const { return op; }
                BinaryOp op;
                evaluator<Lhs> lhsImpl;
                evaluator<Rhs> rhsImpl;
            };

            Data m_d;
        };
    }

} // end namespace Eigen

#endif // EIGEN_CWISE_HETERO_BINARY_OP_H
