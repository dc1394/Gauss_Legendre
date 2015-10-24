/*! \file Gauss_Legendre.h
    \brief Gauss-Legendre積分を行うクラスの宣言

    Copyright ©  2014 @dc1394 All Rights Reserved.
*/
#ifndef _GAUSS_LEGENDRE_H_
#define _GAUSS_LEGENDRE_H_

#pragma once

#include "Functional.h"
#include <array>                            // for std::array
#include <cstdint>                          // for std::int32_t
#include <vector>                           // for std::vector
#include <intrin.h>                         // for _xgetbv
#include <boost/simd/memory/allocator.hpp>  // for boost::simd::allocator
#include <dvec.h>                           // for F64vec4, F64vec2

namespace gausslegendre {
    //! A class.
    /*!
        Gauss-Legendre積分を行うクラス
    */
	class Gauss_Legendre final
	{
    public:
        // #region コンストラクタ

        //! A constructor.
        /*!
            唯一のコンストラクタ
            Gauss-Legendreの重みと節を計算して、それぞれw_とx_に格納する
            \param n Gauss-Legendreの分点
        */
        explicit Gauss_Legendre(std::uint32_t n);

        // #endregion コンストラクタ

        // #region メンバ関数

        //! A public member function (template function).
        /*!
            Gauss-Legendre積分を実行する
            \param func 被積分関数
            \param usesimd SIMDを使用するかどうか
            \param x1 積分の下端
            \param x2 積分の上端
            \return 積分値
        */
        template <typename FUNCTYPE>
        double qgauss(myfunctional::Functional<FUNCTYPE> const & func, bool usesimd, double x1, double x2) const;

    private:
        //! A private member function.
        /*!
            AVX命令が使用可能かどうかをチェックする
            \return AVX命令が使用可能ならtrue、使用不可能ならfalse
        */
        bool availableAVX() const;

        // #endregion メンバ関数

        // #region メンバ変数

        //! A private member variable (constant).
        /*!
            AVX命令が使用可能かどうか
        */
        const bool avxSupported;

        //! A private member variable (constant).
        /*!
            Gauss-Legendreの分点
        */
        const std::uint32_t n_;

        //! A private member variable.
        /*!
        Gauss-Legendreの重み（alignmentが揃っている）
        */
        std::vector<double, boost::simd::allocator<double, 64>> w_;

        //! A private member variable.
        /*!
            Gauss-Legendreの節（alignmentが揃っている）
        */
        std::vector<double, boost::simd::allocator<double, 64>> x_;
        
        // #endregion メンバ変数

        // #region 禁止されたコンストラクタ・メンバ関数

        //! A private constructor (deleted).
        /*!
            デフォルトコンストラクタ（禁止）
        */
        Gauss_Legendre() = delete;

        //! A private copy constructor (deleted).
        /*!
            コピーコンストラクタ（禁止）
        */
		Gauss_Legendre(Gauss_Legendre const &) = delete;

        //! A private member function (deleted).
        /*!
            operator=()の宣言（禁止）
            \param コピー元のオブジェクト
            \return コピー元のオブジェクト
        */
		Gauss_Legendre & operator=(Gauss_Legendre const &) = delete;

        // #endregion 禁止されたコンストラクタ・メンバ関数
	};

	inline bool Gauss_Legendre::availableAVX() const
	{
#if (_MSC_FULL_VER >= 160040219)
        std::array<std::int32_t, 4> cpuInfo = { 0 };
		::__cpuid(cpuInfo.data(), 1);
 			
		auto const osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
		auto const cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;
 
		if (osUsesXSAVE_XRSTORE && cpuAVXSuport)
		{
			// Check if the OS will save the YMM registers
			auto const xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
			return (xcrFeatureMask & 0x6) || false;
		}
#endif
		return false;
	}

    template <typename FUNCTYPE>
    inline double Gauss_Legendre::qgauss(myfunctional::Functional<FUNCTYPE> const & func, bool usesimd, double x1, double x2) const
    {
        auto const xm = 0.5 * (x1 + x2);
        auto const xr = 0.5 * (x2 - x1);

        auto sum = 0.0;
        if (usesimd && avxSupported) {
            auto const loop = n_ >> 2;
            for (auto i = 0U; i < loop; i++) {
                auto const xi(
                	F64vec4(
                		_mm256_load_pd(&x_[(i << 2)]) * F64vec4(xr) + F64vec4(xm)));

                auto const t = F64vec4(func(xi[3]), func(xi[2]), func(xi[1]), func(xi[0]));
                
                sum += add_horizontal(F64vec4(_mm256_load_pd(&w_[(i << 2)]) * t));
            }

            auto const remainder = n_ & 0x03;
            for (auto i = n_ - remainder; i < n_; i++)
                sum += w_[i] * func(xm + xr * x_[i]);
        }
        else if (usesimd) {
            auto const loop = n_ >> 1;
            for (auto i = 0U; i < loop; i++) {
                auto const xi(
                	F64vec2(
                		_mm_load_pd(&x_[(i << 1)]) * F64vec2(xr) + F64vec2(xm)));
                sum += add_horizontal(
                	F64vec2(
                		_mm_load_pd(&w_[(i << 1)]) * F64vec2(func(xi[1]), 
                		func(xi[0]))));
            }

            if (n_ & 0x01)
                sum += w_[n_ - 1] * func(xm + xr * x_[n_ - 1]);
        }
        else {
            for (auto i = 0U; i < n_; i++) {
                auto const xi = xm + xr * x_[i];
                sum += w_[i] * func(xi);
            }
        }

        return sum * xr;
    }
}

#endif  // _GAUSS_LEGENDRE_H_
