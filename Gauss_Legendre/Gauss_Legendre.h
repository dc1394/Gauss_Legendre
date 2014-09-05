//-----------------------------------------------------------------------
// <copyright file="Function.h" company="dc1394's software">
//     Copyright ©  2014 @dc1394 All Rights Reserved.
// </copyright>
//-----------------------------------------------------------------------
#pragma once

#include "Function.h"
#include <cstdint>
#include <vector>
//#include <intrin.h>
#include <ap.h>
#include <boost/simd/memory/allocator.hpp>
//#include <dvec.h>
#include <boost/simd/sdk/simd/native.hpp>
//#include <boost/simd/include/functions/sum.hpp>
//#include <boost/simd/include/functions/load.hpp>
//#include <boost/simd/include/functions/plus.hpp>
//#include <boost/simd/include/functions/multiplies.hpp>

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
        double qgauss(const myfunctional::Function<FUNCTYPE> & func, bool usesimd, double x1, double x2) const;

    private:
        //! A private member function.
        /*!
        AVX命令が使用可能かどうかをチェックする
        \return AVX命令が使用可能ならtrue、使用不可能ならfalse
        */
        bool availableAVX() const;

        // #endregion メンバ関数

        // #region メンバ変数

        //! A private member variable (const).
        /*!
        AVX命令が使用可能かどうか
        */
        const bool avxSupported;

        //! A private member variable (const).
        /*!
        Gauss-Legendreの分点
        */
        const std::uint32_t n_;

        //! A private member variable.
        /*!
        Gauss-Legendreの重み（alignmentが揃ってない）
        */
        alglib::real_1d_array w_;

        //! A private member variable.
        /*!
        Gauss-Legendreの重み（alignmentが揃っている）
        */
        std::vector<double, boost::simd::allocator<double, 32>> waligned_;

        //! A private member variable.
        /*!
        Gauss-Legendreの節（alignmentが揃ってない）
        */
        alglib::real_1d_array x_;

        //! A private member variable.
        /*!
        Gauss-Legendreの節（alignmentが揃っている）
        */
        std::vector<double, boost::simd::allocator<double, 32>> xaligned_;
        
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
		Gauss_Legendre(const Gauss_Legendre &) = delete;

        //! operator=() (deleted).
        /*!
        デフォルトコンストラクタ（禁止）
        \param コピー元のオブジェクト
        \return コピー元のオブジェクト
        */
		Gauss_Legendre & operator=(const Gauss_Legendre &) = delete;

        // #endregion 禁止されたコンストラクタ・メンバ関数

	};

	inline bool Gauss_Legendre::availableAVX() const
	{
#if (_MSC_FULL_VER >= 160040219)
		int cpuInfo[4];
		__cpuid(cpuInfo, 1);
 			
		const bool osUsesXSAVE_XRSTORE = cpuInfo[2] & (1 << 27) || false;
		const bool cpuAVXSuport = cpuInfo[2] & (1 << 28) || false;
 
		if (osUsesXSAVE_XRSTORE && cpuAVXSuport)
		{
			// Check if the OS will save the YMM registers
			const std::uint64_t xcrFeatureMask = _xgetbv(_XCR_XFEATURE_ENABLED_MASK);
			return (xcrFeatureMask & 0x6) || false;
		}
#endif
		return false;
	}

    template <typename FUNCTYPE>
    double Gauss_Legendre::qgauss(const myfunctional::Function<FUNCTYPE> & func, bool usesimd, double x1, double x2) const
    {
        using boost::simd::pack;

        const double xm = 0.5 * (x1 + x2);
        const double xr = 0.5 * (x2 - x1);

        double sum = 0.0;
        if (usesimd && avxSupported) {
            const std::uint32_t loop = n_ >> 2;
            for (std::uint32_t i = 0; i < loop; i++) {
                //auto tmp = boost::simd::pack<32>(xr);
                const F64vec4 xi(F64vec4(_mm256_load_pd(&x_[(i << 2)]) * F64vec4(xr) + F64vec4(xm)));
                sum += add_horizontal(F64vec4(_mm256_load_pd(&w_[(i << 2)]) *
                    F64vec4(func(xi[3]), func(xi[2]), func(xi[1]), func(xi[0]))));
            }

            const std::uint32_t remainder = n_ & 0x03;
            for (int i = n_ - remainder; i < n_; i++)
                sum += w_[i] * func(xm + xr * x_[i]);
        }
        else if (usesimd) {
            const std::uint32_t loop = n_ >> 1;
            for (std::uint32_t i = 0; i < loop; i++) {
                const F64vec2 xi(F64vec2(_mm_load_pd(&x_[(i << 1)]) * F64vec2(xr) + F64vec2(xm)));
                sum += add_horizontal(F64vec2(_mm_load_pd(&w_[(i << 1)]) * F64vec2(func(xi[1]), func(xi[0]))));
            }

            if (n_ & 0x01)
                sum += w_[n_ - 1] * func(xm + xr * x_[n_ - 1]);
        }
        else {
            for (std::uint32_t i = 0; i < n_; i++) {
                const double xi = xm + xr * x_[i];
                sum += w_[i] * func(xi);
            }
        }
        return sum * xr;
    }

}
