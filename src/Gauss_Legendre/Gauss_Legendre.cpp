﻿/*! \file gauss_legendre.cpp
    \brief Gauss-Legendre積分を行うクラスの実装

    Copyright ©  2014 @dc1394 All Rights Reserved.
*/
#include "gauss_legendre.h"
#include <stdexcept>
#include <integration.h>

namespace gausslegendre {
    Gauss_Legendre::Gauss_Legendre(std::uint32_t n)
        : avxSupported(availableAVX()), n_(n)
    {
        alglib::ae_int_t info = 0;
        alglib::real_1d_array x, w;
        alglib::gqgenerategausslegendre(n, info, x, w);
        switch (info) {
        case 1:
            break;

        default:
            throw std::runtime_error("alglib::gqgenerategausslegendreが失敗");
            break;
        }

        x_.assign(x.getcontent(), x.getcontent() + x.length());
        w_.assign(w.getcontent(), w.getcontent() + w.length());
    }
}
