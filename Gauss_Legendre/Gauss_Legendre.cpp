#include "Gauss_Legendre.h"
#include <stdexcept>
#include <integration.h>

namespace gausslegendre {
    Gauss_Legendre::Gauss_Legendre(std::uint32_t n)
        :	n_(n), avxSupported(availableAVX())
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
