#include "Gauss_Legendre.h"
#include <stdexcept>
#include <integration.h>

namespace gausslegendre {
    Gauss_Legendre::Gauss_Legendre(std::uint32_t n)
        :	n_(n), avxSupported(availableAVX())
    {
        alglib::ae_int_t info = 0;
        alglib::gqgenerategausslegendre(n, info, x_, w_);
        switch (info) {
        case 1:
            break;

        default:
            throw std::runtime_error("alglib::gqgenerategausslegendre‚ª¸”s");
            break;
        }
        
    }
}
