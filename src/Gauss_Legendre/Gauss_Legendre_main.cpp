#include "checkpoint.h"
#include "gauss_Legendre.h"
#include <array>            // for std::array
#include <cmath>            // for std::sqrt
#include <iomanip>
#include <iostream>
#include <string>

namespace {
    static auto constexpr DIGIT = 15U;
    static auto constexpr LOOPMAX = 1000000000UL;
    static auto constexpr N = 10001U;
}

int main()
{
    auto const func = myfunctional::make_functional([](double x) { return 1.0 / (2.0 * std::sqrt(x)); });
	auto const exact = static_cast<double>(LOOPMAX);
    std::array<double, 2> res;

    checkpoint::CheckPoint chk;
    
    chk.checkpoint("処理開始", __LINE__);
	
	gausslegendre::Gauss_Legendre gl(N);
	
    chk.checkpoint("Gauss-Legendreの分点を求める処理", __LINE__);

	{
		auto sum = 0.0;
        for (auto i = 0UL; i < LOOPMAX; i++) {
            sum += gl.qgauss(func, false, 1.0, 4.0);
        }
		
		res[0] = sum;
	}

    chk.checkpoint("AVX無効", __LINE__);
	
	{
		auto sum = 0.0;
        for (auto i = 0UL; i < LOOPMAX; i++) {
            sum += gl.qgauss(func, true, 1.0, 4.0);
        }
		
		res[1] = sum;
	}
	chk.checkpoint("AVX有効", __LINE__);

	chk.checkpoint_print();

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout << "正確な値：\t" << std::setprecision(DIGIT) << exact  << '\n';
	std::cout << "AVX無効：\t" << std::setprecision(DIGIT) << res[0] << '\n';
    std::cout << "AVX有効：\t" << std::setprecision(DIGIT) << res[1] << '\n';

	return 0;
}
