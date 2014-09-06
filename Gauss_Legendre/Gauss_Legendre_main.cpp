#include "CheckPoint.h"
#include "Gauss_Legendre.h"
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <string>
#include <boost/math/constants/constants.hpp>
#include <boost/math/special_functions/erf.hpp>

#if defined(_OPENMP)
	#include <omp.h>
	#define cilk_for for
#elif defined(__cilk)
	#include <cilk/cilk.h>
	#include <cilk/reducer_opadd.h>
#else
	BOOST_STATIC_ASSERT(false)
#endif

namespace {
		static auto constexpr N = 10000;
		static auto constexpr LOOPMAX = 500;
		static auto constexpr DIGIT = 10;
		static auto constexpr z = 1.0;
		static auto const a = 2.0 / std::sqrt(boost::math::constants::pi<double>());

#ifdef _OPENMP
		static std::string const str("OpenMP");
#else
		static std::string const str("Intel Cilk Plus");
#endif
	}

int main()
{
    auto const func = myfunctional::make_functional([](double x) { return a * exp(-x * x); });
	auto const exact = boost::math::erf(z) * static_cast<double>(LOOPMAX);
	std::array<double, 4> res;
	std::array<std::string, 4> out;

    checkpoint::CheckPoint chk;
    
    chk.checkpoint("処理開始", __LINE__);
	
	gausslegendre::Gauss_Legendre gl(N);
	
    chk.checkpoint("Gauss-Legendreの分点を求める処理", __LINE__);

	{
		double sum = 0.0;
		for (std::int32_t i = 0; i < LOOPMAX; i++)
			sum += gl.qgauss(func, false, 0.0, z);
		
		res[0] = sum;
	}
	out[0] = "AVX無効、" + str + "無効";
	chk.checkpoint(out[0].c_str(), __LINE__);
	
	{
		double sum = 0.0;
		for (std::int32_t i = 0; i < LOOPMAX; i++)
			sum += gl.qgauss(func, true, 0.0, z);
		
		res[1] = sum;
	}
	out[1] = "AVX有効、" + str + "無効";
	chk.checkpoint(out[1].c_str(), __LINE__);

	{
#ifdef _OPENMP
		double sum = 0.0;
		#pragma omp parallel for reduction(+:sum) schedule(guided)
#else
		cilk::reducer_opadd<double> sum;
		sum.set_value(0.0);
#endif
		cilk_for (std::int32_t i = 0; i< LOOPMAX; i++)
			sum += gl.qgauss(func, false, 0.0, z);

#ifdef _OPENMP
		res[2] = sum;
#else
		res[2] = sum.get_value();
#endif
	}
	out[2] = "AVX無効、" + str + "有効";
	chk.checkpoint(out[2].c_str(), __LINE__);

	{
#ifdef _OPENMP
		double sum = 0.0;
		#pragma omp parallel for reduction(+:sum) schedule(guided)
#else
		cilk::reducer_opadd<double> sum;
		sum.set_value(0.0);
#endif
		cilk_for (std::int32_t i = 0; i < LOOPMAX; i++)
			sum += gl.qgauss(func, true, 0.0, z);

#ifdef _OPENMP
		res[3] = sum;
#else
		res[3] = sum.get_value();
#endif
	}
	out[3] = "AVX有効、" + str + "有効";
	chk.checkpoint(out[3].c_str(), __LINE__);
	
	chk.checkpoint_print();

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout << "正確な値：\t\t\t" << std::setprecision(DIGIT) << exact  << '\n';
	std::cout << out[0] << "：\t" << std::setprecision(DIGIT) << res[0] << '\n';
    std::cout << out[1] << "：\t" << std::setprecision(DIGIT) << res[1] << '\n';
	std::cout << out[2] << "：\t" << std::setprecision(DIGIT) << res[2] << '\n';
	std::cout << out[3] << "：\t" << std::setprecision(DIGIT) << res[3] << std::endl;

	return EXIT_SUCCESS;
}
