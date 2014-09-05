#pragma warning(disable: 4819)
#include "ChkPoint.h"
#include "Gauss_Legendre.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <array>
#include <cmath>
#include <boost/optional.hpp>
#include <boost/utility/in_place_factory.hpp>
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
#if defined(__INTEL_COMPILER) || defined(__GXX_EXPERIMENTAL_CXX0X__)
	static constexpr std::uint32_t N = 49999;
	static constexpr std::int32_t LOOPMAX = 100000;
	static constexpr std::uint32_t DIGIT = 10;
	static constexpr double a = 2.0 / std::sqrt(boost::math::constants::pi<double>());
	static constexpr double z = 1.0;

#ifdef _OPENMP
	static constexpr std::string str("OpenMP");
#else
	static constexpr std::string str("Intel Cilk Plus");
#endif

#else
	static const std::uint32_t N = 49999;
	static const std::int32_t LOOPMAX = 100000;
	static const std::uint32_t DIGIT = 10;
	static const double a = 2.0 / std::sqrt(boost::math::constants::pi<double>());
	static const double z = 1.0;

#ifdef _OPENMP
	static const std::string str("OpenMP");
#else
	static const std::string str("Intel Cilk Plus");
#endif

#endif
}

int main()
{
	const auto func([](double x) { return a * exp(- x * x); });
	const double exact = boost::math::erf(z) * static_cast<double>(LOOPMAX);
	std::array<double, 4> res;
	std::array<std::string, 4> out;

	CheckPoint::ChkPoint chk("�����J�n", __LINE__);
	
	boost::optional<Gauss_Legendre::Gauss_Legendre> pgl(N);
	pgl = boost::none;
	const std::string tmp("Gauss-Legendre�̕��_�v�Z�A" + str + "����");
	chk.checkpoint(tmp.c_str(), __LINE__);
	
	pgl = boost::in_place(N, true);
	const std::string tmp2("Gauss-Legendre�̕��_�v�Z�A" + str + "�L��");
	chk.checkpoint(tmp2.c_str(), __LINE__);
	
	{
		double sum = 0.0;
		for (std::int32_t i = 0; i < LOOPMAX; i++)
			sum += pgl->qgauss(0.0, z, func, false);
		
		res[0] = sum;
	}
	out[0] = "AVX�����A" + str + "����";
	chk.checkpoint(out[0].c_str(), __LINE__);
	
	{
		double sum = 0.0;
		for (std::int32_t i = 0; i < LOOPMAX; i++)
			sum += pgl->qgauss(0.0, z, func, true);
		
		res[1] = sum;
	}
	out[1] = "AVX�L���A" + str + "����";
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
			sum += pgl->qgauss(0.0, z, func, false);

#ifdef _OPENMP
		res[2] = sum;
#else
		res[2] = sum.get_value();
#endif
	}
	out[2] = "AVX�����A" + str + "�L��";
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
			sum += pgl->qgauss(0.0, z, func, true);

#ifdef _OPENMP
		res[3] = sum;
#else
		res[3] = sum.get_value();
#endif
	}
	out[3] = "AVX�L���A" + str + "�L��";
	chk.checkpoint(out[3].c_str(), __LINE__);
	
	chk.checkpoint_print();

	pgl = boost::none;

	std::cout.setf(std::ios::fixed, std::ios::floatfield);
	std::cout << "���m�Ȓl�F\t\t\t" << std::setprecision(DIGIT) << exact  << '\n';
	std::cout << out[0] << "�F\t" << std::setprecision(DIGIT) << res[0] << '\n';
	std::cout << out[1] << "�F\t" << std::setprecision(DIGIT) << res[1] << '\n';
	std::cout << out[2] << "�F\t" << std::setprecision(DIGIT) << res[2] << '\n';
	std::cout << out[3] << "�F\t" << std::setprecision(DIGIT) << res[3] << std::endl;

	return EXIT_SUCCESS;
}
