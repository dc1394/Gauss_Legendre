#pragma warning(disable: 4819)
#include "Gauss_Legendre.h"
#include <boost/cast.hpp>
#include <boost/static_assert.hpp>
#include <boost/math/constants/constants.hpp>
#include <dvec.h>

#if defined(_OPENMP)
	#include <omp.h>
	#define cilk_for for
#elif defined(__cilk)
	#include <cilk/cilk.h>
	#include <cilk/reducer_opadd.h>
#else
	BOOST_STATIC_ASSERT(false)
#endif

namespace Gauss_Legendre {
#if !defined(__INTEL_COMPILER) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
	const double Gauss_Legendre::EPS = 1.0E-15;
#endif

	void Gauss_Legendre::gauleg(bool usecilkoropenmp)
	{
		const std::int32_t m = boost::numeric_cast<std::int32_t>((n_ + 1) >> 1);
		if (usecilkoropenmp) {
#ifdef _OPENMP
			#pragma omp parallel for schedule(guided)
#endif
			cilk_for (std::int32_t i = 1; i <= m; i++) {
				double z = std::cos(boost::math::constants::pi<double>() *
						   (static_cast<double>(i) - 0.25) /
						   (static_cast<double>(n_) + 0.5));

				double z1, pp;

				do {
					double p1 = 1.0;
					double p2 = 0.0;

					for (std::uint32_t j = 1; j <= n_; j++) {
						const double p3 = p2;
						p2 = p1;
						p1 = ((2.0 * static_cast<double>(j) - 1.0) * z * p2 -
							 (static_cast<double>(j) - 1.0) * p3) /
							 static_cast<double>(j);
					}

					pp = static_cast<double>(n_) * (z * p1 - p2) /
						 (z * z - 1.0);
					z1 = z;
					z = z1 - p1 / pp;
				} while (std::fabs(z - z1) > EPS);

				x_[i - 1] = - z;
				x_[n_ - i] = z;
				w_[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp);
				w_[n_ - i] = w_[i - 1];
			}
		} else {
			for (std::int32_t i = 1; i <= m; i++) {
				double z = std::cos(boost::math::constants::pi<double>() *
						   (static_cast<double>(i) - 0.25) /
						   (static_cast<double>(n_) + 0.5));

				double z1, pp;

				do {
					double p1 = 1.0;
					double p2 = 0.0;

					for (std::uint32_t j = 1; j <= n_; j++) {
						const double p3 = p2;
						p2 = p1;
						p1 = ((2.0 * static_cast<double>(j) - 1.0) * z * p2 -
							 (static_cast<double>(j) - 1.0) * p3) /
							 static_cast<double>(j);
					}

					pp = static_cast<double>(n_) * (z * p1 - p2) /
						 (z * z - 1.0);
					z1 = z;
					z = z1 - p1 / pp;
				} while (std::fabs(z - z1) > EPS);

				x_[i - 1] = - z;
				x_[n_ - i] = z;
				w_[i - 1] = 2.0 / ((1.0 - z * z) * pp * pp);
				w_[n_ - i] = w_[i - 1];
			}
		}
	}

	double Gauss_Legendre::qgauss(double x1, double x2, const std::function<double (double)> & func, bool useSSEorAVX) const
	{
		const double xm = 0.5 * (x1 + x2);
		const double xr = 0.5 * (x2 - x1);

		double sum = 0.0;
		if (useSSEorAVX && avxSupported) {
			const std::uint32_t loop = n_ >> 2;
			for (std::uint32_t i = 0; i < loop; i++) {
				const F64vec4 xi(F64vec4(_mm256_load_pd(&x_[(i << 2)]) * F64vec4(xr) + F64vec4(xm)));
				sum += add_horizontal(F64vec4(_mm256_load_pd(&w_[(i << 2)]) *
									  F64vec4(func(xi[3]), func(xi[2]), func(xi[1]), func(xi[0]))));
			}

			const std::uint32_t remainder = n_ & 0x03;
			for (int i = n_ - remainder; i < n_; i++)
				sum += w_[i] * func(xm + xr * x_[i]);
		} else if (useSSEorAVX) {
			const std::uint32_t loop = n_ >> 1;
			for (std::uint32_t i = 0; i < loop; i++) {
				const F64vec2 xi(F64vec2(_mm_load_pd(&x_[(i << 1)]) * F64vec2(xr) + F64vec2(xm)));
				sum += add_horizontal(F64vec2(_mm_load_pd(&w_[(i << 1)]) * F64vec2(func(xi[1]), func(xi[0]))));
			}

			if (n_ & 0x01)
				sum += w_[n_ - 1] * func(xm + xr * x_[n_ - 1]);
		} else {
			for (std::uint32_t i = 0; i < n_; i++) {
				const double xi = xm + xr * x_[i];
				sum += w_[i] * func(xi);
			}
		}
		return sum * xr;
	}
}
