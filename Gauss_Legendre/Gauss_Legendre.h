#include "aligned_malloc_allocator.h"
#include <vector>
#include <functional>
#include <cstdint>
#include <intrin.h>

#if !defined(__INTEL_COMPILER) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
	#include <boost/noncopyable.hpp>
#endif

namespace Gauss_Legendre {
	class Gauss_Legendre
#if !defined(__INTEL_COMPILER) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
		: private boost::noncopyable
#endif
	{
#if defined(__INTEL_COMPILER) || defined(__GXX_EXPERIMENTAL_CXX0X__)
		Gauss_Legendre(const Gauss_Legendre &) = delete;
		Gauss_Legendre & operator=(const Gauss_Legendre &) = delete;
		Gauss_Legendre() = delete;
		static constexpr double EPS = 1.0E-15;
#else
		static const double EPS;
#endif
		typedef std::vector<double, aligned_malloc_allocator<double>> dvector;

		const std::uint32_t n_;
		const bool avxSupported;
		dvector x_;
		dvector w_;
		bool availableAVX() const;
		void gauleg(bool usecilkoropenmp);

	public:
		explicit Gauss_Legendre(std::uint32_t n, bool usecilkoropenmp = false);
		double qgauss(double x1, double x2, const std::function<double (double)> & func, bool useSSEorAVX) const;
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

	inline Gauss_Legendre::Gauss_Legendre(std::uint32_t n, bool usecilkoropenmp)
		:	n_(n), avxSupported(availableAVX()),
			x_(n), w_(n)
	{
		gauleg(usecilkoropenmp);
	}
}
