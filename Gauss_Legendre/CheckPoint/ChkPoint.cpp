#include "ChkPoint.h"
#include "FastArenaObject.h"
#include <iostream>
#include <boost/cast.hpp>
#include <boost/format.hpp>
#include <boost/assert.hpp>
#include <boost/optional.hpp>

#if (_MSC_VER >= 1700)
	#include <chrono>
#else
	#include <boost/chrono/chrono.hpp>
	#include <boost/chrono/chrono_io.hpp>
	#include <boost/chrono/process_cpu_clocks.hpp>
#endif

#ifdef _WIN32
	#include <windows.h>
	#include <Psapi.h>
#if (_MSC_VER >= 1700)
	#include <system_error>
#else
	#include <boost/system/error_code.hpp>
	#include <boost/system/system_error.hpp>
#endif
	#pragma comment(lib, "Psapi.Lib")
#endif

namespace CheckPoint {
#if (_MSC_VER >= 1700)
	using namespace std::chrono;

	using std::milli;
#else
	using namespace boost::chrono;

	using boost::milli;
#endif

	struct ChkPoint::Timestamp {
		const char * func;
		int line;
		steady_clock::time_point realtime;
#ifdef _CPUREAL
		process_cpu_clock::time_point cputime;
#endif
	};

	struct ChkPoint::ChkPointFastImpl {
		static const int N = 30;
		int cur;
		ChkPoint::Timestamp points[N];
			
		ChkPointFastImpl() : cur(0) {}
		~ChkPointFastImpl() {}
	};

	ChkPoint::ChkPoint(const char * const func, int line)
	 :	cfp(reinterpret_cast<ChkPointFastImpl *>(FastArenaObject<sizeof(ChkPointFastImpl)>::
												 operator new(0)))
	{
		checkpoint(func, line);
	}

	void ChkPoint::checkpoint(const char * const func, int line)
	{
		BOOST_ASSERT(cfp->cur < ChkPoint::ChkPointFastImpl::N);

		Timestamp * const p = cfp->points + cfp->cur;
		p->func = func;
		p->line = line;
#ifdef _CPUREAL
		p->cputime = process_cpu_clock::now();
#endif
		p->realtime = steady_clock::now();

		cfp->cur++;
	}

#if (_MSC_VER < 1600)
	const
#endif
	ChkPoint::dpair ChkPoint::totalpassageoftime() const
	{
#ifdef _CPUREAL
		const process_cpu_clock::times time = (cfp->points[cfp->cur - 1].cputime -
			cfp->points[0].cputime).count();
		const duration<double, milliseconds> cputime(
			duration<double, nanoseconds>(time.user + time.system));
#endif
		const auto realtime = duration_cast<milliseconds>(cfp->points[cfp->cur - 1].realtime -
			cfp->points[0].realtime);

#ifdef _CPUREAL
		return std::make_pair(cputime.count(), realtime.count());
#else
		return std::make_pair(0, realtime.count());
#endif
	}

	void ChkPoint::checkpoint_print() const
	{
#ifdef _CPUREAL
		boost::optional<process_cpu_clock::time_point> prevcpu(boost::none);
#endif
		boost::optional<steady_clock::time_point> prevreal(boost::none);

		for (int i = 0; i < cfp->cur; i++) {
			Timestamp * const p = &cfp->points[i];

#ifdef _CPUREAL
			if (prevcpu && prevreal) {
				const process_cpu_clock::times time((p->cputime - *prevcpu).count());
				const duration<double, milliseconds> cputime(
					duration<double, nanoseconds>(time.user + time.system));
				const duration<double, milliseconds> realtime(p->realtime - *prevreal);

				std::cout << p->func << ", CPU����:" << boost::format("%.4f") % cputime
						  << " (msec), ������:" << boost::format("%.4f") % realtime
						  << " (msec), ���񉻌���:" << boost::format("%.4f") % (cputime / realtime)
						  << "�i�{�j" << std::endl;
#else
			if (prevreal) {
				const auto realtime(duration_cast<duration<double, milli>>(p->realtime - *prevreal));
				std::cout << p->func << " �ɂ����������� = "
						  << boost::format("%.4f") % realtime.count()
						  << " (msec)\n";
#endif
			}

#ifdef _CPUREAL
			prevcpu = boost::optional<process_cpu_clock::time_point>(p->cputime);
#endif
			prevreal = boost::optional<steady_clock::time_point>(p->realtime);
		}
	}

	ChkPoint::~ChkPoint()
	{
	}

#ifdef _WIN32
	void usedmem()
	{
		PROCESS_MEMORY_COUNTERS memInfo = {0};
		if (!::GetProcessMemoryInfo(::GetCurrentProcess(), &memInfo, sizeof(memInfo))) {
#if (_MSC_VER >= 1700)
			throw std::system_error(std::error_code(::GetLastError(), std::system_category()));
#else
			throw boost::system::system_error(boost::system::error_code(
											  ::GetLastError(),
											  boost::system::get_system_category()));
#endif
		}
		std::cout << "Used Memory Size: "
				  << boost::numeric_cast<boost::uint32_t>(memInfo.PeakWorkingSetSize >> 10)
				  << "(kB)" << std::endl; 
	}
#endif
}