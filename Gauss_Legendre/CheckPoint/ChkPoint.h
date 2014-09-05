#ifndef _CHKPOINT_H_
#define _CHKPOINT_H_

#include <utility>
#include <boost/cstdint.hpp>

#if !defined(__INTEL_COMPILER) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
	#include <boost/noncopyable.hpp>
#endif

#if (_MSC_VER >= 1600)
	#include <memory>
#else
	#include <boost/checked_delete.hpp>
	#include <boost/interprocess/smart_ptr/unique_ptr.hpp>
#endif

namespace CheckPoint {
#if (_MSC_VER >= 1600)
	using std::unique_ptr;
#else
	using boost::interprocess::unique_ptr;
#endif

#ifdef _WIN32
	void usedmem();
#endif

	class ChkPoint
#if !defined(__INTEL_COMPILER) && !defined(__GXX_EXPERIMENTAL_CXX0X__)
		: private boost::noncopyable
#endif
	{
#if defined(__INTEL_COMPILER) || defined(__GXX_EXPERIMENTAL_CXX0X__)
		ChkPoint(const ChkPoint &) = delete;
		ChkPoint & operator=(const ChkPoint &) = delete;
		ChkPoint() = delete;
#endif
		struct ChkPointFastImpl;
		struct Timestamp;

		template <typename T>
		struct fastpimpl_deleter {
			void operator()(T * const p) const {
				FastArenaObject<sizeof(ChkPoint::ChkPointFastImpl)>::
					operator delete(reinterpret_cast<void *>(p));
			}
		};

		const unique_ptr<ChkPointFastImpl, fastpimpl_deleter<ChkPointFastImpl> > cfp;
	
	public:
#if (_MSC_VER >= 1700)
		typedef std::pair<boost::int64_t, boost::int64_t> dpair;
#else
		typedef std::pair<const double, const double> dpair;
#endif
		ChkPoint(const char * const func, int line);
		~ChkPoint();
		void checkpoint(const char * const func, int line);
#if (_MSC_VER < 1600)
		const
#endif
		ChkPoint::dpair ChkPoint::totalpassageoftime() const;
		void checkpoint_print() const;
	};
}

#endif	// _CHKPOINT_H_
