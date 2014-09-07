﻿/*! \file CheckPoint.cpp
    \brief 時間計測のためのクラスの実装

    Copyright ©  2014 @dc1394 All Rights Reserved.
*/
#include "CheckPoint.h"
#include "FastArenaObject.h"
#include <array>                // for std::array
#include <chrono>               // for std::chrono
#include <iostream>             // for std::cout

#ifdef _WIN32
    #include <system_error>
#endif

#include <boost/assert.hpp>     // for boost::assert
#include <boost/cast.hpp>       // for boost::numeric_cast
#include <boost/format.hpp>     // for boost::format
#include <boost/optional.hpp>   // for boost::optional

#ifdef _WIN32
    #include <Windows.h>
    #include <Psapi.h>

	#pragma comment(lib, "Psapi.Lib")
#endif

namespace checkpoint {
    //! A structure.
    /*!
    チェックポイントの情報を格納する構造体
    */
	struct CheckPoint::Timestamp {
        //! A public member variable.
        /*!
        行数
        */
		std::int32_t line;

        //! A public member variable.
        /*!
        チェックポイントの名称
        */
        char const * action;

        //! A public member variable.
        /*!
        チェックポイントの時間
        */
        std::chrono::high_resolution_clock::time_point realtime;
	};

    //! A struct.
    /*!
    チェックポイントの情報の配列を格納する構造体
    */
	struct CheckPoint::CheckPointFastImpl {
        // #region コンストラクタ・デストラクタ

        //! A constructor.
        /*!
        唯一のコンストラクタ
        */
        CheckPointFastImpl() : cur(0) {}

        //! A destructor.
        /*!
        */
        ~CheckPointFastImpl() {}

        // #endregion コンストラクタ・デストラクタ

        // #region メンバ変数

        //! A public static member variable (constant).
        /*!
        チェックポイントの時間
        */
        static std::size_t const N = 30;

        //! A public member variable.
        /*!
        現在の場所
        */
		std::int32_t cur;
		
        //! A public member variable.
        /*!
        チェックポイントの情報の配列
        */
        std::array<CheckPoint::Timestamp, N> points;

        // #endregion メンバ変数
	};

    CheckPoint::CheckPoint()
        : cfp(
            reinterpret_cast<CheckPoint::CheckPointFastImpl *>(
                FastArenaObject<sizeof(CheckPoint::CheckPointFastImpl)>::operator new(0)))
	{
	}

    CheckPoint::~CheckPoint()
    {
    }

    void CheckPoint::checkpoint(char const * action, std::int32_t line)
	{
		BOOST_ASSERT(cfp->cur < CheckPoint::CheckPointFastImpl::N);

		auto const p = cfp->points.begin() + cfp->cur;

        p->action = action;
        p->line = line;
		p->realtime = std::chrono::high_resolution_clock::now();

		cfp->cur++;
	}
	
	void CheckPoint::checkpoint_print() const
	{
        using namespace std::chrono;

        boost::optional<high_resolution_clock::time_point> prevreal(boost::none);

        auto itr = cfp->points.begin();
		for (std::int32_t i = 0; i < cfp->cur; ++i, ++itr) {
			if (prevreal) {
				auto const realtime(duration_cast<duration<double, std::milli>>(itr->realtime - *prevreal));
				std::cout << itr->action << " にかかった時間 = "
						  << boost::format("%.4f") % realtime.count()
						  << " (msec)\n";
			}

            prevreal = boost::optional<high_resolution_clock::time_point>(itr->realtime);
		}
	}

    double CheckPoint::totalpassageoftime() const
    {
        using namespace std::chrono;

        auto const realtime = duration_cast<duration<double, std::milli>>(
            cfp->points[cfp->cur - 1].realtime - cfp->points[0].realtime);

        return realtime.count();
    }

#ifdef _WIN32
	void usedmem()
	{
		PROCESS_MEMORY_COUNTERS memInfo = { 0 };
		
        if (!::GetProcessMemoryInfo(::GetCurrentProcess(), &memInfo, sizeof(memInfo))) {
			throw std::system_error(std::error_code(::GetLastError(), std::system_category()));
		}

		std::cout << "Used Memory Size: "
				  << boost::numeric_cast<std::uint32_t>(memInfo.PeakWorkingSetSize >> 10)
				  << "(kB)" << std::endl; 
	}
#endif
}