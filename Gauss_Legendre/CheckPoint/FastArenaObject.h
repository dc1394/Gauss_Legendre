#ifndef _FASTARENAOBJECT_H_
#define _FASTARENAOBJECT_H_

#pragma once

#include "ArraiedAllocator.h"

namespace checkpoint {
	template <size_t TTypeSize, size_t TNumArray = 1>
	struct FastArenaObject final
	{
#if defined(__INTEL_COMPILER) || defined(__GXX_EXPERIMENTAL_CXX0X__)
		FastArenaObject(const FastArenaObject &) = delete;
		FastArenaObject & operator=(const FastArenaObject &) = delete;
		FastArenaObject() = delete;
#endif
		// サイズは絶対０より大きくなくちゃダメ
		BOOST_STATIC_ASSERT(TNumArray > 0);

		static void * operator new(std::size_t dummy) {
			return ArraiedAllocator<TTypeSize, TNumArray>::GetAllocator().Alloc();
		}
		static void operator delete(void * p) {
			ArraiedAllocator<TTypeSize, TNumArray>::GetAllocator().Free(p);
		}
	};
}

#endif // _FASTARENAOBJECT_H_
