#ifndef _ARRAYIEDALLOCATOR_H_
#define _ARRAYIEDALLOCATOR_H_

#pragma once

#include <boost/static_assert.hpp>

namespace checkpoint {
	template <std::size_t TTypeSize, std::size_t TNumArray>
	class ArraiedAllocator final
	{
#if defined(__INTEL_COMPILER) || defined(__GXX_EXPERIMENTAL_CXX0X__)
		ArraiedAllocator(const ArraiedAllocator &) = delete;
		ArraiedAllocator & operator=(const ArraiedAllocator &) = delete;
		static constexpr int MAX_SIZE = TNumArray;
#else
		static const int MAX_SIZE = TNumArray;
#endif
		// サイズは絶対０より大きくなくちゃダメ
		BOOST_STATIC_ASSERT(TNumArray > 0);


		struct Item {
			union {
				char value_[TTypeSize];
				struct Item * next_;
			};
		};

		static Item items_[MAX_SIZE];
		static Item * first_;
		static ArraiedAllocator allocator_;
		ArraiedAllocator();

	public:
		static void * Alloc() {
			Item * ret = first_;
			first_ = ret->next_;
			return reinterpret_cast<void *>(ret);
		}
		static void Free(void * item) {
			Item * rev = reinterpret_cast<Item *>(item);
			rev->next_ = first_;
			first_ = rev;
		}
		static ArraiedAllocator& GetAllocator() { return allocator_; }
		static int Max() { return MAX_SIZE; }
	};

	template <std::size_t TTypeSize, std::size_t TNumArray>
	typename ArraiedAllocator<TTypeSize,TNumArray>::Item
		ArraiedAllocator<TTypeSize,TNumArray>::items_[ArraiedAllocator<TTypeSize,TNumArray>::MAX_SIZE];

	template <std::size_t TTypeSize, std::size_t TNumArray>
	typename ArraiedAllocator<TTypeSize, TNumArray>::Item* ArraiedAllocator<TTypeSize,TNumArray>::first_;

	template <std::size_t TTypeSize, std::size_t TNumArray>
	ArraiedAllocator<TTypeSize,TNumArray> ArraiedAllocator<TTypeSize,TNumArray>::allocator_;

	template <std::size_t TTypeSize, std::size_t TNumArray>
	inline ArraiedAllocator<TTypeSize,TNumArray>::ArraiedAllocator() {
		first_ = &items_[0];
		for (int i = 0; i < TNumArray; i++) {
			items_[i].next_ = &items_[i + 1];
		}
		items_[TNumArray - 1].next_ = NULL;
	}
}

#endif // _ARRAYIEDALLOCATOR_H_
