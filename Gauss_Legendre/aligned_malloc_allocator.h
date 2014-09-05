#include <cstddef>
#include <limits>
#include <boost/static_assert.hpp>
#ifdef _MSC_VER
    #include <malloc.h>
#else
    BOOST_STATIC_ASSERT(false)
#endif

namespace myallocator {
	template <typename T> class aligned_malloc_allocator {
	public:
		// typedefs
		typedef std::size_t size_type;
		typedef std::ptrdiff_t difference_type;
		typedef T * pointer;
		typedef const T * const_pointer;
		typedef T & reference;
		typedef const T & const_reference;
		typedef T value_type;
			
		// convert an allocator<T> to allocator<U>
		template <class U> 
		struct rebind { typedef aligned_malloc_allocator<U> other; };

		// constructors
		aligned_malloc_allocator() throw() {}
		aligned_malloc_allocator(aligned_malloc_allocator const &) throw() {}
		template <typename U>
		aligned_malloc_allocator(aligned_malloc_allocator<U> const &) throw() {}

		// destructor
		~aligned_malloc_allocator() throw() {}

		// メモリを割り当てる
		pointer allocate(size_type size, const_pointer hint = 0) {
			return reinterpret_cast<pointer>(_aligned_malloc(size * sizeof(T), 64));
		}

		// 割当て済みの領域を初期化する
		void construct(pointer p, const T & val) {
			new (reinterpret_cast<void *>(p)) T(val);
		}

		// メモリを解放する
		void deallocate(pointer p, size_type n) {
			_aligned_free(reinterpret_cast<void *>(p));
		}

		// 初期化済みの領域を削除する
		void destroy(pointer p)
		{ p->~T(); }

		// アドレスを返す
		pointer address(reference value) const
		{ return &value; }
		const_pointer address(const_reference value) const
		{ return &value; }
		
		// 割当てることができる最大の要素数を返す
		size_type max_size() const throw() {
			return (std::numeric_limits<std::size_t>::max() >> 2) / sizeof(T) ;
		}
	};

	template <typename T, typename U>
	inline bool operator ==(const aligned_malloc_allocator<T>&, const aligned_malloc_allocator<U>)
	{ return true; }

	template <typename T, typename U>
	inline bool operator !=(const aligned_malloc_allocator<T>&, const aligned_malloc_allocator<U>)
	{ return false; }
}
