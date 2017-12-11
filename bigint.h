// -std=c++14 -O3 -march=native [-g -Wall -Wextra]
// change 8 to CHAR_BITS(?) from limits.h (std?)
// align vectors by cache line + TLS
// test <op=> with self-argument

#pragma once

#include <vector>
#include <stack>
#include <string>
#include <stdexcept>
#include <thread>
#include <cstring> // memcpy

//#include <sys/mman.h>
//#include <sys/types.h>
#include <stdlib.h> // posix_memalign
#include <unordered_map>

//#include <x86intrin.h>


#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1)
#endif
#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif


// It seems to be a bug: static thread_local initialization of template class member generates an UD2 instruction
class BucketPoolAllocatorBase
{
protected:
	/*class pool_t : public std::vector<std::vector<void*>>
	{
	public:
		~pool_t()
		{
			size_t n = m_pool.size();
			std::cout << "Allocator stats:" << std::endl << "\tBuckets: " << n << std::endl;
			for(size_t i = 0; i < n; ++i)
			{
				if(size_t blocks = m_pool[i].size())
					std::cout << "\t" << i << ": " << blocks << std::endl;
			}
		}
	};//*/
	using pool_t = std::vector<std::vector<void*>>;
	thread_local static pool_t m_pool;

	bool operator==(const BucketPoolAllocatorBase& in_allocator) const noexcept { return &m_pool == &in_allocator.m_pool; }
};
thread_local BucketPoolAllocatorBase::pool_t BucketPoolAllocatorBase::m_pool;

template<typename T, bool Align = true>
class BucketPoolAllocator : public BucketPoolAllocatorBase
{
	static const unsigned BlockGranularity = 64;
	//static const unsigned MaxBinSize = 4;

	/*class SpinLock
	{
		std::atomic_bool m_busy;

	public:
		SpinLock(): m_busy(false)
		{
			if(!std::atomic_is_lock_free(&m_busy))
				throw std::runtime_error("The atomic object is not lock free");
		}

		void lock()
		{
			using namespace std::chrono_literals;
			for(bool expected = false; !m_busy.compare_exchange_weak(expected, true); expected = false)
				std::this_thread::sleep_for(1us);
		}
		void unlock() { m_busy = false; }
	};//*/

public:
	using value_type = T;

	BucketPoolAllocator() = default;
	template<typename U>
	constexpr BucketPoolAllocator(const BucketPoolAllocator<U>&) noexcept {}

	bool operator==(const BucketPoolAllocator& in_allocator) const noexcept { return BucketPoolAllocatorBase::operator==(in_allocator); }

	T* allocate(size_t in_elements)
	{
		size_t size_aligned = aligned_size(in_elements);
		if(size_aligned / sizeof(T) < in_elements)
			throw std::bad_alloc();

		size_t i = bucket_index(size_aligned);
		if(i < m_pool.size())
		{
			auto& blocks = m_pool[i];
			if(!blocks.empty())
			{
				auto p = *blocks.rbegin();
				blocks.pop_back();
				return static_cast<T*>(p);
			}
		}

		if(Align)
		{
			void* p;
			if(posix_memalign(&p, BlockGranularity, size_aligned))
				throw std::bad_alloc();
			return static_cast<T*>(p);
		}
		else
			return new T[size_aligned / sizeof(T)];
			//return new T[(size_aligned + sizeof(T) - 1) / sizeof(T)];
	}

	void deallocate(T* in_addr, size_t in_elements) noexcept
	{
		size_t size_aligned = aligned_size(in_elements);
		size_t i = bucket_index(size_aligned);
		if(i >= m_pool.size())
			m_pool.resize(i + 1);

		auto& buckets = m_pool[i];
		//if(buckets.size() < MaxBinSize)
			buckets.push_back(in_addr);
		//else
		//	delete [] in_p;
	}

private:
	static constexpr size_t aligned_size(size_t in_elements)
	{
		return (in_elements * sizeof(T) + BlockGranularity - 1) & ~(BlockGranularity - 1);
	}
	static constexpr size_t bucket_index(size_t in_size_aligned)
	{
		return in_size_aligned / BlockGranularity - 1;
	}
	static constexpr size_t bucket_size(size_t in_index)
	{
		return (in_index + 1) * BlockGranularity;
	}
};
template<class T, class U>
bool operator==(const BucketPoolAllocator<T>& in_allocator_a, const BucketPoolAllocator<U>& in_allocator_b) noexcept
{
	return in_allocator_a == in_allocator_b;
}
template<class T, class U>
bool operator!=(const BucketPoolAllocator<T>& in_allocator_a, const BucketPoolAllocator<U>& in_allocator_b) noexcept
{
	return !(in_allocator_a == in_allocator_b);
}

template<typename T>
class StackAllocator
{
public:
	using value_type = T;

	StackAllocator(void* in_mem, size_t in_size):
		m_mem(in_mem),
		m_elements(in_size / sizeof(T))
	{}

	T* allocate(size_t in_elements)
	{
		if(in_elements > m_elements)
		{
			class alloca_overflow : public std::bad_alloc {};
			throw alloca_overflow();
		}
		if(m_allocated)
		{
			class alloca_already_allocated : public std::bad_alloc {};
			throw alloca_already_allocated();
		}
		
		m_allocated = true;
		return static_cast<T*>(m_mem);
	}

	void deallocate(T*, size_t) noexcept { m_allocated = false; }

private:
	void*	m_mem		= nullptr;
	size_t	m_elements	= 0;
	bool	m_allocated	= false;
};
template<class T, class U>
bool operator==(const StackAllocator<T>& in_allocator_a, const StackAllocator<U>& in_allocator_b) noexcept { return in_allocator_a.m_mem == in_allocator_b.m_mem; }
template<class T, class U>
bool operator!=(const StackAllocator<T>& in_allocator_a, const StackAllocator<U>& in_allocator_b) noexcept { return !(in_allocator_a == in_allocator_b); }



class uint_big_t
{
public:
	using element_t = std::conditional_t<sizeof(void*) == sizeof(uint64_t), uint64_t, uint32_t>;

private:
	using element_ex_t = std::conditional_t<sizeof(void*) == sizeof(uint64_t), __uint128_t, uint64_t>;
	using element_half_t = std::conditional_t<sizeof(void*) == sizeof(uint64_t), uint32_t, uint16_t>;

	static const size_t m_bits_per_element = sizeof(element_t) * 8;
	static const size_t m_bits_per_half_element = sizeof(element_half_t) * 8;

	static_assert(std::is_unsigned<element_t>::value, "base element type must be unsigned");
	static_assert(sizeof(element_ex_t) == sizeof(element_t) * 2, "extended element type must be twice as large as a base element type");
	static_assert(sizeof(element_half_t) * 2 == sizeof(element_t), "half element type must be twice as small as a base element type");

	using Allocator = BucketPoolAllocator<element_t>;
	//using Allocator = std::allocator<element_t>;
	using storage_t = std::vector<element_t, Allocator>;

	using stack_allocator_t = StackAllocator<element_t>;
	using local_vector_t = std::vector<element_t, stack_allocator_t>;

public:
	// ctor-s
	uint_big_t() = default;

	uint_big_t(const char* const in_str, unsigned in_radix = 10)
	{
		element_t t_factor = 1;
		element_t max_t_factor = std::numeric_limits<element_t>::max() / in_radix;
		element_t t = 0;

		auto p = in_str;
		for(; *p == '0'; ++p) {}
		for(; *p; ++p)
		{
			if(t_factor >= max_t_factor)
			{
				operator*=(t_factor);
				t_factor = 1;

				operator+=(t);
				t = 0;
			}
			
			t_factor *= in_radix;
			t *= in_radix;
			switch(in_radix)
			{
			case 10:
				t += *p - '0';
				break;

			default:
				throw std::invalid_argument("Invalid radix " + std::to_string(in_radix));
			}
		}
		if(t_factor != 1)
		{
			operator*=(t_factor);
			operator+=(t);
		}
	}

	uint_big_t(const std::string& in_str, unsigned in_radix = 10):
		uint_big_t(in_str.c_str(), in_radix)
	{}

	explicit uint_big_t(element_t in_x): m_value(in_x ? 1 : 0, in_x) {}
	explicit uint_big_t(bool in_x): uint_big_t(element_t{in_x}) {}
	explicit uint_big_t(unsigned in_x): uint_big_t(element_t{in_x}) {}

	//uint_big_t(uint_big_t&& in_x): m_value(std::move(in_x.m_value)) { LG("move ctor"); }
	//uint_big_t& operator=(uint_big_t&& in_x) { m_value = std::move(in_x.m_value); LG("move ="); return *this; }

	//uint_big_t(const uint_big_t& in_x): m_value(in_x.m_value) { LG("copy ctor"); }
	//uint_big_t& operator=(const uint_big_t& in_x) { m_value = in_x.m_value; LG("copy ="); return *this; }

	// Output
	std::string str(unsigned in_radix = 10) const
	{
		auto to_hex = [](unsigned in_x) -> char
		{
			if(in_x < 10)
				return in_x + '0';
			else
				return in_x - 10 + 'A';
		};

		std::stack<char> digits;
		if(in_radix == 16)
		{
			for(const unsigned char *p = reinterpret_cast<const unsigned char*>(&m_value[0]), *pEnd = reinterpret_cast<const unsigned char*>(&*m_value.rbegin() + 1);
				p < pEnd;
				++p)
			{
				digits.push( to_hex(*p & 0x0F) );
				digits.push( to_hex(*p >> 4) );
			}

			while(!digits.empty() && digits.top() == '0')
				digits.pop();
		}
		else
		{
			element_t r;
			for(uint_big_t x{*this}; static_cast<bool>(x); )
			{
				div(x, in_radix, &x, &r);
				digits.push( to_hex(r) );
			}
		}

		if(digits.empty())
			return "0";
		else
		{
			std::ostringstream oss;
			for(; !digits.empty(); digits.pop())
				oss << digits.top();
			return oss.str();
		}
	}

	explicit operator element_t() const
	{
		size_t elements = m_value.size();
		if(elements > 2)
		{
			class bad_element_cast : public std::bad_cast
			{
			public:
				bad_element_cast(const std::string& in_str): m_what(in_str) {}
				const char* what() const noexcept override { return m_what.c_str(); }
			private:
				std::string m_what;
			};

			throw bad_element_cast("The number is too large for the " + std::string(typeid(element_t).name()) + " type");
		}
		else if(elements)
			return m_value[0];
		else
			return 0;
	}

	explicit operator bool() const { return !m_value.empty(); }
	
	// Cmp operators
	bool operator<(const uint_big_t& in_x) const { return cmp_l(m_value, in_x.m_value); }
	bool operator>(const uint_big_t& in_x) const { return cmp_g(m_value, in_x.m_value); }

	bool operator<=(const uint_big_t& in_x) const { return cmp_le(m_value, in_x.m_value); }
	bool operator>=(const uint_big_t& in_x) const { return cmp_ge(m_value, in_x.m_value); }

	bool operator==(const element_t in_x) const { return (m_value.size() == 1 && m_value[0] == in_x); }
	bool operator==(const uint_big_t& in_x) const
	{
		size_t len = m_value.size();
		if(len != in_x.m_value.size())
			return false;

		for(const element_t *p = &m_value[0], *p_x = &in_x.m_value[0]; len; --len, ++p, ++p_x)
		{
			if(*p != *p_x)
				return false;
		}

		return true;
	}

	// Arithmetic <op>= operators
	uint_big_t& operator+=(element_t in_x)
	{
		size_t len = m_value.size();
		if(!len)
		{
			m_value.push_back(in_x);
			return *this;
		}

		bool carry = add(m_value[0], in_x, false);
		for(size_t i = 1; carry && i < len; ++i)
			carry = add(m_value[i], 0, carry);
		if(carry)
			m_value.push_back(carry);

		return *this;
	}

	uint_big_t& operator+=(const uint_big_t& in_x)
	{
		auto& value_x = in_x.m_value;
		size_t len_x = value_x.size();
		if(!len_x)
			return *this;

		size_t len_this = m_value.size();

		bool carry = false;
		for(size_t i = 0, n = std::min(len_this, len_x); i < n; ++i)
			carry = add(m_value[i], value_x[i], carry);

		if(len_this < len_x)
		{
			for(size_t i = len_this; i < len_x; ++i)
			{
				element_t x = 0;
				carry = add(x, value_x[i], carry);
				m_value.push_back(x);
			}
		}
		else
		{
			for(size_t i = len_x; carry && i < len_this; ++i)
				carry = add(m_value[i], 0, true);
		}
		if(carry)
			m_value.push_back(carry);

		return *this;
	}

	uint_big_t& operator-=(element_t in_x)
	{
		if(!in_x)
			return *this;

		size_t len_this = m_value.size();
		if(!len_this)
			throw std::overflow_error("The first argument of an unsigned negation cannot be less than the second argument");

		bool carry = sub(m_value[0], in_x, false);
		for(size_t i = 1; i < len_this; ++i)
			carry = sub(m_value[i], 0, carry);
		if(carry)
			throw std::overflow_error("The result of an unsigned negation is negative");

		normalize(len_this);
		return *this;
	}

	uint_big_t& operator-=(const uint_big_t& in_x) { sub(m_value, in_x.m_value); return *this; }
	uint_big_t& operator*=(element_t in_x) { mul(m_value, in_x); return *this; }

	/*uint_big_t& operator*=(const uint_big_t& in_x)
	{
		auto& value_x = in_x.m_value;
		const size_t size = m_value.size();
		const size_t size_x = value_x.size();
		if(!size || !size_x)
		{
			m_value.clear();
			return *this;
		}

		const size_t sz_small = (size < size_x) ? size : size_x;
		const size_t sz_large = (size >= size_x) ? size : size_x;
		auto& v_small = (size < size_x) ? m_value : value_x;
		auto& v_large = (size >= size_x) ? m_value : value_x;

		std::vector<element_t, Allocator> z(sz_small + sz_large);

		const size_t wnd_max = sz_small;
		const size_t steps = sz_small + sz_large - 1;
		size_t i_z = 0;

		auto mul_add = [&z](element_t in_a, element_t in_b, size_t in_i_z)
		{
			element_ex_t c = element_ex_t{in_a} * in_b;

			bool carry = add(z[in_i_z], static_cast<element_t>(c), false);
			carry = add(z[++in_i_z], c >> m_bits_per_element, carry);
			while(carry)
				carry = add(z[++in_i_z], 0, true);
		};

		// low
		for(size_t i_mid = wnd_max - 1, wnd = 1; i_z < i_mid; ++i_z, ++wnd)
		{
			for(size_t i = wnd - 1, j = 0; j < wnd; --i, ++j)
				mul_add(v_large[i], v_small[j], i_z);
		}

		// middle
		for(size_t i_high = steps - wnd_max + 1, i_large = 0; i_z < i_high; ++i_z, ++i_large)
		{
			for(size_t i = i_large, j = sz_small - 1; j < wnd_max; ++i, --j)
				mul_add(v_large[i], v_small[j], i_z);
		}

		// high
		for(size_t wnd = wnd_max - 1; i_z < steps; ++i_z, --wnd)
		{
			for(size_t i = sz_large - 1, j = sz_small - wnd; j < sz_small; --i, ++j)
				mul_add(v_large[i], v_small[j], i_z);
		}

		m_value = std::move(z);
		normalize();

		return *this;
	}//*/

	uint_big_t& operator*=(const uint_big_t& in_x)
	{
		const size_t size = m_value.size();
		const size_t size_x = in_x.m_value.size();
		auto x_ptr = &in_x.m_value[0];

		storage_t y(size + size_x);

		for(size_t i = 0; i < size_x; ++i)
		{
			element_ex_t x{*(x_ptr + i)};
			auto res_ptr = &y[i];
			element_t carry = 0;
			bool res_carry = false;

			for(size_t j = 0; j < size; ++j, ++res_ptr)
			{
				element_ex_t z = x * m_value[j] + carry;
				carry = z >> m_bits_per_element;
				res_carry = add(*res_ptr, static_cast<element_t>(z), res_carry);
			}
			*res_ptr = res_carry + carry;
		}

		m_value = std::move(y);
		normalize(m_value.size());

		return *this;
	}//*/
	/*uint_big_t& operator*=(const uint_big_t& in_x)
	{
		const size_t size = m_value.size();
		const size_t size_x = in_x.m_value.size();
		auto x_ptr = &in_x.m_value[0];

		storage_t y(size + size_x);

		for(size_t i = 0; i < size_x; ++i)
		{
			element_ex_t x{*(x_ptr + i)};
			auto res_ptr = &y[i];
			element_t carry = 0;
			bool res_carry = false;

			for(size_t j = 0; j < size; ++j, ++res_ptr)
			{
				element_ex_t z = x * m_value[j] + carry;
				carry = z >> m_bits_per_element;
				res_carry = add(*res_ptr, static_cast<element_t>(z), res_carry);
			}
			*res_ptr = res_carry + carry;
		}

		m_value = std::move(y);
		normalize(m_value.size());

		return *this;
	}//*/

	uint_big_t& operator/=(const uint_big_t& in_x)
	{
		div(*this, in_x, this, nullptr);
		return *this;
	}
	uint_big_t& operator/=(element_t in_x)
	{
		div(*this, in_x, this, nullptr);
		return *this;
	}

	uint_big_t& operator%=(const uint_big_t& in_x)
	{
		div(*this, in_x, nullptr, this);
		return *this;
	}
	uint_big_t& operator%=(element_t in_x)
	{
		element_t r;
		div(*this, in_x, nullptr, &r);
		*this = uint_big_t{r};
		return *this;
	}

	uint_big_t& operator^(element_t in_x)
	{
		uint_big_t y{1u};
		uint_big_t pow = *this;

		for(element_t x = in_x; x; x >>= 1)
		{
			if(x & 1)
				y *= pow;
			pow *= pow;
		}

		*this = std::move(y);
		return *this;
	}

	// Shift operators
	uint_big_t& operator<<=(size_t in_shift) { shl(m_value, in_shift); return *this; }
	uint_big_t& operator>>=(size_t in_shift) { shr(m_value, in_shift); return *this;	}

	// Advanced
	uint_big_t& pow_mod(const uint_big_t& in_pow, const uint_big_t& in_mod)
	{
		uint_big_t y{1u};
		uint_big_t pow = *this;

		if(likely(static_cast<bool>(in_pow)))
		{
			//size_t pow_limit = pow.m_value.size() + 1;
			//size_t y_limit = y.m_value.size() * 2;
			auto value_pow = &in_pow.m_value[0];
			size_t size_pow = in_pow.m_value.size();

			for(size_t i = 0, n = in_pow.msb(); i <= n; ++i)
			{
				if(in_pow.get_bit_unsafe(value_pow, size_pow, i))
				{
					y *= pow;
					//if(y.m_value.size() > y_limit)
						y %= in_mod;
				}
				pow *= pow;
				//if(pow.m_value.size() > pow_limit)
					pow %= in_mod;
			}
		}
		y %= in_mod;

		*this = std::move(y);
		return *this;
	}

	// Other
	static void div(const uint_big_t& in_x, const uint_big_t& in_y, uint_big_t* out_z_ptr, uint_big_t* out_r_ptr)
	{
		if(in_x.m_value.empty())
		{
			if(out_z_ptr)
				out_z_ptr->m_value.clear();
			if(out_r_ptr)
				out_r_ptr->m_value.clear();
			return;
		}
		if(unlikely(in_y.m_value.empty()))
			throw std::invalid_argument("Division by zero");

		// call clear() last, because it can be &in_x, which is used for output as well
		if(in_x < in_y)
		{
			if(out_r_ptr)
				*out_r_ptr = in_x;
			if(out_z_ptr)
				out_z_ptr->m_value.clear();
			return;
		}
		/*if(unlikely(in_y == element_t{1}))
		{
			if(out_z_ptr)
				*out_z_ptr = in_x;
			if(out_r_ptr)
				out_r_ptr->m_value.clear();
			return;
		}//*/

		const size_t x_size = in_x.m_value.size() * 2 - ((*in_x.m_value.rbegin() >> m_bits_per_half_element) ? 0 : 1);
		const size_t y_size = in_y.m_value.size() * 2 - ((*in_y.m_value.rbegin() >> m_bits_per_half_element) ? 0 : 1);

		size_t shifts = x_size - y_size;

		size_t max_size = in_x.m_value.size();
		size_t stack_alloc_size = in_x.m_value.capacity() * sizeof(decltype(in_x.m_value)::value_type);

		storage_t x(in_x.m_value);

		void* placement_y = alloca(stack_alloc_size);
		stack_allocator_t allocator_y(placement_y, stack_alloc_size);
		local_vector_t y(max_size, allocator_y);

		storage_t z(shifts / (sizeof(storage_t::value_type) / sizeof(decltype(y)::value_type)) + 1);

		auto x_value = reinterpret_cast<element_half_t*>(&x[0]);
		auto y_value = reinterpret_cast<element_half_t*>(&y[0]);
		auto z_value = reinterpret_cast<element_half_t*>(&z[0]);

		// Init+Shl Y
		memcpy(y_value + shifts, &in_y.m_value[0], y_size * sizeof(*y_value));

		// Y*F: reserve 1 extra element for overflows during mul
		const size_t y_f_alloca_size = stack_alloc_size + sizeof(local_vector_t::value_type);
		void* placement_y_f = alloca(y_f_alloca_size);
		stack_allocator_t allocator_y_f(placement_y_f, y_f_alloca_size);
		local_vector_t y_f(allocator_y_f);
		y_f.reserve(y_f_alloca_size / sizeof(decltype(y_f)::value_type));

		auto adjust_sub = [&y_f](auto& io_x, const auto& in_y, element_half_t in_factor)
		{
			auto assign = [](auto& io_x, const auto& in_y)
			{
				size_t size = in_y.size();
				io_x.resize(size);
				std::memcpy(&io_x[0], &in_y[0], size * sizeof(element_t));
			};

			assign(y_f, in_y);
			mul(y_f, in_factor);

			if(cmp_g(y_f, io_x))
			{
				// y_size is always > 1 here.
				// It is 1 only for the last iteration during a divison by a single const,
				// but in this case a factor is always correct and doesn't require an adjustment
				// (the check above will always fail)
				sub(y_f, io_x);
				if(cmp_g(y_f, in_y))
				{
					auto y_f_value = reinterpret_cast<element_half_t*>(&y_f[0]);
					auto in_y_value = reinterpret_cast<const element_half_t*>(&in_y[0]);
					size_t highest = in_y.size() * 2 - 1;
					if(!in_y_value[highest])
						--highest;

					element_t a = *reinterpret_cast<element_t*>(y_f_value + highest - 1);
					element_t b = *reinterpret_cast<const element_t*>(in_y_value + highest - 1);
					in_factor -= a / b + 1;

					assign(y_f, in_y);
					mul(y_f, in_factor);
					sub(io_x, y_f);
					if(cmp_ge(io_x, in_y))
					{
						++in_factor;
						sub(io_x, in_y);
					}
				}
				else
				{
					--in_factor;
					assign(io_x, in_y);
					sub(io_x, y_f);
				}
			}
			else
				sub(io_x, y_f);

			return in_factor;
		};

		bool skip_cmp = false;
		//asm volatile("int $3");
		size_t cur_element;
		for(cur_element = x_size - 1; shifts; --cur_element, --shifts)
		{
			if(skip_cmp)
			{
				element_t f;
				if(x_value[cur_element + 1] == y_value[cur_element])
					f = std::numeric_limits<element_half_t>::max();
				else
					f = *reinterpret_cast<element_t*>(x_value + cur_element) / y_value[cur_element];
				z_value[shifts] = adjust_sub(x, y, f);
			}
			else if(cmp_ge(x, y))
			{
				element_t f;
				if(y_size == 1)
					f = x_value[cur_element] / y_value[cur_element];
				else
				{
					element_t x_ex = *reinterpret_cast<element_t*>(x_value + cur_element - 1);
					element_t y_ex = *reinterpret_cast<element_t*>(y_value + cur_element - 1);
					f = x_ex / y_ex;
				}
				z_value[shifts] = adjust_sub(x, y, f);
			}

			skip_cmp = (x.size() > cur_element / 2) && x_value[cur_element];
			shr(y, m_bits_per_half_element);
		}

		// Final check (when y is not shifted anymore)
		if(skip_cmp)
		{
			element_t f;
			if(x_value[cur_element + 1] == y_value[cur_element])
				f = std::numeric_limits<element_half_t>::max();
			else
				f = *reinterpret_cast<element_t*>(x_value + cur_element) / y_value[cur_element];
			z_value[0] = adjust_sub(x, y, f);
		}
		else if(cmp_ge(x, y))
		{
			auto f = x_value[cur_element] / y_value[cur_element];
			z_value[0] = adjust_sub(x, y, f);
		}

		if(out_z_ptr)
		{
			size_t elements = z.size();
			out_z_ptr->m_value = std::move(z);
			out_z_ptr->normalize(elements);
		}
		if(out_r_ptr)
			out_r_ptr->m_value = std::move(x);
	}//*/
	/*uint_big_t& operator/=(element_t in_y)
	{
		const size_t x_size = m_value.size();
		if(!x_size)
		{
			m_value.clear();
			return *this;
		}
		if(unlikely(in_y == 0))
			throw std::invalid_argument("Division by zero");

		auto div_mod = [](element_t in_x_low, element_t in_x_high, element_t in_y, element_t& out_r)
		{
			uint64_t result;
			asm volatile
			(
				"div %[factor]\n"
				: "=a"(result), "=d"(out_r)
				: "a"(in_x_low), "d"(in_x_high), [factor]"r"(in_y)
			);
			return result;
		};

		//asm volatile("int $3");
		auto value = &m_value[0];
		element_t x_high = 0;
		for(size_t cur_element = x_size - 1; cur_element < x_size; --cur_element)
			value[cur_element] = div_mod(value[cur_element], x_high, in_y, x_high);

		normalize(x_size);
		return *this;
	}//*/
	//template<typename U>
	//static std::enable_if_t<std::is_arithmetic<U>::value, void>
	static void div(const uint_big_t& in_x, element_t in_y, uint_big_t* out_z_ptr, element_t* out_r_ptr)
	{
		const size_t x_size = in_x.m_value.size();

		if(!x_size)
		{
			if(out_z_ptr)
				out_z_ptr->m_value.clear();
			if(out_r_ptr)
				*out_r_ptr = 0;
			return;
		}
		if(unlikely(in_y == 0))
			throw std::invalid_argument("Division by zero");

		auto x_value = &in_x.m_value[0];

		storage_t z(x_size);
		auto z_value = &z[0];

		auto div_mod = [](element_t in_x_low, element_t in_x_high, element_t in_y, element_t& out_r)
		{
			uint64_t result;
			asm volatile
			(
				"div %[factor]\n"
				: "=a"(result), "=d"(out_r)
				: "a"(in_x_low), "d"(in_x_high), [factor]"r"(in_y)
			);
			return result;
		};

		element_t x_high = 0;
		for(size_t cur_element = x_size - 1; cur_element < x_size; --cur_element)
		{
			z_value[cur_element] = div_mod(x_value[cur_element], x_high, in_y, x_high);
		}

		if(out_z_ptr)
		{
			out_z_ptr->m_value = std::move(z);
			out_z_ptr->normalize(x_size);
		}
		if(out_r_ptr)
			*out_r_ptr = x_high;
	}//*/

	/*static void div(const uint_big_t& in_x, const uint_big_t& in_y, uint_big_t* out_z_ptr, uint_big_t* out_r_ptr)
	{
		if(in_x.m_value.empty())
		{
			if(out_z_ptr)
				out_z_ptr->m_value.clear();
			if(out_r_ptr)
				out_r_ptr->m_value.clear();
			return;
		}
		if(in_y.m_value.empty())
			throw std::invalid_argument("Division by zero");

		size_t msb_x = in_x.msb();
		size_t msb_y = in_y.msb();
		if(msb_x < msb_y)
		{
			// out_z_ptr can be &in_x, so reset it last
			if(out_r_ptr)
				*out_r_ptr = in_x;
			if(out_z_ptr)
				out_z_ptr->m_value.clear();
			return;
		}
		if(unlikely(msb_y == 0))
		{
			if(out_z_ptr)
				*out_z_ptr = in_x;
			if(out_r_ptr)
				out_r_ptr->m_value.clear();
			return;
		}
		size_t shifts = msb_x - msb_y;
		//size_t bits_y = msb_y + 1;

		uint_big_t x{in_x};
		uint_big_t y{in_y}; y <<= shifts;

		bool skip_cmp = false;
		auto x_value = &x.m_value[0];
		auto y_value = &y.m_value[0];
		size_t size_x = x.m_value.size();
		size_t size_y = y.m_value.size();
		storage_t z(shifts / m_bits_per_element + 1);

		//auto sub = [x_value, y_value, &size_x, &size_y](size_t in_offset)
		//{
		//	bool carry = true; // z = x + (not(y) + _1_)
		//	size_t i;
		//	for(i = in_offset; i != size_y; ++i)
		//		carry = add(x_value[i], ~y_value[i], carry);
		//	for(; i != size_x; ++i)
		//		carry = add(x_value[i], ~element_t{0}, carry);
		//};
		auto shr1 = [](element_t* in_x, size_t in_size)
		{
			auto p_last = in_x + in_size - 1;

			element_t carry = 0;
			for(auto p = p_last, p_lower_bound = in_x - 1; p != p_lower_bound; --p)
			{
				element_t x = *p;
				*p = carry | (x >> 1);
				carry = x << (m_bits_per_element - 1);
			}

			return !*p_last;
		};

		for(size_t cur_bit = msb_x; shifts; --cur_bit, --shifts)
		{
			if(skip_cmp || x >= y)
			//if(skip_cmp || cmp_ge(x_value, size_x, y_shift, lower_bound))
			{
				size_t lower_bound = shifts / m_bits_per_element;

				z[lower_bound] |= (element_t{1} << (shifts % m_bits_per_element));

				x -= y;
				//_sub(x, y);
				//size_x = _sub(x, y);

				//sub(lower_bound);
				//size_x = x.normalize(size_x);

				//sub(&x_value[0], size_x, lower_bound, &y_shift[0], y_shift.size());
				//size_x = x.normalize(size_x);
				//size_x = x.m_value.size();
			}
			// 1. x has the cur_bit set but y is larger - x is guaranteed to be larger at the next iteration so the next cmp can be skipped
			// 2. p.1 took place at the previous step and at the current step the cur_bit is set after a subtraction
			skip_cmp = x.get_bit(cur_bit);
			//skip_cmp = get_bit_unsafe(x_value, size_x, cur_bit);

			if(shr1(y_value, size_y))
			{
				y.m_value.pop_back();
				--size_y;
			}
		}
		//x.m_value.resize(size_x);

		// Final check (when y is not shifted anymore)
		if(x >= y)
		//if(size_x && (skip_cmp || cmp_ge(x_value, size_x, y_shifts[0], 0)))
		{
			z[0] |= 1;
			x -= y;
		}

		if(out_z_ptr)
		{
			out_z_ptr->m_value = std::move(z);
			out_z_ptr->normalize();
		}
		if(out_r_ptr)
			*out_r_ptr = std::move(x);
	}//*/

private:
	//explicit uint_big_t(element_t in_x): m_value(in_x ? 1 : 0, in_x) {}

	// template
	static bool add(element_t& io_x, const element_t in_y, bool in_carry)
	{
		element_t carry;
		//carry = _addcarry_u64(0, d, carry, &d);
		io_x = __builtin_addcll(io_x, in_y, in_carry, &carry);
		return carry;
	}
	static bool sub(element_t& io_x, const element_t in_y, bool in_carry)
	{
		element_t carry;
		io_x = __builtin_subcll(io_x, in_y, in_carry, &carry);
		return carry;
	}

	template<class TAllocator>
	static size_t normalize(std::vector<element_t, TAllocator>& io_x, size_t in_size /*to avoid getting a size inside the function*/)
	{
		if(!in_size)
			return 0;

		size_t n = in_size;
		if(io_x[--n])
			return in_size;

		for(; n; --n)
		{
			if(io_x[n - 1])
			{
				io_x.resize(n);
				return n;
			}
		}

		io_x.clear();
		return 0;
	}
	//void normalize() { normalize(m_value, m_value.size()); }
	void normalize(size_t in_size) { normalize(m_value, in_size); }

	template<class TAllocator, class UAllocator>
	static bool cmp_l(const std::vector<element_t, TAllocator>& in_a, const std::vector<element_t, UAllocator>& in_b)
	{
		if(unlikely(static_cast<const void*>(&in_a) == static_cast<const void*>(&in_b)))
			return false;

		size_t len = in_a.size();
		const size_t len_b = in_b.size();

		if(len != len_b)
			return (len < len_b);

		while(len--)
		{
			if(in_a[len] == in_b[len])
				continue;
			return in_a[len] < in_b[len];
		}
		return false;
	}
	template<class TAllocator, class UAllocator>
	static bool cmp_g(const std::vector<element_t, TAllocator>& in_a, const std::vector<element_t, UAllocator>& in_b) { return cmp_l(in_b, in_a); }
	template<class TAllocator, class UAllocator>
	static bool cmp_le(const std::vector<element_t, TAllocator>& in_a, const std::vector<element_t, UAllocator>& in_b) { return !cmp_g(in_a, in_b); }
	template<class TAllocator, class UAllocator>
	static bool cmp_ge(const std::vector<element_t, TAllocator>& in_a, const std::vector<element_t, UAllocator>& in_b) { return !cmp_l(in_a, in_b); }

	template<class TAllocator, class UAllocator>
	static void sub(std::vector<element_t, TAllocator>& io_a, const std::vector<element_t, UAllocator>& in_b)
	{
		size_t len_b = in_b.size();
		//if(!len_x)
		//	return *this;

		size_t len_a = io_a.size();
		if(len_a < len_b)
			throw std::overflow_error("The first argument of an unsigned negation cannot be less than the second argument");

		bool carry = false;
		size_t i;
		for(i = 0; i < len_b; ++i)
			carry = sub(io_a[i], in_b[i], carry);
		for(; i < len_a; ++i)
			carry = sub(io_a[i], 0, carry);
		if(carry)
			throw std::overflow_error("The result of an unsigned negation is negative");

		normalize(io_a, len_a);
	}

	template<class TAllocator>
	static void mul(std::vector<element_t, TAllocator>& io_x, element_t in_c)
	{
		element_ex_t x{in_c};
		element_t carry = 0;

		for(auto& d : io_x)
		{
			element_ex_t z = x * d + carry;
			d = static_cast<element_t>(z);
			carry = static_cast<element_t>(z >> m_bits_per_element);
		}
		if(carry)
			io_x.push_back(carry);
	}

	template<typename TAllocator>
	static void shl(std::vector<element_t, TAllocator>& io_x, size_t in_shift)
	{
		size_t inc_size = in_shift / m_bits_per_element;
		if(inc_size)
			io_x.insert(io_x.begin(), inc_size, 0);

		if(size_t shift = in_shift % m_bits_per_element)
		{
			element_t carry = 0;
			for(auto it = io_x.begin() + inc_size, end = io_x.end(); it != end; ++it)
			{
				element_t x = *it;
				*it = (x << shift) | carry;
				carry = x >> (m_bits_per_element - shift);
			}
			if(carry)
				io_x.push_back(carry);
		}
	}
	template<typename TAllocator>
	static void shr(std::vector<element_t, TAllocator>& io_x, size_t in_shift)
	{
		size_t len = io_x.size();
		if(size_t dec_size = in_shift / m_bits_per_element)
		{
			dec_size = std::min(dec_size, len);
			io_x.erase(io_x.begin(), io_x.begin() + dec_size);
			len -= dec_size;
		}
		if(!len)
			return;

		if(size_t shift = in_shift % m_bits_per_element)
		{
			element_t carry = 0;
			for(auto it = io_x.rbegin(), end = io_x.rend(); it != end; ++it)
			{
				element_t x = *it;
				*it = carry | (x >> shift);
				carry = x << (m_bits_per_element - shift);
			}
			if(!io_x[len - 1])
				io_x.pop_back();
		}
	}

	template<typename T>
	static unsigned msb(T in_x)
	{
		switch(sizeof(T))
		{
			case 8:
				return (sizeof(T) * 8 - 1 - __builtin_clzll(in_x));
			case 4:
				return (sizeof(T) * 8 - 1 - __builtin_clz(in_x));
			default:
				throw std::invalid_argument("MSB is not implemented for a given type");
		}
	}
	template<class TAllocator>
	static size_t msb(const std::vector<element_t, TAllocator>& in_x)
	{
		size_t size = in_x.size();
		if(!size)
			throw std::logic_error("The most significant bit is undefined - the value is zero");
		if(auto highest_element = *in_x.crbegin())
			return msb(highest_element) + (size - 1) * m_bits_per_element;
		else
			throw std::logic_error("The number is not normalized");
	}
	size_t msb() const { return msb(m_value); }

	bool get_bit(size_t in_bit) const
	{
		size_t i = in_bit / m_bits_per_element;
		if(i >= m_value.size())
			return false;
		return (m_value[i] >> (in_bit % m_bits_per_element)) & 1;
	}
	static bool get_bit_unsafe(const element_t* in_x, size_t in_size, size_t in_bit)
	{
		size_t i = in_bit / m_bits_per_element;
		return (i < in_size) ? ((in_x[i] >> (in_bit % m_bits_per_element)) & 1) : false;
	}

	uint_big_t& operator|=(element_t in_x)
	{
		if(in_x)
		{
			if(unlikely(m_value.empty()))
				m_value.push_back(in_x);
			else
				m_value[0] |= in_x;
		}

		return *this;
	}

	/*static int cmp_bits(const storage_t& in_x_value, const storage_t& in_y_value, size_t in_at_bit, size_t in_bits)
	{
		if(unlikely(!in_bits))
			throw std::invalid_argument("Number of bits to comapre must be greater than zero");
		if(unlikely(in_bits > in_at_bit + 1))
			throw std::logic_error("Too many (" + std::to_string(in_bits) + ") to compare at position " + std::to_string(in_at_bit));

		size_t bits = in_bits;
		size_t pos = in_at_bit / m_bits_per_element;
		if(unlikely(pos >= in_x_value.size()))
		{
			if(unlikely(pos >= in_y_value.size()))
				throw std::out_of_range("A starting bit is out of range of the second argument");
			return -1;
		}

		// High bits
		if(size_t high_bits = (in_at_bit + 1) % m_bits_per_element)
		{
			element_t mask = (element_t{1} << high_bits) - 1;
			element_t x = in_x_value[pos] & mask;
			element_t y = in_y_value[pos] & mask;

			size_t shift_back = (bits < high_bits) ? (high_bits - bits) : 0;
			if(shift_back)
			{
				x >>= shift_back;
				y >>= shift_back;
			}

			if(x > y)
				return 1;
			else if(x < y)
				return -1;
			else if(shift_back)
				return 0;
			
			bits -= high_bits;
			--pos;
		}

		// Middle bits
		for(size_t n = bits / m_bits_per_element; n; --n, --pos)
		{
			if(in_x_value[pos] > in_y_value[pos])
				return 1;
			else if(in_x_value[pos] < in_y_value[pos])
				return -1;
		}

		// Low bits
		if(size_t low_bits = bits % m_bits_per_element)
		{
			element_t mask = ~element_t{0} << (m_bits_per_element - low_bits);
			element_t x = in_x_value[pos] & mask;
			element_t y = in_y_value[pos] & mask;
			if(x > y)
				return 1;
			else if(x < y)
				return -1;
		}

		return 0;
	}*/

private:
	storage_t m_value;
};


// shift
template<typename U>
std::enable_if_t<std::is_class<typename std::remove_reference<U>::type>() && std::is_same<typename std::remove_reference<U>::type, uint_big_t>(), uint_big_t>
operator>>(U&& in_x, size_t in_shift)
{
	uint_big_t y(in_x);
	y >>= in_shift;
	return y;
}
template<typename U>
std::enable_if_t<std::is_class<typename std::remove_reference<U>::type>() && std::is_same<typename std::remove_reference<U>::type, uint_big_t>(), uint_big_t>
operator<<(U&& in_x, size_t in_shift)
{
	uint_big_t y(in_x);
	y <<= in_shift;
	return y;
}

// add
template<typename U, typename T>
std::enable_if_t<std::is_same<typename std::remove_reference<U>::type, uint_big_t>() && std::is_arithmetic<T>(), uint_big_t>
operator+(U&& in_x, T in_y)
{
	uint_big_t z(in_x);
	z += in_y;
	return z;
}
template<typename U, typename T>
std::enable_if_t<std::is_same<typename std::remove_reference<U>::type, uint_big_t>() && !std::is_arithmetic<T>(), uint_big_t>
operator+(U&& in_x, const T& in_y)
{
	uint_big_t z(in_x);
	z += in_y;
	return z;
}

// sub
template<typename U, typename T>
std::enable_if_t<std::is_same<typename std::remove_reference<U>::type, uint_big_t>() && std::is_arithmetic<T>(), uint_big_t>
operator-(U&& in_x, T in_y)
{
	uint_big_t z(in_x);
	z -= in_y;
	return z;
}
template<typename U, typename T>
std::enable_if_t<std::is_same<typename std::remove_reference<U>::type, uint_big_t>() && !std::is_arithmetic<T>(), uint_big_t>
operator-(U&& in_x, const T& in_y)
{
	uint_big_t z(in_x);
	z -= in_y;
	return z;
}

// mul
template<typename U, typename T>
std::enable_if_t<std::is_same<typename std::remove_reference<U>::type, uint_big_t>() && std::is_arithmetic<T>(), uint_big_t>
operator*(U&& in_x, T in_y)
{
	uint_big_t z(in_x);
	z *= in_y;
	return z;
}
template<typename U, typename T>
std::enable_if_t<std::is_same<typename std::remove_reference<U>::type, uint_big_t>() && !std::is_arithmetic<T>(), uint_big_t>
operator*(U&& in_x, const T& in_y)
{
	uint_big_t z(in_x);
	z *= in_y;
	return z;
}

// div
template<typename T>
uint_big_t operator/(const uint_big_t& in_x, T&& in_y)
{
	uint_big_t z;
	uint_big_t::div(in_x, in_y, &z, nullptr);
	return z;
}

// mod
template<typename T>
struct _decay_to_base
{
	using _base_type = std::decay_t<T>;
	using type = std::conditional_t<std::is_arithmetic<_base_type>::value, uint_big_t::element_t, _base_type>;
};

template<typename T>
typename _decay_to_base<T>::type operator%(const uint_big_t& in_x, T&& in_y)
{
	typename _decay_to_base<T>::type r;
	uint_big_t::div(in_x, in_y, nullptr, &r);
	return r;
}

/*template<typename T>
struct reference_if_not_arithmetic
{
	std::conditional<std::is_arithmetic<std::decay_t<T>>(), >
};*/
