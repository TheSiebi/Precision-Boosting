#ifndef TSC_X86_H
#define TSC_X86_H

#include <stdint.h>

/* ==================== GNU C and possibly other UNIX compilers ===================== */
#if !defined(WIN32) || defined(__GNUC__)

	#if defined(__GNUC__) || defined(__linux__)
		#define VOLATILE __volatile__
		#define ASM __asm__
	#else
		/* if we're neither compiling with gcc or under linux, we can hope
		 * the following lines work, they probably won't */
		#define ASM asm
		#define VOLATILE
	#endif

	#define myInt64 unsigned long long
	#define INT32 unsigned int

/* ======================== WIN32 ======================= */
#else

	#define myInt64 signed __int64
	#define INT32 unsigned __int32

#endif

/* This is the RDTSC timer.
 * RDTSC is an instruction on several Intel and compatible CPUs that Reads the
 * Time Stamp Counter. The Intel manuals contain more information.
 */


#define COUNTER_LO(a) ((a).int32.lo)
#define COUNTER_HI(a) ((a).int32.hi)
#define COUNTER_VAL(a) ((a).int64)

#define COUNTER(a) \
	((unsigned long long)COUNTER_VAL(a))

#define COUNTER_DIFF(a,b) \
	(COUNTER(a)-COUNTER(b))

union tsc_counter
{
    uint64_t int64;
    struct {uint32_t lo, hi;} int32;
};

/* ==================== GNU C and possibly other UNIX compilers ===================== */
#if !defined(WIN32) || defined(__GNUC__)
  #define RDTSC(cpu_c) \
	  ASM VOLATILE ("rdtsc" : "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))
  #define RDTSCP(cpu_c) \
	  ASM VOLATILE ("rdtscp" : "=a" ((cpu_c).int32.lo), "=d"((cpu_c).int32.hi))
  #define CPUID() \
		ASM VOLATILE ("cpuid" : : "a" (0) : "bx", "cx", "dx" )
/* ======================== WIN32 ======================= */
#else
	#define RDTSC(cpu_c)   \
	{       __asm rdtsc    \
			__asm mov (cpu_c).int32.lo,eax  \
			__asm mov (cpu_c).int32.hi,edx  \
	}
	#define RDTSCP(cpu_c)   \
	{       __asm rdtscp    \
			__asm mov (cpu_c).int32.lo,eax  \
			__asm mov (cpu_c).int32.hi,edx  \
	}

	#define CPUID() \
	{ \
		__asm mov eax, 0 \
		__asm cpuid \
	}
#endif

static inline uint64_t getRDTSC()
{
    union tsc_counter counter;
    RDTSC(counter);
    return counter.int64;
}

static inline uint64_t getRDTSCP()
{
    union tsc_counter counter;
    RDTSC(counter);
    return counter.int64;
}

#endif //TSC_X86_H
