/* Shared helpers for the validation microbenchmarks. */
#ifndef NP_COMMON_H
#define NP_COMMON_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* gem5's work-begin operation: bytes 0F 04 followed by the 16-bit function
 * number 0x5a, arguments in rdi and rsi. On real hardware this is an invalid
 * opcode, so it runs only when NP_GEM5 is set in the environment. */
static void __attribute__((noinline)) np_gem5_work_begin(void)
{
    __asm__ volatile(".byte 0x0F, 0x04\n\t.word 0x5a"
                     : /* no outputs */
                     : "D"(0UL), "S"(0UL)
                     : "rax", "memory");
}

/* Call once, where the measured kernel starts. */
static inline void np_marker(void)
{
    if (getenv("NP_GEM5"))
        np_gem5_work_begin();
}

static inline uint64_t np_next(uint64_t *state)
{
    uint64_t x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    return x;
}

static inline void *np_alloc(size_t bytes)
{
    void *p = NULL;
    if (posix_memalign(&p, 4096, bytes) != 0 || p == NULL) {
        fprintf(stderr, "allocation of %zu bytes failed\n", bytes);
        exit(2);
    }
    return p;
}

static inline uint64_t np_arg(int argc, char **argv, int i, uint64_t fallback)
{
    return argc > i ? strtoull(argv[i], NULL, 0) : fallback;
}

#endif
