/* stride: one array read at a fixed step. No load depends on another load.
 * usage: stride [elements=8388608] [step=8] [passes=64] */
#include "np_common.h"

int main(int argc, char **argv)
{
    const uint64_t n = np_arg(argc, argv, 1, 8u << 20);
    const uint64_t step = np_arg(argc, argv, 2, 8);
    const uint64_t passes = np_arg(argc, argv, 3, 64);

    uint64_t *a = np_alloc(n * sizeof(uint64_t));
    uint64_t seed = 88172645463325252ULL;
    for (uint64_t i = 0; i < n; i++)
        a[i] = np_next(&seed);

    np_marker();
    uint64_t sum = 0;
    for (uint64_t p = 0; p < passes; p++)
        for (uint64_t i = 0; i < n; i += step)
            sum += a[i];

    printf("stride sum %llu\n", (unsigned long long)sum);
    return 0;
}
