/* csr: a gather over a random graph in compressed sparse row form.
 * Constructed ground truth: values[neighbours[e]] is an index edge with
 * shift 3 and the values array as base, applied over the range
 * offsets[v] .. offsets[v + 1] of 4-byte neighbour ids.
 * usage: csr [vertices=4194304] [degree=4] [passes=16] */
#include "np_common.h"

int main(int argc, char **argv)
{
    const uint64_t vertices = np_arg(argc, argv, 1, 4u << 20);
    const uint64_t degree = np_arg(argc, argv, 2, 4);
    const uint64_t passes = np_arg(argc, argv, 3, 16);
    const uint64_t edges = vertices * degree;

    uint64_t *offsets = np_alloc((vertices + 1) * sizeof(uint64_t));
    uint32_t *neighbours = np_alloc(edges * sizeof(uint32_t));
    uint64_t *values = np_alloc(vertices * sizeof(uint64_t));
    uint64_t seed = 88172645463325252ULL;

    /* Degrees vary between 0 and 2 * degree so that list lengths differ. */
    uint64_t next = 0;
    for (uint64_t v = 0; v < vertices; v++) {
        offsets[v] = next;
        uint64_t d = np_next(&seed) % (2 * degree + 1);
        if (next + d > edges)
            d = edges - next;
        for (uint64_t k = 0; k < d; k++)
            neighbours[next++] = (uint32_t)(np_next(&seed) % vertices);
        values[v] = np_next(&seed) & 0xffff;
    }
    offsets[vertices] = next;

    np_marker();
    uint64_t sum = 0;
    for (uint64_t p = 0; p < passes; p++)
        for (uint64_t v = 0; v < vertices; v++)
            for (uint64_t e = offsets[v]; e < offsets[v + 1]; e++)
                sum += values[neighbours[e]];

    printf("csr sum %llu edges %llu\n", (unsigned long long)sum,
           (unsigned long long)next);
    return 0;
}
