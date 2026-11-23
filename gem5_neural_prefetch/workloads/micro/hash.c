/* hash: lookups in a chained hash table with computed keys.
 * Constructed ground truth: chain edges through the next pointer at byte
 * offset 8. The bucket slot address comes from a hash of a computed key, so
 * it is not affine in any loaded value.
 * usage: hash [entries=2097152] [buckets=1048576] [lookups=8388608] */
#include "np_common.h"

struct entry {
    uint64_t key;        /* offset 0 */
    struct entry *next;  /* offset 8 */
    uint64_t value;      /* offset 16 */
    uint64_t pad[5];     /* fills the entry to 64 bytes */
};

static inline uint64_t mix(uint64_t x)
{
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33;
    return x;
}

int main(int argc, char **argv)
{
    const uint64_t n = np_arg(argc, argv, 1, 2u << 20);
    const uint64_t buckets = np_arg(argc, argv, 2, 1u << 20);
    const uint64_t lookups = np_arg(argc, argv, 3, 8u << 20);

    struct entry *entries = np_alloc(n * sizeof(struct entry));
    struct entry **table = np_alloc(buckets * sizeof(struct entry *));
    for (uint64_t b = 0; b < buckets; b++)
        table[b] = NULL;
    for (uint64_t i = 0; i < n; i++) {
        struct entry *e = &entries[i];
        e->key = i * 2654435761ULL + 12345;
        e->value = i;
        uint64_t b = mix(e->key) % buckets;
        e->next = table[b];
        table[b] = e;
    }

    np_marker();
    uint64_t sum = 0, steps = 0;
    uint64_t probe = 1442695040888963407ULL;
    for (uint64_t q = 0; q < lookups; q++) {
        /* The key is computed from a counter-like state, never loaded. */
        uint64_t key = (np_next(&probe) % n) * 2654435761ULL + 12345;
        const struct entry *e = table[mix(key) % buckets];
        while (e != NULL && e->key != key) {
            e = e->next;
            steps++;
        }
        if (e != NULL)
            sum += e->value;
    }

    printf("hash sum %llu steps %llu\n", (unsigned long long)sum,
           (unsigned long long)steps);
    return 0;
}
