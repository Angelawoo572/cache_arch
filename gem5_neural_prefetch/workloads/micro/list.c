/* list: a singly linked list laid out in shuffled memory order.
 * Constructed ground truth: one self edge, next pointer at byte offset 16.
 * usage: list [nodes=1048576] [passes=64] */
#include "np_common.h"

struct node {
    uint64_t key;       /* offset 0 */
    uint64_t pad0;      /* offset 8 */
    struct node *next;  /* offset 16 */
    uint64_t pad1[5];   /* fills the node to 64 bytes */
};

int main(int argc, char **argv)
{
    const uint64_t n = np_arg(argc, argv, 1, 1u << 20);
    const uint64_t passes = np_arg(argc, argv, 2, 64);

    struct node *nodes = np_alloc(n * sizeof(struct node));
    uint64_t *order = np_alloc(n * sizeof(uint64_t));
    uint64_t seed = 88172645463325252ULL;
    for (uint64_t i = 0; i < n; i++)
        order[i] = i;
    for (uint64_t i = n - 1; i > 0; i--) {
        uint64_t j = np_next(&seed) % (i + 1);
        uint64_t t = order[i];
        order[i] = order[j];
        order[j] = t;
    }
    for (uint64_t i = 0; i < n; i++) {
        struct node *cur = &nodes[order[i]];
        cur->key = np_next(&seed);
        cur->pad0 = 0;
        cur->next = i + 1 < n ? &nodes[order[i + 1]] : NULL;
    }
    struct node *head = &nodes[order[0]];

    np_marker();
    uint64_t sum = 0;
    for (uint64_t p = 0; p < passes; p++)
        for (struct node *cur = head; cur != NULL; cur = cur->next)
            sum += cur->key;

    printf("list sum %llu\n", (unsigned long long)sum);
    return 0;
}
