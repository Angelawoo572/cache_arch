/* tree: random lookups in an unbalanced binary search tree.
 * Constructed ground truth: two self edges, left pointer at byte offset 8 and
 * right pointer at byte offset 16.
 * usage: tree [nodes=1048576] [lookups=4194304] */
#include "np_common.h"

struct node {
    uint64_t key;        /* offset 0 */
    struct node *left;   /* offset 8 */
    struct node *right;  /* offset 16 */
    uint64_t pad[5];     /* fills the node to 64 bytes */
};

int main(int argc, char **argv)
{
    const uint64_t n = np_arg(argc, argv, 1, 1u << 20);
    const uint64_t lookups = np_arg(argc, argv, 2, 4u << 20);

    struct node *nodes = np_alloc(n * sizeof(struct node));
    uint64_t seed = 88172645463325252ULL;
    struct node *root = NULL;
    for (uint64_t i = 0; i < n; i++) {
        struct node *fresh = &nodes[i];
        fresh->key = np_next(&seed);
        fresh->left = fresh->right = NULL;
        struct node **link = &root;
        while (*link != NULL)
            link = fresh->key < (*link)->key ? &(*link)->left
                                             : &(*link)->right;
        *link = fresh;
    }

    np_marker();
    uint64_t found = 0, depth = 0;
    uint64_t probe = 1442695040888963407ULL;
    for (uint64_t q = 0; q < lookups; q++) {
        /* Half the probes are keys that exist, half are random. */
        uint64_t key = (q & 1) ? nodes[np_next(&probe) % n].key
                               : np_next(&probe);
        const struct node *cur = root;
        while (cur != NULL && cur->key != key) {
            cur = key < cur->key ? cur->left : cur->right;
            depth++;
        }
        found += cur != NULL;
    }

    printf("tree found %llu depth %llu\n", (unsigned long long)found,
           (unsigned long long)depth);
    return 0;
}
