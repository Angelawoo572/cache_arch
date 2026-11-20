#include <stdint.h>

/* Start with queue[0] visited. Stop after a fixed number of examined edges. */
uint32_t bfs(const uint32_t *offset, const uint32_t *neighbor, uint8_t *seen,
             uint32_t *queue, uint32_t edge_budget) {
    uint32_t head = 0, tail = 1;
    while (head < tail) {
        uint32_t v = queue[head++];
        for (uint32_t i = offset[v]; i < offset[v + 1]; ++i) {
            uint32_t next = neighbor[i];
            if (!seen[next]) {
                seen[next] = 1;
                queue[tail++] = next;
            }
            if (--edge_budget == 0)
                return tail;
        }
    }
    return tail;
}
