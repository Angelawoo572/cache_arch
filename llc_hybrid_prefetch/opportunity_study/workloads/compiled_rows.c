/* Real trace control. Unsigned integer arithmetic is defined modulo 2^64. */
typedef unsigned long long word;

__attribute__((noinline))
word compiled_rows(volatile const word *data, unsigned rows)
{
    word value = 1;
    for (unsigned row = 0; row < rows; ++row) {
        unsigned length = (row & 1u) ? 24u : 8u;
        for (unsigned column = 0; column < length; ++column) {
            value += data[(row * 32u + column) * 8u];
            for (unsigned step = 0; step < 18; ++step) {
                value ^= value >> 17;
                value *= 13;
            }
        }
    }
    return value;
}
