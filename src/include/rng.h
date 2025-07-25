LOOT_LIB_FUNCTION static inline void set_seed(uint64_t *seed, uint64_t value)
{
    *seed = (value ^ 0x5deece66d) & ((1ULL << 48) - 1);
}

LOOT_LIB_FUNCTION static inline int next(uint64_t *seed, const int bits)
{
    *seed = (*seed * 0x5deece66d + 0xb) & ((1ULL << 48) - 1);
    return (int) ((int64_t)*seed >> (48 - bits));
}

LOOT_LIB_FUNCTION static inline int next_int(uint64_t *seed, const int n)
{
    int bits, val;
    const int m = n - 1;

    if ((m & n) == 0) {
        uint64_t x = n * (uint64_t)next(seed, 31);
        return (int) ((int64_t) x >> 31);
    }

    do {
        bits = next(seed, 31);
        val = bits % n;
    }
    while (bits - val + m < 0);
    return val;
}

LOOT_LIB_FUNCTION static inline uint64_t next_long(uint64_t *seed)
{
    return ((uint64_t) next(seed, 32) << 32) + next(seed, 32);
}

LOOT_LIB_FUNCTION static inline float next_float(uint64_t *seed)
{
    return next(seed, 24) / (float) (1 << 24);
}

LOOT_LIB_FUNCTION static inline double next_double(uint64_t *seed)
{
    uint64_t x = (uint64_t)next(seed, 26);
    x <<= 27;
    x += next(seed, 27);
    return (int64_t) x / (double) (1ULL << 53);
}

LOOT_LIB_FUNCTION static inline int next_int_bounded(uint64_t *seed, int min, int max) {
    if (min >= max) {
        return min;
    }
    return next_int(seed, max - min + 1) + min;
}
