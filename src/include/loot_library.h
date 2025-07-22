#ifndef LOOT_LIBRARY_
#define LOOT_LIBRARY_

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <assert.h>

#ifdef USE_CUDA
    #define LOOT_LIB_FUNCTION __device__ __host__
#else
    #define LOOT_LIB_FUNCTION
#endif

#include "loot_data.h"
#include "rng.h"

typedef struct {
    short min_rolls;
    short max_rolls;    
    short size; // number of items in the table
    short total_weight;
    short items[40]; // this is a map from an index to an item 
    short providers[40]; // don't store the function pointers themselves to make it easier to run with CUDA 
    short counts[80];
    short data[512]; // this is the precomputed loot table data
} LootPool;

typedef Item (*loot_provider)(const LootPool *loot_pool, int, uint64_t *);

typedef struct {
    short size;
    LootPool *pools;
} LootTable;

extern "C" { // prevent nvcc name mangling
    LootPool loot_pool_new();
    LOOT_LIB_FUNCTION Item provide_loot_no_function(const LootPool *loot_pool, int index, uint64_t *rng);
    LOOT_LIB_FUNCTION Item provide_loot_uniform_roll(const LootPool *loot_pool, int index, uint64_t *rng);
    LOOT_LIB_FUNCTION Item provide_loot_enchant_randomly(const LootPool *loot_pool, int index, uint64_t *rng);
    void loot_pool_add_new_item(LootPool *pool, ItemType tp, int weight, int min, int max, Item(*provider)(void *, int, uint64_t *));
    void loot_table_add_pool(LootTable *table, LootPool pool);
    LootTable parse_table_from_json(const char *filepath);
    LOOT_LIB_FUNCTION void get_loot_from_table(LootTable *table, uint64_t loot_seed, Item *items, size_t *num_items);
    LOOT_LIB_FUNCTION void get_loot_from_pool(const LootPool *pool, Item *items, size_t *num_items, uint64_t *rng);
    size_t save_table(LootTable *table, const char *filepath);
    LootTable read_table(const char *filepath);
}
// #endif

#endif //LOOT_LIBRARY_