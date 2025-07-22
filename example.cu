#include <stdio.h>
#include <sys/time.h>
#include "src/include/loot_library.h"

__host__ __device__ void print_items(Item *items, size_t num_items) {
    for (size_t i = 0; i < num_items; i++) {
        if (items[i].tp != EMPTY) {
            printf("%d x %d\n", items[i].tp, items[i].amount);
            if (items[i].enchant.level != 0) {
                printf("    %d %d\n", items[i].enchant.tp, items[i].enchant.level);
            }
        }
    }
}

__device__ __managed__ LootTable table;

// __constant__ ItemType[64] items;

// __constant__ int min_rolls;
// __constant__ int max_rolls;
// __constant__ size_t size;
// __constant__ size_t total_weight;
// __constant__ ItemType items[64];
// __constant__ int counts[64];
// __constant__ int data[512];
// __constant__ LootProvider providers[64];

//IDEA: a big ass array of constant memory, first element of said memory
//
/*
    first element of the memory tells us how many pools there are, (memory + 1) can
    be reinterpreded as the first pool! a pool just a bunch of shorts
*/

// __constant__ short test[10];
__constant__ short memory[20000];

__global__ void kernel(uint64_t s) {
    uint64_t loot_seed = blockDim.x * blockIdx.x + threadIdx.x + s;
    __shared__ short shared[1353];

    int tid = threadIdx.x;
    int blockSize = blockDim.x;

    for (int i = tid; i < 1353; i += blockSize) {
        shared[i] = memory[i];
    }
    __syncthreads();

    uint64_t rng;
    set_seed(&rng, loot_seed);

    Item items[24];
    size_t num_items = 0;

    #pragma unroll
    for (short i = 0; i < shared[0]; i++) {
        const LootPool *reconstruced_pool = (const LootPool *)(shared + 1 + (sizeof(LootPool) / sizeof(short)) * i);
        get_loot_from_pool(reconstruced_pool, items, &num_items, &rng);
    }

    int gapples = 0;
    #pragma unroll
    for (size_t i = 0; i < num_items; i++) {
        if (items[i].tp == ENCHANTED_GOLDEN_APPLE) {
            gapples += items[i].amount;
        }
    }

    if (gapples >= 6) {
        printf("Found loot seed: %lu\n", loot_seed);
    }
}

int main() {
    int blocks = 262144 * 2;
    int threads = 512;
    int iters = 10;

    table = parse_table_from_json("chests/ruined_portal.json");
    cudaMemcpyToSymbol(memory, &table.size, sizeof(short), 0);
    cudaMemcpyToSymbol(memory, &table.pools[0], sizeof(LootPool), sizeof(short));
    cudaMemcpyToSymbol(memory, &table.pools[1], sizeof(LootPool), sizeof(short) + sizeof(LootPool));

    short size = 1 + 2 * (sizeof(LootPool) / sizeof(short));

    struct timeval start, end;
    gettimeofday(&start, NULL);
    for (uint64_t s = 0; s < iters; s++) {
        kernel<<<blocks, threads>>>(blocks * threads * s);
    }
    cudaDeviceSynchronize();
    gettimeofday(&end, NULL);

    uint64_t seeds_searched = (uint64_t)blocks * (uint64_t)threads * (uint64_t)iters;
    double time_taken = end.tv_sec + end.tv_usec / 1e6 - start.tv_sec + start.tv_usec / 1e6;
    printf("Searched %lu seeds in %lf seconds\n", seeds_searched, time_taken);
    printf("    %lf/s\n", seeds_searched / time_taken);

    return 0;
}

