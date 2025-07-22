#include <loot_library.h>

LootPool loot_pool_new() {
    LootPool pool = {0};
    return pool;
}

LOOT_LIB_FUNCTION Item provide_loot_no_function(const LootPool *loot_pool, int index, uint64_t *rng) {
    (void)rng;
    LootPool *pool = (LootPool *)loot_pool; 
    return (Item){.tp=(ItemType)pool->items[index], .amount=1, .enchant=(Enchant){.tp=NO_ENCHANTMENT, .level=0}};
}

LOOT_LIB_FUNCTION Item provide_loot_uniform_roll(const LootPool *loot_pool, int index, uint64_t *rng) {
    // lets say the first item in the pool is
    LootPool *pool = (LootPool *)loot_pool; 
    int min = pool->counts[index * 2 + 0];
    int max = pool->counts[index * 2 + 1];
    return (Item){.tp=(ItemType)pool->items[index], .amount=(short)next_int_bounded(rng, min, max), .enchant=(Enchant){.tp=NO_ENCHANTMENT, .level=0}};
}

LOOT_LIB_FUNCTION Item provide_loot_enchant_randomly(const LootPool *loot_pool, int index, uint64_t *rng) {
    LootPool *pool = (LootPool *)loot_pool;
    EnchantType enchantments[64];
    size_t size = 0;
    get_applicable_enchantments(get_item_category((ItemType)pool->items[index]), enchantments, &size);

    int enchant_num = next_int(rng, size);
    EnchantType enchant = enchantments[enchant_num]; 
    
    int level = 1;
    int min_level = 1;
    int max_level = enchantment_max_level(enchant);
    if (min_level != max_level) {
        level = next_int(rng, max_level) + 1; 
    }

    return (Item){.tp=(ItemType)pool->items[index], .amount=1, .enchant=(Enchant){.tp=enchant, .level=(short)level}};   
}

LOOT_LIB_FUNCTION LootProvider get_provider(loot_provider provider) {
    if (provider == &provide_loot_no_function) return PROVIDE_NO_FUNCTION;
    if (provider == &provide_loot_uniform_roll) return PROVIDE_UNIFORM_ROLL;
    if (provider == &provide_loot_enchant_randomly) return PROVIDE_ENCHANT_RANDOMLY;
    assert(false && "unknown provider");
}

LOOT_LIB_FUNCTION loot_provider get_provider_function(LootProvider provider) {
    switch (provider) {
        case PROVIDE_NO_FUNCTION:
            return provide_loot_no_function;
        case PROVIDE_ENCHANT_RANDOMLY:
            return provide_loot_enchant_randomly;
        case PROVIDE_UNIFORM_ROLL:
            return provide_loot_uniform_roll;
        default:
            assert(false && "unknown provider");
    }
}

void loot_pool_add_new_item(LootPool *pool, ItemType tp, int weight, int min, int max, loot_provider provider) {
    // pool->data = (int *)realloc(pool->data,           sizeof(int) * (pool->total_weight + weight));
    // pool->counts = (int *)realloc(pool->counts,       sizeof(int) * (pool->size * 2 + 2));
    // pool->providers = (LootProvider *)realloc(pool->providers, sizeof(loot_provider) * (pool->size + 1));
    // pool->items = (ItemType *)realloc(pool->items,         sizeof(ItemType) * (pool->size + 1));
    for (int w = 0; w < weight; w++) {
        pool->data[pool->total_weight + w] = pool->size; 
    }
    pool->total_weight += weight;
    pool->counts[pool->size * 2 + 0] = min;
    pool->counts[pool->size * 2 + 1] = max;
    pool->items[pool->size] = tp;
    pool->providers[pool->size] = get_provider(provider);
    pool->size++;
}

void loot_table_add_pool(LootTable *table, LootPool pool) {
    // printf("%p\n", table->pools);
    // table->pools = (LootPool *)realloc(table->pools, sizeof(LootPool) * (table->size + 1));
    table->pools[table->size] = pool;
    table->size++;
} 

LootTable loot_table_new() {
    LootTable table = {0};
    cudaMallocManaged(&table.pools, sizeof(LootPool) * 10);
    return table;
}

LootTable parse_table_from_json(const char *filepath) {
    FILE *file = fopen(filepath, "r");
    if (file == NULL) {
        printf("Failed to open file!\n");
        // fprintf(stderr, "Failed to open file %s!\n", filepath);
        exit(1);
    }

    fseek(file, 0, SEEK_END);
    size_t file_size = ftell(file);
    fseek(file, 0, SEEK_SET); 
    char *buffer = (char *)malloc(file_size + 1);
    fread(buffer, file_size, 1, file);

    LootTable table = loot_table_new();

    cJSON *json = cJSON_Parse(buffer);
    cJSON *pools = cJSON_GetObjectItemCaseSensitive(json, "pools");
    cJSON *pool;
    cJSON *entry;
    cJSON_ArrayForEach(pool, pools) {
        LootPool loot_pool = loot_pool_new();
        cJSON *rolls = cJSON_GetObjectItemCaseSensitive(pool, "rolls");
        if (cJSON_IsNumber(rolls)) {
            loot_pool.max_rolls = rolls->valueint;
            loot_pool.min_rolls = rolls->valueint;
        }
        else {
            loot_pool.max_rolls = cJSON_GetObjectItemCaseSensitive(rolls, "max")->valueint;
            loot_pool.min_rolls = cJSON_GetObjectItemCaseSensitive(rolls, "min")->valueint;
        }

        cJSON *entries = cJSON_GetObjectItemCaseSensitive(pool, "entries");
        cJSON_ArrayForEach(entry, entries) {
            const char *type = cJSON_GetObjectItemCaseSensitive(entry, "type")->valuestring;
            if (strcmp(type, "minecraft:item") == 0) {
                const char *name = cJSON_GetObjectItemCaseSensitive(entry, "name")->valuestring;
                int weight = 1;
                if (cJSON_HasObjectItem(entry, "weight")) {
                    weight = cJSON_GetObjectItemCaseSensitive(entry, "weight")->valueint;
                }
                const char *function_name = "minecraft:none";
                int min = 1;
                int max = 1;
                if (cJSON_HasObjectItem(entry, "functions")) {
                    cJSON *functions = cJSON_GetObjectItemCaseSensitive(entry, "functions");
                    cJSON *function = cJSON_GetArrayItem(functions, 0);
                    function_name = cJSON_GetObjectItemCaseSensitive(function, "function")->valuestring;
                    if (cJSON_HasObjectItem(function, "count")) {
                        cJSON *count = cJSON_GetObjectItemCaseSensitive(function, "count");
                        min = cJSON_GetObjectItemCaseSensitive(count, "min")->valueint;
                        max = cJSON_GetObjectItemCaseSensitive(count, "max")->valueint;
                    }
                }
                ItemType tp = get_item_from_name(name);

                if (strcmp("minecraft:set_count", function_name) == 0) {
                    loot_pool_add_new_item(&loot_pool, tp, weight, min, max, provide_loot_uniform_roll);
                }
                else if (strcmp("minecraft:enchant_randomly", function_name) == 0) {
                    loot_pool_add_new_item(&loot_pool, tp, weight, min, max, provide_loot_enchant_randomly);
                }
                else {
                    loot_pool_add_new_item(&loot_pool, tp, weight, min, max, provide_loot_no_function);
                }
            }
            else {
                loot_pool_add_new_item(&loot_pool, EMPTY, 1, 1, 1, provide_loot_no_function);
            }
        }
        loot_table_add_pool(&table, loot_pool);
    }

    free(buffer);
    fclose(file);
    cJSON_Delete(json);

    return table;
}

__constant__ Item (*provider_functions[])(const LootPool*, int, uint64_t*) = {
    provide_loot_no_function,
    provide_loot_enchant_randomly,
    provide_loot_uniform_roll
};

LOOT_LIB_FUNCTION void get_loot_from_pool(const LootPool *pool, Item *items, size_t *num_items, uint64_t *rng) {
    int rolls = pool->min_rolls;
    if (pool->min_rolls != pool->max_rolls) {
        rolls = next_int_bounded(rng, pool->min_rolls, pool->max_rolls);
    }

    for (int r = 0; r < rolls; r++) {
        int index = pool->data[next_int(rng, pool->total_weight)];
        switch ((LootProvider)pool->providers[index]) {
            case PROVIDE_UNIFORM_ROLL:
                items[*num_items] = provide_loot_uniform_roll(pool, index, rng);
                break;      
            case PROVIDE_ENCHANT_RANDOMLY:
                items[*num_items] = provide_loot_uniform_roll(pool, index, rng);
                break;      
            default:
                items[*num_items] = provide_loot_no_function(pool, index, rng);
                break;
        }
        (*num_items)++;
    }

}

LOOT_LIB_FUNCTION void get_loot_from_table(LootTable *table, uint64_t loot_seed, Item *items, size_t *num_items) {
    uint64_t rng;
    set_seed(&rng, loot_seed);

    for (size_t i = 0; i < table->size; i++) {
        LootPool *pool = &table->pools[i];
        int rolls = pool->min_rolls;
        if (pool->min_rolls != pool->max_rolls) {
            rolls = next_int_bounded(&rng, pool->min_rolls, pool->max_rolls);
            for (int r = 0; r < rolls; r++) {
                int index = pool->data[next_int(&rng, pool->total_weight)];
                loot_provider p = get_provider_function((LootProvider)pool->providers[index]);
                Item item = p(pool, index, &rng);
                items[*num_items] = item;
                (*num_items)++;
            }
        } 
        else {
            int index = pool->data[next_int(&rng, pool->total_weight)];
            loot_provider p = get_provider_function((LootProvider)pool->providers[index]);
            Item item = p(pool, index, &rng);
            items[*num_items] = item;
            (*num_items)++;
        }
    }
}

// typedef struct {
//     int min_rolls;
//     int max_rolls;    
//     ItemType *items; // this is a map from an index to an item 
//     int *data; // this is the precomputed loot table data
//     int *counts;
//     size_t size; // number of items in the table
//     size_t total_weight;
//     Item (**providers) (void *, int, uint64_t *); //expect this signature to change! (I think it's final now :D)
// } LootPool;


size_t save_pool(LootPool *pool, FILE *file) {
    size_t size = 0;
    size += fwrite(&pool->min_rolls, sizeof(int), 1, file);
    size += fwrite(&pool->max_rolls, sizeof(int), 1, file);
    size += fwrite(&pool->size, sizeof(size_t), 1, file);
    size += fwrite(&pool->total_weight, sizeof(size_t), 1, file);
    for (size_t i = 0; i < pool->size; i++) {
        size += fwrite(&pool->items[i], sizeof(ItemType), 1, file);
        size += fwrite(&pool->counts[i * 2 + 0], sizeof(int), 1, file);
        size += fwrite(&pool->items[i * 2 + 1], sizeof(int), 1, file);
        size += fwrite(&pool->providers[i], sizeof(LootProvider), 1, file);
    }
    return size;
}   

size_t save_table(LootTable *table, const char *filepath) {
    FILE *file = fopen(filepath, "wb");
    uint64_t magic = 0xDEADC0DE;
    size_t size = 0;
    fwrite(&magic, sizeof(uint64_t), 1, file);
    size += fwrite(&table->size, sizeof(size_t), 1, file);
    for (size_t i = 0; i < table->size; i++) {
        LootPool *p = &table->pools[i]; 
        size += save_pool(p, file);
    }
    fclose(file);
    return size;
}

LootTable read_table(const char *filepath) {
    assert(false && "TODO: Reading table from file");
    // LootTable table;
    // FILE *file = fopen(filepath, "rb");
    // uint64_t magic;
    // fread(&magic, sizeof(uint64_t), 1, file);
    // assert(magic == 0xDEADC0DE); 
    // fread(&table.size, sizeof(size_t), 1, file);
    // fclose(file);   
}