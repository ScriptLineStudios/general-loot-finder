#ifndef LOOT_DATA_
#define LOOT_DATA_

#include <stdbool.h>
#include <stdint.h>
#include <string.h>
#include "../../cJSON/cJSON.h"

#ifdef USE_CUDA
    #define LOOT_LIB_FUNCTION __device__ __host__
#else
    #define LOOT_LIB_FUNCTION
#endif

typedef enum {
    PROVIDE_NO_FUNCTION, PROVIDE_ENCHANT_RANDOMLY, PROVIDE_UNIFORM_ROLL
} LootProvider;

typedef enum {
    EMPTY, OBSIDIAN, FLINT, IRON_NUGGET, FLINT_AND_STEEL, FIRE_CHARGE, GOLDEN_APPLE, GOLD_NUGGET, GOLDEN_SWORD, GOLDEN_AXE, GOLDEN_HOE, GOLDEN_SHOVEL, GOLDEN_PICKAXE, GOLDEN_BOOTS, GOLDEN_CHESTPLATE, GOLDEN_HELMET, GOLDEN_LEGGINGS, GLISTERING_MELON_SLICE, GOLDEN_HORSE_ARMOR, LIGHT_WEIGHTED_PRESSURE_PLATE, GOLDEN_CARROT, CLOCK, GOLD_INGOT, BELL, ENCHANTED_GOLDEN_APPLE, GOLD_BLOCK, LODESTONE     
} ItemType;

static const char* item_type_strings[] = {
    "minecraft:empty","minecraft:obsidian","minecraft:flint","minecraft:iron_nugget",
    "minecraft:flint_and_steel",
    "minecraft:fire_charge",
    "minecraft:golden_apple",
    "minecraft:gold_nugget",
    "minecraft:golden_sword",
    "minecraft:golden_axe",
    "minecraft:golden_hoe",
    "minecraft:golden_shovel",
    "minecraft:golden_pickaxe",
    "minecraft:golden_boots",
    "minecraft:golden_chestplate",
    "minecraft:golden_helmet",
    "minecraft:golden_leggings",
    "minecraft:glistering_melon_slice",
    "minecraft:golden_horse_armor",
    "minecraft:light_weighted_pressure_plate",
    "minecraft:golden_carrot",
    "minecraft:clock",
    "minecraft:gold_ingot",
    "minecraft:bell",
    "minecraft:enchanted_golden_apple",
    "minecraft:gold_block",
    "minecraft:lodestone"
};

typedef enum {
	NO_CATEGORY, CATEGORY_HELMET, CATEGORY_CHESTPLATE, CATEGORY_LEGGINGS, CATEGORY_BOOTS, CATEGORY_SWORD, CATEGORY_PICKAXE, CATEGORY_SHOVEL, 
    CATEGORY_AXE, CATEGORY_HOE, CATEGORY_FISHING_ROD, CATEGORY_BOW, CATEGORY_CROSSBOW, CATEGORY_TRIDENT, CATEGORY_MACE, CATEGORY_BOOK
} ItemCategory;

typedef enum {
	NO_ENCHANTMENT = 0,
	PROTECTION,
	FIRE_PROTECTION,
	BLAST_PROTECTION,
	PROJECTILE_PROTECTION,
	RESPIRATION,
	AQUA_AFFINITY,
	THORNS,
	SWIFT_SNEAK,
	FEATHER_FALLING,
	DEPTH_STRIDER,
	FROST_WALKER,
	SOUL_SPEED,
	SHARPNESS,
	SMITE,
	BANE_OF_ARTHROPODS,
	KNOCKBACK,
	FIRE_ASPECT,
	LOOTING,
	SWEEPING_EDGE,
	EFFICIENCY,
	SILK_TOUCH,
	FORTUNE,
	LUCK_OF_THE_SEA,
	LURE,
	POWER,
	PUNCH,
	FLAME,
	INFINITY_ENCHANTMENT,
	QUICK_CHARGE,
	MULTISHOT,
	PIERCING,
	IMPALING,
	RIPTIDE,
	LOYALTY,
	CHANNELING,
	DENSITY,
	BREACH,
	WIND_BURST,
	MENDING,
	UNBREAKING,
	CURSE_OF_VANISHING,
	CURSE_OF_BINDING
} EnchantType;

static const char* enchant_type_strings[] = {
    "minecraft:no_enchantment",
    "minecraft:protection",
    "minecraft:fire_protection",
    "minecraft:blast_protection",
    "minecraft:projectile_protection",
    "minecraft:respiration",
    "minecraft:aqua_affinity",
    "minecraft:thorns",
    "minecraft:swift_sneak",
    "minecraft:feather_falling",
    "minecraft:depth_strider",
    "minecraft:frost_walker",
    "minecraft:soul_speed",
    "minecraft:sharpness",
    "minecraft:smite",
    "minecraft:bane_of_arthropods",
    "minecraft:knockback",
    "minecraft:fire_aspect",
    "minecraft:looting",
    "minecraft:sweeping_edge",
    "minecraft:efficiency",
    "minecraft:silk_touch",
    "minecraft:fortune",
    "minecraft:luck_of_the_sea",
    "minecraft:lure",
    "minecraft:power",
    "minecraft:punch",
    "minecraft:flame",
    "minecraft:infinity",
    "minecraft:quick_charge",
    "minecraft:multishot",
    "minecraft:piercing",
    "minecraft:impaling",
    "minecraft:riptide",
    "minecraft:loyalty",
    "minecraft:channeling",
    "minecraft:density",
    "minecraft:breach",
    "minecraft:wind_burst",
    "minecraft:mending",
    "minecraft:unbreaking",
    "minecraft:vanishing_curse",
    "minecraft:binding_curse"
};

typedef struct {
    EnchantType tp;
    short level;
} Enchant;

typedef struct {
    ItemType tp;
    short amount;
    Enchant enchant; // FOR NOW, we will only be supporting loot tables where Items are given one item, this will change in the future!
} Item;

typedef struct {
    Item *items;
    size_t size;
} Items;

ItemType get_item_from_name(const char *name);
LOOT_LIB_FUNCTION ItemCategory get_item_category(ItemType tp);
LOOT_LIB_FUNCTION bool enchantment_can_apply(ItemCategory item, EnchantType enchantment);
LOOT_LIB_FUNCTION int enchantment_max_level(EnchantType enchantment);
LOOT_LIB_FUNCTION void get_applicable_enchantments(ItemCategory category, EnchantType *enchants, size_t *size);

#endif //LOOT_DATA_
