#include <loot_data.h>

#define LENGTH(x) sizeof(x) / sizeof((x)[0])

ItemType get_item_from_name(const char *name) {
    for (size_t i = 0; i < LENGTH(item_type_strings); i++) {
        if (strcmp(name, item_type_strings[i]) == 0) {
            return (ItemType)(i); 
        }
    }
    return EMPTY;
}

LOOT_LIB_FUNCTION ItemCategory get_item_category(ItemType tp) {
    // const char *item_name = item_type_strings[tp];
	// if (strstr(item_name, "_pickaxe") != NULL) return CATEGORY_PICKAXE;
	// if (strstr(item_name, "_axe") != NULL) return CATEGORY_AXE;
	// if (strstr(item_name, "_shovel") != NULL) return CATEGORY_SHOVEL;
	// if (strstr(item_name, "_hoe") != NULL) return CATEGORY_HOE;
	// if (strstr(item_name, "_sword") != NULL) return CATEGORY_SWORD;
	// if (strstr(item_name, "_helmet") != NULL) return CATEGORY_HELMET;
	// if (strstr(item_name, "_chestplate") != NULL) return CATEGORY_CHESTPLATE;
	// if (strstr(item_name, "_leggings") != NULL) return CATEGORY_LEGGINGS;
	// if (strstr(item_name, "_boots") != NULL) return CATEGORY_BOOTS;

	// if (strcmp(item_name, "fishing_rod") == 0) return CATEGORY_FISHING_ROD;
	// if (strcmp(item_name, "crossbow") == 0) return CATEGORY_CROSSBOW;
	// if (strcmp(item_name, "trident") == 0) return CATEGORY_TRIDENT;
	// if (strcmp(item_name, "bow") == 0) return CATEGORY_BOW;
	// if (strcmp(item_name, "book") == 0) return CATEGORY_BOOK;
	// if (strcmp(item_name, "mace") == 0) return CATEGORY_MACE;
	switch (tp) { // this function is massively imcomplete! but it should get the job done for now :)
		case GOLDEN_PICKAXE:
			return CATEGORY_PICKAXE;
		case GOLDEN_AXE:
			return CATEGORY_AXE;
		case GOLDEN_SHOVEL:
			return CATEGORY_SHOVEL;
		case GOLDEN_HOE:
			return CATEGORY_HOE;
		case GOLDEN_CHESTPLATE:
			return CATEGORY_CHESTPLATE;
		case GOLDEN_HELMET:
			return CATEGORY_HELMET;
		case GOLDEN_LEGGINGS:
			return CATEGORY_LEGGINGS;
		case GOLDEN_BOOTS:
			return CATEGORY_BOOTS;
	}

    return NO_CATEGORY;
}

LOOT_LIB_FUNCTION bool enchantment_can_apply(ItemCategory item, EnchantType enchantment) {
    bool use_overrides = true;
	if (enchantment == NO_ENCHANTMENT) return 0;
	if (item == CATEGORY_BOOK) return 1; // the wildcard

	switch (enchantment)
	{
    case NO_ENCHANTMENT:
	case CURSE_OF_VANISHING:
	case UNBREAKING:
	case MENDING:
		return 1;

	case THORNS:
		return item == CATEGORY_CHESTPLATE || (use_overrides == 1 && (item == CATEGORY_LEGGINGS || item == CATEGORY_BOOTS || item == CATEGORY_HELMET));
	case CURSE_OF_BINDING:
	case PROTECTION:
	case FIRE_PROTECTION:
	case BLAST_PROTECTION:
	case PROJECTILE_PROTECTION:
		return item == CATEGORY_CHESTPLATE || item == CATEGORY_LEGGINGS || item == CATEGORY_BOOTS || item == CATEGORY_HELMET;

	case RESPIRATION:
	case AQUA_AFFINITY:
		return item == CATEGORY_HELMET;

	case FEATHER_FALLING:
	case DEPTH_STRIDER:
	case FROST_WALKER:
	case SOUL_SPEED:
		return item == CATEGORY_BOOTS;

	case SWIFT_SNEAK:
		return item == CATEGORY_LEGGINGS;

	case SHARPNESS:
	case SMITE:
	case BANE_OF_ARTHROPODS:
		return item == CATEGORY_SWORD || (use_overrides == 1 && item == CATEGORY_AXE);
	case KNOCKBACK:
	case FIRE_ASPECT:
	case LOOTING:
	case SWEEPING_EDGE:
		return item == CATEGORY_SWORD;

	case EFFICIENCY:
	case SILK_TOUCH:
	case FORTUNE:
		return item == CATEGORY_PICKAXE || item == CATEGORY_SHOVEL || item == CATEGORY_AXE || item == CATEGORY_HOE;

	case POWER:
	case PUNCH:
	case FLAME:
	case INFINITY_ENCHANTMENT:
		return item == CATEGORY_BOW;

	case MULTISHOT:
	case QUICK_CHARGE:
	case PIERCING:
		return item == CATEGORY_CROSSBOW;

	case LUCK_OF_THE_SEA:
	case LURE:
		return item == CATEGORY_FISHING_ROD;

	case IMPALING:
	case RIPTIDE:
	case LOYALTY:
	case CHANNELING:
		return item == CATEGORY_TRIDENT;

	case DENSITY:
	case BREACH:
	case WIND_BURST:
		return item == CATEGORY_MACE;
	}

	return 0;
}

LOOT_LIB_FUNCTION int enchantment_max_level(EnchantType enchantment) {
	static const int MAX_LEVEL[] = {
		0, // no_enchantment
		4, 4, 4, 4, 3, 1, 3, 3, 4, 3, 2, 3, // armor
		5, 5, 5, 2, 2, 3, 3, // swords
		5, 1, 3, // tools + unbreaking
		3, 3, // fishing rods
		5, 2, 1, 1, // bows
		3, 3, 4, // crossbows
		5, 3, 3, 1, // trident
		5, 4, 3, // mace
		1, 3, 1, 1 // general
	};

	return MAX_LEVEL[enchantment];
}

LOOT_LIB_FUNCTION void get_applicable_enchantments(ItemCategory category, EnchantType *enchants, size_t *size) {
    const EnchantType enchantments[] = {
        PROTECTION, FIRE_PROTECTION, FEATHER_FALLING, BLAST_PROTECTION, PROJECTILE_PROTECTION,
        RESPIRATION, AQUA_AFFINITY, THORNS, DEPTH_STRIDER, 
        SHARPNESS, SMITE, BANE_OF_ARTHROPODS, KNOCKBACK, FIRE_ASPECT, LOOTING, SWEEPING_EDGE,
        EFFICIENCY, SILK_TOUCH, UNBREAKING, FORTUNE, 
        POWER, PUNCH, FLAME, INFINITY_ENCHANTMENT,
        LUCK_OF_THE_SEA, LURE, LOYALTY, IMPALING, RIPTIDE, CHANNELING,
        MULTISHOT, QUICK_CHARGE, PIERCING, DENSITY, BREACH,
        CURSE_OF_BINDING, CURSE_OF_VANISHING, FROST_WALKER, MENDING, NO_ENCHANTMENT
    };

    for (size_t i = 0; i < LENGTH(enchantments); i++) {
        if (enchantment_can_apply(category, enchantments[i])) {
            enchants[*size] = enchantments[i];
            (*size)++;
        }
    }
}