#include "c_core/hashmap.h"
#include <stdlib.h>
#include <string.h>

#define HASHMAP_DEFAULT_CAPACITY 64

// --- Private Helper Functions ---

/**
 * @brief The djb2 hash function for strings.
 * @param str The string to hash.
 * @return The calculated hash value.
 */
static unsigned long hash_string(const char* str) {
    unsigned long hash = 5381;
    int c;
    while ((c = *str++)) {
        hash = ((hash << 5) + hash) + c; // hash * 33 + c
    }
    return hash;
}

// --- Public API Implementation ---

hashmap_t* hashmap_create(size_t initial_capacity) {
    hashmap_t* map = (hashmap_t*)malloc(sizeof(hashmap_t));
    if (!map) {
        return NULL;
    }

    map->size = 0;
    map->bucket_count = (initial_capacity > 0) ? initial_capacity : HASHMAP_DEFAULT_CAPACITY;

    // Use calloc to initialize all buckets to NULL
    map->buckets = (hashmap_entry_t**)calloc(map->bucket_count, sizeof(hashmap_entry_t*));
    if (!map->buckets) {
        free(map);
        return NULL;
    }
    return map;
}

void hashmap_destroy(hashmap_t* map, void (*free_value_fn)(void*)) {
    if (!map) {
        return;
    }

    for (size_t i = 0; i < map->bucket_count; ++i) {
        hashmap_entry_t* entry = map->buckets[i];
        while (entry) {
            hashmap_entry_t* next = entry->next;
            if (free_value_fn && entry->value) {
                free_value_fn(entry->value);
            }
            free(entry->key); // Key was allocated with strdup
            free(entry);
            entry = next;
        }
    }
    free(map->buckets);
    free(map);
}

bool hashmap_put(hashmap_t* map, const char* key, void* value) {
    if (!map || !key) {
        return false;
    }

    // Note: A production-ready implementation should check the load factor
    // and resize the hash map if it gets too full. This is omitted for brevity.

    unsigned long hash = hash_string(key);
    size_t index = hash % map->bucket_count;

    hashmap_entry_t* entry = map->buckets[index];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            // Key already exists, update the value.
            // If the old value needs to be freed, the caller is responsible.
            entry->value = value;
            return true;
        }
        entry = entry->next;
    }

    // Key does not exist, create a new entry.
    hashmap_entry_t* new_entry = (hashmap_entry_t*)malloc(sizeof(hashmap_entry_t));
    if (!new_entry) {
        return false;
    }

    new_entry->key = strdup(key);
    if (!new_entry->key) {
        free(new_entry);
        return false;
    }
    new_entry->value = value;

    // Add to the front of the bucket's linked list.
    new_entry->next = map->buckets[index];
    map->buckets[index] = new_entry;
    map->size++;

    return true;
}

void* hashmap_get(const hashmap_t* map, const char* key) {
    if (!map || !key) {
        return NULL;
    }

    unsigned long hash = hash_string(key);
    size_t index = hash % map->bucket_count;

    hashmap_entry_t* entry = map->buckets[index];
    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            return entry->value;
        }
        entry = entry->next;
    }
    return NULL;
}

bool hashmap_remove(hashmap_t* map, const char* key, void (*free_value_fn)(void*)) {
    if (!map || !key) {
        return false;
    }

    unsigned long hash = hash_string(key);
    size_t index = hash % map->bucket_count;

    hashmap_entry_t* entry = map->buckets[index];
    hashmap_entry_t* prev = NULL;

    while (entry) {
        if (strcmp(entry->key, key) == 0) {
            if (prev) {
                prev->next = entry->next;
            } else {
                map->buckets[index] = entry->next;
            }

            if (free_value_fn && entry->value) {
                free_value_fn(entry->value);
            }
            free(entry->key);
            free(entry);
            map->size--;
            return true;
        }
        prev = entry;
        entry = entry->next;
    }

    return false; // Key not found
}

size_t hashmap_size(const hashmap_t* map) {
    return map ? map->size : 0;
}
