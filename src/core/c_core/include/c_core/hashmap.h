#ifndef C_CORE_HASHMAP_H
#define C_CORE_HASHMAP_H

#include <stddef.h>
#include <stdbool.h>

/**
 * @struct hashmap_entry_t
 * @brief An entry in a hash map bucket (a key-value pair).
 */
typedef struct hashmap_entry {
    char* key;                  /**< The string key for this entry. */
    void* value;                /**< The value associated with the key. */
    struct hashmap_entry* next; /**< Pointer to the next entry in case of collision. */
} hashmap_entry_t;

/**
 * @struct hashmap_t
 * @brief The hash map structure itself.
 */
typedef struct {
    hashmap_entry_t** buckets;  /**< Array of pointers to bucket entries. */
    size_t bucket_count;        /**< The number of buckets in the hash map. */
    size_t size;                /**< The number of key-value pairs stored. */
} hashmap_t;


// --- Function Prototypes ---

/**
 * @brief Creates a new hash map.
 * @param initial_capacity The initial number of buckets. If 0, a default is used.
 * @return A pointer to the new hash map, or NULL on failure.
 */
hashmap_t* hashmap_create(size_t initial_capacity);

/**
 * @brief Destroys a hash map and frees all associated memory.
 * @param map The hash map to destroy.
 * @param free_value_fn A function pointer to free the stored values. Can be NULL.
 */
void hashmap_destroy(hashmap_t* map, void (*free_value_fn)(void*));

/**
 * @brief Inserts a key-value pair. If the key already exists, its value is updated.
 *
 * This function makes a copy of the key, so the caller retains ownership of the original key.
 *
 * @param map The hash map.
 * @param key The string key.
 * @param value The value to store.
 * @return True on success, false on failure (e.g., memory allocation error).
 */
bool hashmap_put(hashmap_t* map, const char* key, void* value);

/**
 * @brief Gets a value from the hash map by its key.
 * @param map The hash map.
 * @param key The string key.
 * @return The value associated with the key, or NULL if the key is not found.
 */
void* hashmap_get(const hashmap_t* map, const char* key);

/**
 * @brief Removes a key-value pair from the hash map.
 * @param map The hash map.
 * @param key The string key to remove.
 * @param free_value_fn A function to free the removed value. Can be NULL.
 * @return True if the key was found and removed, false otherwise.
 */
bool hashmap_remove(hashmap_t* map, const char* key, void (*free_value_fn)(void*));

/**
 * @brief Gets the number of elements currently stored in the hash map.
 * @param map The hash map.
 * @return The number of elements.
 */
size_t hashmap_size(const hashmap_t* map);

#endif // C_CORE_HASHMAP_H
