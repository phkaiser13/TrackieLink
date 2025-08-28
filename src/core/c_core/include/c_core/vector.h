#ifndef C_CORE_VECTOR_H
#define C_CORE_VECTOR_H

#include <stddef.h>
#include <stdbool.h>

/**
 * @struct c_vector
 * @brief A generic dynamic array (vector) that holds void pointers.
 */
typedef struct {
    void** data;      /**< Array of void pointers to store any type. */
    size_t size;      /**< Number of elements currently in the vector. */
    size_t capacity;  /**< Total storage capacity of the vector. */
} c_vector;


// --- Function Prototypes ---

/**
 * @brief Creates a new vector with an initial capacity.
 * @param initial_capacity The initial capacity of the vector. If 0, a default capacity is used.
 * @return A pointer to the newly created vector, or NULL on failure.
 */
c_vector* vector_create(size_t initial_capacity);

/**
 * @brief Destroys a vector and frees its associated memory.
 *
 * @param vec The vector to destroy.
 * @param free_data_fn A function pointer to free the data stored in the vector.
 *                     If NULL, the data pointers themselves are not freed, only the
 *                     vector structure. This is useful if the vector stores primitives
 *                     or references managed elsewhere.
 */
void vector_destroy(c_vector* vec, void (*free_data_fn)(void*));

/**
 * @brief Pushes an element to the end of the vector. The vector will resize if necessary.
 * @param vec The vector.
 * @param item The item (void pointer) to push.
 * @return True on success, false on failure (e.g., memory allocation error).
 */
bool vector_push_back(c_vector* vec, void* item);

/**
 * @brief Gets the element at a specific index.
 * @param vec The vector.
 * @param index The index of the element.
 * @return The element (void pointer) at the given index, or NULL if the index is out of bounds.
 */
void* vector_get(const c_vector* vec, size_t index);

/**
 * @brief Removes an element from a specific index, shifting subsequent elements.
 * @param vec The vector.
 * @param index The index of the element to remove.
 * @param free_data_fn A function pointer to free the data being removed. Can be NULL.
 * @return True on success, false if the index is out of bounds.
 */
bool vector_remove(c_vector* vec, size_t index, void (*free_data_fn)(void*));

/**
 * @brief Gets the current number of elements in the vector.
 * @param vec The vector.
 * @return The size of the vector.
 */
size_t vector_size(const c_vector* vec);

/**
 * @brief Gets the current storage capacity of the vector.
 * @param vec The vector.
 * @return The capacity of the vector.
 */
size_t vector_capacity(const c_vector* vec);


#endif // C_CORE_VECTOR_H
