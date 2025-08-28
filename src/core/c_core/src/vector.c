#include "c_core/vector.h"
#include <stdlib.h>
#include <string.h> // For memmove

#define VECTOR_DEFAULT_CAPACITY 16
#define VECTOR_GROWTH_FACTOR 2

// --- Private Helper Functions ---

/**
 * @brief Resizes the vector to a new capacity.
 * @param vec The vector to resize.
 * @param new_capacity The new capacity.
 * @return True on success, false on failure.
 */
static bool vector_resize(c_vector* vec, size_t new_capacity) {
    if (new_capacity <= vec->capacity) {
        // Cannot shrink, this function only grows
        return false;
    }

    void** new_data = realloc(vec->data, new_capacity * sizeof(void*));
    if (!new_data) {
        return false; // realloc failed
    }

    vec->data = new_data;
    vec->capacity = new_capacity;
    return true;
}


// --- Public API Implementation ---

c_vector* vector_create(size_t initial_capacity) {
    c_vector* vec = (c_vector*)malloc(sizeof(c_vector));
    if (!vec) {
        return NULL;
    }

    vec->size = 0;
    vec->capacity = (initial_capacity > 0) ? initial_capacity : VECTOR_DEFAULT_CAPACITY;

    vec->data = (void**)malloc(vec->capacity * sizeof(void*));
    if (!vec->data) {
        free(vec);
        return NULL;
    }

    return vec;
}

void vector_destroy(c_vector* vec, void (*free_data_fn)(void*)) {
    if (!vec) {
        return;
    }

    if (free_data_fn) {
        for (size_t i = 0; i < vec->size; ++i) {
            if (vec->data[i] != NULL) {
                free_data_fn(vec->data[i]);
            }
        }
    }

    free(vec->data);
    free(vec);
}

bool vector_push_back(c_vector* vec, void* item) {
    if (!vec) {
        return false;
    }

    if (vec->size >= vec->capacity) {
        size_t new_capacity = (vec->capacity == 0) ? VECTOR_DEFAULT_CAPACITY : vec->capacity * VECTOR_GROWTH_FACTOR;
        if (!vector_resize(vec, new_capacity)) {
            return false;
        }
    }

    vec->data[vec->size++] = item;
    return true;
}

void* vector_get(const c_vector* vec, size_t index) {
    if (!vec || index >= vec->size) {
        return NULL;
    }
    return vec->data[index];
}

bool vector_remove(c_vector* vec, size_t index, void (*free_data_fn)(void*)) {
    if (!vec || index >= vec->size) {
        return false;
    }

    if (free_data_fn && vec->data[index] != NULL) {
        free_data_fn(vec->data[index]);
    }

    // Shift elements to the left to fill the gap
    if (index < vec->size - 1) {
        memmove(&vec->data[index], &vec->data[index + 1], (vec->size - index - 1) * sizeof(void*));
    }

    vec->size--;
    return true;
}

size_t vector_size(const c_vector* vec) {
    return vec ? vec->size : 0;
}

size_t vector_capacity(const c_vector* vec) {
    return vec ? vec->capacity : 0;
}
