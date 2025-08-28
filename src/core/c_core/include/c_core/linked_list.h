#ifndef C_CORE_LINKED_LIST_H
#define C_CORE_LINKED_LIST_H

#include <stddef.h>
#include <stdbool.h>

/**
 * @struct list_node_t
 * @brief A node in a doubly linked list.
 */
typedef struct list_node {
    void* data;                 /**< Pointer to the data stored in the node. */
    struct list_node* next;     /**< Pointer to the next node in the list. */
    struct list_node* prev;     /**< Pointer to the previous node in the list. */
} list_node_t;

/**
 * @struct linked_list_t
 * @brief The doubly linked list structure.
 */
typedef struct {
    list_node_t* head;          /**< Pointer to the first node in the list. */
    list_node_t* tail;          /**< Pointer to the last node in the list. */
    size_t size;                /**< The number of nodes in the list. */
    void (*free_data_fn)(void*);/**< Optional function to free the data in each node. */
} linked_list_t;


// --- Function Prototypes ---

/**
 * @brief Creates a new, empty doubly linked list.
 * @param free_data_fn A function pointer to be used for freeing the data stored in each node
 *                     when the list is destroyed. Can be NULL if data does not need to be freed.
 * @return A pointer to the newly created list, or NULL on failure.
 */
linked_list_t* list_create(void (*free_data_fn)(void*));

/**
 * @brief Destroys a linked list and all of its nodes, freeing associated memory.
 * @param list The linked list to destroy.
 */
void list_destroy(linked_list_t* list);

/**
 * @brief Adds a new element to the front of the list.
 * @param list The linked list.
 * @param data A pointer to the data to be stored.
 * @return True on success, false on failure (e.g., memory allocation error).
 */
bool list_push_front(linked_list_t* list, void* data);

/**
 * @brief Adds a new element to the back of the list.
 * @param list The linked list.
 * @param data A pointer to the data to be stored.
 * @return True on success, false on failure.
 */
bool list_push_back(linked_list_t* list, void* data);

/**
 * @brief Removes and returns the element from the front of the list.
 * @param list The linked list.
 * @return A pointer to the data from the removed node, or NULL if the list is empty.
 */
void* list_pop_front(linked_list_t* list);

/**
 * @brief Removes and returns the element from the back of the list.
 * @param list The linked list.
 * @return A pointer to the data from the removed node, or NULL if the list is empty.
 */
void* list_pop_back(linked_list_t* list);

/**
 * @brief Gets the current number of elements in the list.
 * @param list The linked list.
 * @return The size of the list.
 */
size_t list_size(const linked_list_t* list);

#endif // C_CORE_LINKED_LIST_H
