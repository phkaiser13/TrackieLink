#include "c_core/linked_list.h"
#include <stdlib.h>

// --- Public API Implementation ---

linked_list_t* list_create(void (*free_data_fn)(void*)) {
    linked_list_t* list = (linked_list_t*)malloc(sizeof(linked_list_t));
    if (!list) {
        return NULL;
    }
    list->head = NULL;
    list->tail = NULL;
    list->size = 0;
    list->free_data_fn = free_data_fn;
    return list;
}

void list_destroy(linked_list_t* list) {
    if (!list) {
        return;
    }

    list_node_t* current = list->head;
    while (current != NULL) {
        list_node_t* next = current->next;
        if (list->free_data_fn && current->data) {
            list->free_data_fn(current->data);
        }
        free(current);
        current = next;
    }
    free(list);
}

bool list_push_front(linked_list_t* list, void* data) {
    if (!list) {
        return false;
    }
    list_node_t* new_node = (list_node_t*)malloc(sizeof(list_node_t));
    if (!new_node) {
        return false;
    }
    new_node->data = data;
    new_node->prev = NULL;
    new_node->next = list->head;

    if (list->head) {
        list->head->prev = new_node;
    } else {
        // List was empty, so this new node is also the tail.
        list->tail = new_node;
    }
    list->head = new_node;
    list->size++;
    return true;
}

bool list_push_back(linked_list_t* list, void* data) {
    if (!list) {
        return false;
    }
    list_node_t* new_node = (list_node_t*)malloc(sizeof(list_node_t));
    if (!new_node) {
        return false;
    }
    new_node->data = data;
    new_node->next = NULL;
    new_node->prev = list->tail;

    if (list->tail) {
        list->tail->next = new_node;
    } else {
        // List was empty, so this new node is also the head.
        list->head = new_node;
    }
    list->tail = new_node;
    list->size++;
    return true;
}

void* list_pop_front(linked_list_t* list) {
    if (!list || !list->head) {
        return NULL;
    }

    list_node_t* node_to_remove = list->head;
    void* data = node_to_remove->data;

    list->head = node_to_remove->next;
    if (list->head) {
        list->head->prev = NULL;
    } else {
        // The list is now empty.
        list->tail = NULL;
    }

    free(node_to_remove);
    list->size--;
    return data;
}

void* list_pop_back(linked_list_t* list) {
    if (!list || !list->tail) {
        return NULL;
    }

    list_node_t* node_to_remove = list->tail;
    void* data = node_to_remove->data;

    list->tail = node_to_remove->prev;
    if (list->tail) {
        list->tail->next = NULL;
    } else {
        // The list is now empty.
        list->head = NULL;
    }

    free(node_to_remove);
    list->size--;
    return data;
}

size_t list_size(const linked_list_t* list) {
    return list ? list->size : 0;
}
