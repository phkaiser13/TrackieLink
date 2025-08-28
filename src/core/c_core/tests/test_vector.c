#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include "c_core/vector.h"

// --- Test Functions ---

void test_create_and_destroy() {
    printf("Running test: %s\n", __func__);
    c_vector* vec = vector_create(10);
    assert(vec != NULL);
    assert(vector_size(vec) == 0);
    assert(vector_capacity(vec) == 10);
    vector_destroy(vec, NULL);
    printf("...PASSED\n");
}

void test_push_and_get() {
    printf("Running test: %s\n", __func__);
    c_vector* vec = vector_create(2);
    int a = 10, b = 20, c = 30;

    assert(vector_push_back(vec, &a) == true);
    assert(vector_push_back(vec, &b) == true);
    assert(vector_size(vec) == 2);
    assert(vector_capacity(vec) == 2);

    // This push should trigger a resize
    assert(vector_push_back(vec, &c) == true);
    assert(vector_size(vec) == 3);
    assert(vector_capacity(vec) > 2);

    int* val_a = (int*)vector_get(vec, 0);
    int* val_b = (int*)vector_get(vec, 1);
    int* val_c = (int*)vector_get(vec, 2);

    assert(val_a != NULL && *val_a == 10);
    assert(val_b != NULL && *val_b == 20);
    assert(val_c != NULL && *val_c == 30);

    assert(vector_get(vec, 3) == NULL); // Out of bounds check

    vector_destroy(vec, NULL); // Data is on the stack, no need to free
    printf("...PASSED\n");
}

void test_remove() {
    printf("Running test: %s\n", __func__);
    c_vector* vec = vector_create(5);

    // Use malloc'd data to test the free function
    int* d1 = (int*)malloc(sizeof(int)); *d1 = 1;
    int* d2 = (int*)malloc(sizeof(int)); *d2 = 2;
    int* d3 = (int*)malloc(sizeof(int)); *d3 = 3;

    vector_push_back(vec, d1);
    vector_push_back(vec, d2);
    vector_push_back(vec, d3);

    assert(vector_size(vec) == 3);

    // Remove middle element
    assert(vector_remove(vec, 1, free) == true); // free d2
    assert(vector_size(vec) == 2);

    int* val_0 = (int*)vector_get(vec, 0);
    int* val_1 = (int*)vector_get(vec, 1);

    assert(val_0 != NULL && *val_0 == 1);
    assert(val_1 != NULL && *val_1 == 3); // d3 should have shifted to index 1

    // Test removing from the end
    assert(vector_remove(vec, 1, free) == true); // free d3
    assert(vector_size(vec) == 1);
    int* val_final = (int*)vector_get(vec, 0);
    assert(val_final != NULL && *val_final == 1);

    vector_destroy(vec, free); // free the remaining d1
    printf("...PASSED\n");
}


int main() {
    printf("--- Starting c_vector tests ---\n");
    test_create_and_destroy();
    test_push_and_get();
    test_remove();
    printf("--- All c_vector tests passed ---\n");
    return 0;
}
