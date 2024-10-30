#include <stdio.h>
#include <stdlib.h>

void testNullPointer();
void testArrayPointer();
void testMatrixPointer();


int main() {
    // testNullPointer();
    // testArrayPointer();
    testMatrixPointer();
}


void testNullPointer() {
    // Initialize pointer to NULL
    int* ptr = NULL;
    printf("Initial ptr value: %p\n", ptr);

    if (ptr == NULL) {
        printf("ptr is NULL, cannot dereference\n");
    }

    ptr = (int*)malloc(sizeof(int));
    if (ptr == NULL) {
        printf("Memory allocation failed\n");
        return;
    }

    printf("After allocation, ptr value: %p\n", (void*)ptr);

    *ptr = 42;
    printf("Value at ptr: %d\n", *ptr);

    free(ptr);
    ptr = NULL;

    printf("After free, ptr value: %p\n", (void*)ptr);

    if (ptr == NULL) {
        printf("ptr is NULL, safely avoided use after free\n");
    }
    // Output:
    // 1. Initial ptr value: 0x0
    // 2. ptr is NULL, cannot dereference
    // 4. After allocation, ptr value: 0x135605e70
    // 5. Value at ptr: 42
    // 6. After free, ptr value: 0x0
    // 7. ptr is NULL, safely avoided use after free
}


void testArrayPointer() {
    int arr[] = {1, 2, 3, 4, 5};
    printf("arr itself is a pointer, its value: %p\n", arr);

    int* ptr = arr;
    printf("It points to the first element whose value is: %d\n", *ptr);

    for (int i = 0; i <5; i++) {
        printf("%d %p\n", *ptr, ptr);
        ptr++;
    }
    // Output:
    // arr itself is a pointer, its value: 0x16b24aeb0
    // It points to the first element whose value is: 1
    // 1 0x16b24aeb0
    // 2 0x16b24aeb4
    // 3 0x16b24aeb8
    // 4 0x16b24aebc
    // 5 0x16b24aec0
}


void testMatrixPointer() {
    int arr1[] = {1, 2, 3, 4, 5};
    int arr2[] = {6, 7, 8, 9, 10};

    int* ptr1 = arr1;
    int* ptr2 = arr2;

    int* matrix[] = {ptr1, ptr2};

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%d ", *matrix[i]);
            matrix[i]++;
        }
        printf("\n");
    }
    // Output:
    // 1 2 3 4 5
    // 6 7 8 9 10
}

