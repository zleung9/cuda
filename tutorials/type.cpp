#include <stdio.h>
#include <stdlib.h>

void testSizeT();
void testStruct();
void typeCasting();

int main() {
    // testSizeT();
    // testStruct();
    typeCasting();
}

void testSizeT() {
    int arr[] = {1, 2, 3, 4, 5};
    size_t size = sizeof(arr) / sizeof(arr[0]);
    printf("size of arr: %zu\n", size);
    printf("size of size_t: %zu\n", sizeof(size_t));
    printf("int size in bytes: %zu\n", sizeof(int));
    // z -> size_t
    // u -> unsigned int
    // %zu -> size_t
    // src: https://cplusplus.com/reference/cstdio/printf/

    // Output:
    // size of arr: 5
    // size of size_t: 8
    // int size in bytes: 4

}

typedef struct {
    int x;
    int y;
} Point;

void testStruct() {
    Point p = {10, -20};
    printf("Point x: %d, y: %d\n", p.x, p.y);
    printf("size of Point: %zu\n", sizeof(Point));
}

void typeCasting() {
    int a = 69;
    double b = 20.5;
    int c = (int)b;
    double d = (double)a;
    char e = (char)a; // 69 -> 'E'
    printf("c: %d, d: %f e: %c\n", c, d, e);

}