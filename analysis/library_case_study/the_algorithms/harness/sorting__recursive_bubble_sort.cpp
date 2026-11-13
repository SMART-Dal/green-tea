// TheAlgorithms/C-Plus-Plus sorting/recursive_bubble_sort.cpp (verbatim, tests and main removed).
#include <cstdio>
#include <string>
/**
 * @file
 * @author [Aditya Prakash](https://adityaprakash.tech)
 * @brief This is an implementation of a recursive version of the [Bubble sort
 algorithm](https://www.geeksforgeeks.org/recursive-bubble-sort/)
 *
 * @details
 * The working principle of the Bubble sort algorithm.

 * Bubble sort is a simple sorting algorithm used to rearrange a set of
 ascending or descending order elements.
 * Bubble sort gets its name from the fact that data "bubbles" to the top of the
 dataset.

 * ### Algorithm

 * What is Swap?

 * Swapping two numbers means that we interchange their values.
 * Often, an additional variable is required for this operation.
 * This is further illustrated in the following:

 * void swap(int x, int y){
 *     int z = x;
 *     x = y;
 *     y = z;
 * }

 * The above process is a typical displacement process.
 * When we assign a value to x, the old value of x is lost.
 * That's why we create a temporary variable z to store the initial value of x.
 * z is further used to assign the initial value of x to y, to complete
 swapping.

 * Recursion

 * While the recursive method does not necessarily have advantages over
 iterative
 * versions, but it is useful to enhance the understanding of the algorithm and
 * recursion itself. In Recursive Bubble sort algorithm, we firstly call the
 * function on the entire array, and for every subsequent function call, we
 exclude
 * the last element. This fixes the last element for that sub-array.Formally,
 for
 * `ith` iteration, we consider elements up to n-i, where n is the number of
 * elements in the array. Exit condition: n==1; i.e. the sub-array contains only
 * one element.

 * Complexity
 * Time complexity: O(n) best case; O(n²) average case; O(n²) worst case
 * Space complexity: O(n)

 * We need to traverse the array `n * (n-1)` times. However, if the entire array
 is
 * already sorted, then we need to traverse it only once. Hence, O(n) is the
 best case
 * complexity
*/

#include <algorithm>  /// for std::is_sorted
#include <cassert>    /// for assert
#include <cstdint>
#include <iostream>   /// for IO operations
#include <vector>     /// for std::vector

/**
 * @namespace sorting
 * @brief Sorting algorithms
 */
namespace sorting {

/**
 * @brief This is an implementation of the recursive_bubble_sort. A vector is
 * passed to the function which is then dereferenced, so that the changes are
 * reflected in the original vector. It also accepts a second parameter of
 * type `int` and name `n`, which is the size of the array.
 *
 * @tparam T type of data variables in the array
 * @param nums our array of elements.
 * @param n size of the array
 */
template <typename T>
void recursive_bubble_sort(std::vector<T> *nums, uint64_t n) {
    if (n == 1) {  //!< base case; when size of the array is 1
        return;
    }

    for (uint64_t i = 0; i < n - 1; i++) {  //!< iterating over the entire array
        //!< if a larger number appears before the smaller one, swap them.
        if ((*nums)[i] > (*nums)[i + 1]) {
            std::swap((*nums)[i], (*nums)[i + 1]);
        }
    }

    //!< calling the function after we have fixed the last element
    recursive_bubble_sort(nums, n - 1);
}
}  // namespace sorting

static unsigned long long ck(const std::vector<long long>& v) { unsigned long long h = 1469598103934665603ULL; for (long long x : v) { h ^= (unsigned long long)x; h *= 1099511628211ULL; } return h; }
int main() {
  int n; if (scanf("%d", &n) != 1) return 1;
  std::vector<long long> in(n); for (auto& x : in) if (scanf("%lld", &x) != 1) return 1;
  std::vector<int64_t> a(in.begin(), in.end());
  if (n > 0) sorting::recursive_bubble_sort(&a, (uint64_t)n);
  printf("%llu\n", ck(std::vector<long long>(a.begin(), a.end())));
  return 0;
}
