// TheAlgorithms/C-Plus-Plus sorting/pancake_sort.cpp (verbatim, tests and main removed).
#include <cstdio>
#include <string>
/**
 * @file
 * @brief pancake sort sorts a disordered stack of pancakes by flipping any
 * number of pancakes using a spatula using minimum number of flips.
 *
 * @details
 * Unlike a traditional sorting algorithm, which attempts to sort with the
 * fewest comparisons possible, the goal is to sort the sequence in as few
 * reversals as possible. Overall time complexity of pancake sort is O(n^2) For
 * example: example 1:- Disordered pancake sizes: {2,5,3,7,8} Sorted:
 * {2,3,5,7,8} For example: example 2:- Disordered pancake sizes:
 * {22,51,37,73,81} Sorted: {22,37,51,73,81}
 * @author [Divyansh Gupta](https://github.com/divyansh12323)
 * @see more on [Pancake sort](https://en.wikipedia.org/wiki/Pancake_sorting)
 * @see related problem at
 * [Leetcode](https://leetcode.com/problems/pancake-sorting/)
 */

#include <algorithm>  // for std::is_sorted
#include <cassert>    // for std::assert
#include <iostream>   // for io operations
#include <vector>     // for std::vector

/**
 * @namespace sorting
 * @brief Sorting algorithms
 */
namespace sorting {
/**
 * @namespace pancake_sort
 * @brief Functions for [Pancake
 * sort](https://en.wikipedia.org/wiki/Pancake_sorting) algorithm
 */
namespace pancake_sort {
/**
 * @brief This implementation is for reversing elements in a a C-style array .
 * @param [start,end] arr our vector of elements.
 * @param start starting index of array
 * @param end ending index of array
 * @returns void
 */
template <typename T>
void reverse(std::vector<T> &arr, int start, int end) {
    T temp;  // Temporary variable
    while (start <= end) {
        temp = arr[start];
        arr[start] = arr[end];
        arr[end] = temp;
        start++;
        end--;
    }
}
/**
 * @brief This implementation is for a C-style array input that gets modified in
 * place.
 * @param [start,end] arr our vector of elements.
 * @param size size of given array
 * @returns 0 on exit
 */
template <typename T>
int pancakeSort(std::vector<T> &arr, int size) {
    for (int i = size; i > 1; --i) {
        int max_index = 0, j = 0;  // intialize some variables.
        T max_value = 0;
        for (j = 0; j < i; j++) {
            if (arr[j] >= max_value) {
                max_value = arr[j];
                max_index = j;
            }
        }
        if (max_index != i - 1)  // check for reversing
        {
            reverse(arr, 0, max_index);
            reverse(arr, 0, i - 1);
        }
    }
    return 0;
}
}  // namespace pancake_sort
}  // namespace sorting

static unsigned long long ck(const std::vector<long long>& v) { unsigned long long h = 1469598103934665603ULL; for (long long x : v) { h ^= (unsigned long long)x; h *= 1099511628211ULL; } return h; }
int main() {
  int n; if (scanf("%d", &n) != 1) return 1;
  std::vector<long long> in(n); for (auto& x : in) if (scanf("%lld", &x) != 1) return 1;
  std::vector<int> a(in.begin(), in.end());
  sorting::pancake_sort::pancakeSort(a, n);
  printf("%llu\n", ck(std::vector<long long>(a.begin(), a.end())));
  return 0;
}
