// TheAlgorithms/C-Plus-Plus sorting/dnf_sort.cpp (verbatim, tests and main removed).
#include <cstdio>
#include <string>
/**
 * @file
 * @brief Implementation of the [DNF
 * sort](https://www.geeksforgeeks.org/sort-an-array-of-0s-1s-and-2s/)
 * implementation
 * @details
 * C++ program to sort an array with 0, 1 and 2 in a single pass(DNF sort).
 * Since one traversal of the array is there hence it works in O(n) time
 * complexity.
 * @author [Sujal Gupta](https://github.com/heysujal)
 */

#include <algorithm>  /// for std::is_sorted
#include <cassert>    /// for assert
#include <cstdint>
#include <iostream>   /// for std::swap and io operations
#include <vector>     /// for std::vector

/**
 * @namespace sorting
 * @breif Sorting algorithms
 */
namespace sorting {
/**
 * @namespace dnf_sort
 * @brief Functions for the [DNF
 * sort](https://en.wikipedia.org/wiki/Dutch_national_flag_problem)
 * implementation
 */
namespace dnf_sort {
/**
 * @brief The main function implements DNF sort
 * @tparam T type of array
 * @param a array to be sorted,
 * @param arr_size size of array
 * @returns void
 */
template <typename T>
std::vector<T> dnfSort(const std::vector<T> &in_arr) {
    std::vector<T> arr(in_arr);
    uint64_t lo = 0;
    uint64_t hi = arr.size() - 1;
    uint64_t mid = 0;

    // Iterate till all the elements
    // are sorted
    while (mid <= hi) {
        switch (arr[mid]) {
            // If the element is 0
            case 0:
                std::swap(arr[lo++], arr[mid++]);
                break;

            // If the element is 1 .
            case 1:
                mid++;
                break;

            // If the element is 2
            case 2:
                std::swap(arr[mid], arr[hi--]);
                break;
        }
    }
    return arr;
}
}  // namespace dnf_sort
}  // namespace sorting

static unsigned long long ck(const std::vector<long long>& v) { unsigned long long h = 1469598103934665603ULL; for (long long x : v) { h ^= (unsigned long long)x; h *= 1099511628211ULL; } return h; }
int main() {
  int n; if (scanf("%d", &n) != 1) return 1;
  std::vector<long long> in(n); for (auto& x : in) if (scanf("%lld", &x) != 1) return 1;
  std::vector<uint64_t> a(in.begin(), in.end());
  std::vector<uint64_t> s = sorting::dnf_sort::dnfSort(a);
  printf("%llu\n", ck(std::vector<long long>(s.begin(), s.end())));
  return 0;
}
