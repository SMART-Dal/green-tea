// TheAlgorithms/C-Plus-Plus sorting/quick_sort_iterative.cpp (verbatim, tests and main removed).
#include <cstdio>
#include <string>
/**
 * @file
 * @brief Quick Sort without recursion. This method uses the stack instead.
 * Both recursive and iterative implementations have O(n log n) best case
 * and O(n^2) worst case.
 * @details
 * https://stackoverflow.com/questions/12553238/quicksort-iterative-or-recursive
 * https://en.wikipedia.org/wiki/Quicksort
 * https://www.geeksforgeeks.org/iterative-quick-sort/
 * @author [Sebe324](https://github.com/sebe324)
 */

#include <iostream> /// for std::cout
#include <vector> /// for std::vector
#include <stack> /// for std::stack
#include <algorithm> /// for std::is_sorted
#include <cassert> /// for assert


/**
 * @namespace sorting
 * @brief Sorting algorithms
 */
namespace sorting {
/**
 * @brief The partition function sorts the array from
 * start to end and uses the last element as the pivot.
 * @param arr the array to be sorted
 * @param start starting index
 * @param end ending index
 * @return int next index of the pivot
 */
int partition(std::vector<int> &arr, int start, int end)
{
    int pivot = arr[end];
    int index = start - 1;

    for (int j = start; j < end; j++) {
        if (arr[j] <= pivot) {
            std::swap(arr[++index], arr[j]);
        }
    }

    std::swap(arr[index + 1], arr[end]);
    return index + 1;
}

/**
 * @brief The main sorting function
 * @details The iterative quick sort uses
 * the stack instead of recursion for saving
 * and restoring the environment between calls.
 * It does not need the end and start params, because
 * it is not recursive.
 * @param arr array to be sorted
 * @return void
 */
void iterativeQuickSort(std::vector<int> &arr)
{
    std::stack<int> stack;
    int start = 0;
    int end = arr.size()-1;
    stack.push(start);
    stack.push(end);

    while(!stack.empty())
    {
        end = stack.top();
        stack.pop();
        start = stack.top();
        stack.pop();

        int pivotIndex = partition(arr,start,end);

        if(pivotIndex -1 > start)
        {
            stack.push(start);
            stack.push(pivotIndex-1);
        }

        if(pivotIndex+1<end)
        {
            stack.push(pivotIndex+1);
            stack.push(end);
        }
    }
}

} // namespace sorting

static unsigned long long ck(const std::vector<long long>& v) { unsigned long long h = 1469598103934665603ULL; for (long long x : v) { h ^= (unsigned long long)x; h *= 1099511628211ULL; } return h; }
int main() {
  int n; if (scanf("%d", &n) != 1) return 1;
  std::vector<long long> in(n); for (auto& x : in) if (scanf("%lld", &x) != 1) return 1;
  std::vector<int> a(in.begin(), in.end());
  if (n > 0) sorting::iterativeQuickSort(a);
  printf("%llu\n", ck(std::vector<long long>(a.begin(), a.end())));
  return 0;
}
