// TheAlgorithms/C-Plus-Plus math/modular_exponentiation.cpp (verbatim, tests and main removed).
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
/**
 * @file
 * @brief C++ Program for Modular Exponentiation Iteratively.
 * @details The task is to calculate the value of an integer a raised to an
 * integer exponent b under modulo c.
 * @note The time complexity of this approach is O(log b).
 *
 * Example:
 * (4^3) % 5 (where ^ stands for exponentiation and % for modulo)
 * (4*4*4) % 5
 * (4 % 5) * ( (4*4) % 5 )
 * 4 * (16 % 5)
 * 4 * 1
 * 4
 * We can also verify the result as 4^3 is 64 and 64 modulo 5 is 4
 *
 * @author [Shri2206](https://github.com/Shri2206)
 */
#include <cassert>   /// for assert
#include <cstdint>
#include <iostream>  /// for io operations
/**
 * @namespace math
 * @brief Mathematical algorithms
 */
namespace math {
/**
 * @brief This function calculates a raised to exponent b under modulo c using
 * modular exponentiation.
 * @param a integer base
 * @param b unsigned integer exponent
 * @param c integer modulo
 * @return a raised to power b modulo c
 */
uint64_t power(uint64_t a, uint64_t b, uint64_t c) {
    uint64_t ans = 1;  /// Initialize the answer to be returned
    a = a % c;         /// Update a if it is more than or equal to c
    if (a == 0) {
        return 0;  /// In case a is divisible by c;
    }
    while (b > 0) {
        /// If b is odd, multiply a with answer
        if (b & 1) {
            ans = ((ans % c) * (a % c)) % c;
        }
        /// b must be even now
        b = b >> 1;  /// b = b/2
        a = ((a % c) * (a % c)) % c;
    }
    return ans;
}

}  // namespace math

int main() {
  int q; if (scanf("%d", &q) != 1) return 1;
  for (int t = 0; t < q; t++) { unsigned long long a, b, c; scanf("%llu %llu %llu", &a, &b, &c); printf("%llu\n", (unsigned long long)math::power(a, b, c)); }
  return 0;
}
