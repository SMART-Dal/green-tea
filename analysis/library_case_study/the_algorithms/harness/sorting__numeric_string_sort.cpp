// TheAlgorithms/C-Plus-Plus sorting/numeric_string_sort.cpp (verbatim, tests and main removed).
#include <cstdio>
// Using general algorithms to sort a collection of strings results in
// alphanumeric sort. If it is a numeric string, it leads to unnatural sorting

// eg, an array of strings 1,10,100,2,20,200,3,30,300
// would be sorted in that same order by using conventional sorting,
// even though we know the correct sorting order is 1,2,3,10,20,30,100,200,300

// This Programme uses a comparator to sort the array in Numerical order instead
// of Alphanumeric order

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

bool NumericSort(std::string a, std::string b) {
    while (a[0] == '0') {
        a.erase(a.begin());
    }
    while (b[0] == '0') {
        b.erase(b.begin());
    }
    int n = a.length();
    int m = b.length();
    if (n == m)
        return a < b;
    return n < m;
}

int main() {
  int n; if (scanf("%d", &n) != 1) return 1;
  static char buf[64]; std::vector<std::string> v(n);
  for (auto& s : v) { scanf("%s", buf); s = buf; }
  std::sort(v.begin(), v.end(), NumericSort);
  unsigned long long h = 1469598103934665603ULL;
  for (auto& s : v) for (char c : s) { h ^= (unsigned char)c; h *= 1099511628211ULL; }
  printf("%llu\n", h);
  return 0;
}
