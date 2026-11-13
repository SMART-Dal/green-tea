// TheAlgorithms/C-Plus-Plus sorting/counting_sort.cpp (verbatim, tests and main removed).
#include <cstdio>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
using namespace std;

int Max(int Arr[], int N) {
    int max = Arr[0];
    for (int i = 1; i < N; i++)
        if (Arr[i] > max)
            max = Arr[i];
    return max;
}

int Min(int Arr[], int N) {
    int min = Arr[0];
    for (int i = 1; i < N; i++)
        if (Arr[i] < min)
            min = Arr[i];
    return min;
}

void Print(int Arr[], int N) {
    for (int i = 0; i < N; i++) cout << Arr[i] << ", ";
}

int *Counting_Sort(int Arr[], int N) {
    int max = Max(Arr, N);
    int min = Min(Arr, N);
    int *Sorted_Arr = new int[N];

    int *Count = new int[max - min + 1];
    for (int i = 0; i < max - min + 1; ++i) {
        Count[i] = 0;
    }

    for (int i = 0; i < N; i++) Count[Arr[i] - min]++;

    for (int i = 1; i < (max - min + 1); i++) Count[i] += Count[i - 1];

    for (int i = N - 1; i >= 0; i--) {
        Sorted_Arr[Count[Arr[i] - min] - 1] = Arr[i];
        Count[Arr[i] - min]--;
    }

    delete[] Count;
    return Sorted_Arr;
}

static unsigned long long ck(const std::vector<long long>& v) { unsigned long long h = 1469598103934665603ULL; for (long long x : v) { h ^= (unsigned long long)x; h *= 1099511628211ULL; } return h; }
int main() {
  int n; if (scanf("%d", &n) != 1) return 1;
  std::vector<long long> in(n); for (auto& x : in) if (scanf("%lld", &x) != 1) return 1;
  std::vector<int> a(in.begin(), in.end());
  int* s = Counting_Sort(a.data(), n);
  std::vector<long long> out(s, s + n); delete[] s;
  printf("%llu\n", ck(out));
  return 0;
}
