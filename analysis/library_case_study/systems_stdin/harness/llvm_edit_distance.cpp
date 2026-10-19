// LLVM llvm/ADT/edit_distance.h: ComputeMappedEditDistance / ComputeEditDistance (verbatim;
// ArrayRef<T> and SmallVector replaced by std::vector).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
template <typename T> using ArrayRef = std::vector<T>;
template <typename T, typename Functor>
unsigned ComputeMappedEditDistance(const ArrayRef<T> &FromArray, const ArrayRef<T> &ToArray,
                                   Functor Map, bool AllowReplacements = true,
                                   unsigned MaxEditDistance = 0) {
  typename ArrayRef<T>::size_type m = FromArray.size();
  typename ArrayRef<T>::size_type n = ToArray.size();
  if (MaxEditDistance) {
    typename ArrayRef<T>::size_type AbsDiff = m > n ? m - n : n - m;
    if (AbsDiff > MaxEditDistance)
      return MaxEditDistance + 1;
  }
  std::vector<unsigned> Row(n + 1);
  for (unsigned i = 1; i < Row.size(); ++i)
    Row[i] = i;
  for (typename ArrayRef<T>::size_type y = 1; y <= m; ++y) {
    Row[0] = y;
    unsigned BestThisRow = Row[0];
    unsigned Previous = y - 1;
    const auto &CurItem = Map(FromArray[y - 1]);
    for (typename ArrayRef<T>::size_type x = 1; x <= n; ++x) {
      int OldRow = Row[x];
      if (AllowReplacements) {
        Row[x] = std::min(Previous + (CurItem == Map(ToArray[x - 1]) ? 0u : 1u),
                          std::min(Row[x - 1], Row[x]) + 1);
      }
      else {
        if (CurItem == Map(ToArray[x - 1]))
          Row[x] = Previous;
        else Row[x] = std::min(Row[x-1], Row[x]) + 1;
      }
      Previous = OldRow;
      BestThisRow = std::min(BestThisRow, Row[x]);
    }
    if (MaxEditDistance && BestThisRow > MaxEditDistance)
      return MaxEditDistance + 1;
  }
  unsigned Result = Row[n];
  return Result;
}
template <typename T>
unsigned ComputeEditDistance(const ArrayRef<T> &FromArray, const ArrayRef<T> &ToArray,
                             bool AllowReplacements = true,
                             unsigned MaxEditDistance = 0) {
  return ComputeMappedEditDistance(
      FromArray, ToArray, [](const T &X) -> const T & { return X; },
      AllowReplacements, MaxEditDistance);
}
int main() {
  int q; if (scanf("%d", &q) != 1) return 1;
  static char a[1 << 16], b[1 << 16];
  for (int i = 0; i < q; i++) {
    if (scanf("%s %s", a, b) != 2) return 1;
    std::vector<char> x(a, a + strlen(a)), y(b, b + strlen(b));
    printf("%u\n", ComputeEditDistance(x, y));
  }
  return 0;
}
