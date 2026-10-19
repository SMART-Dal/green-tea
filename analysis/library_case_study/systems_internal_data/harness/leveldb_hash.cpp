// LevelDB util/hash.cc: leveldb::Hash (verbatim), DecodeFixed32 from util/coding.h.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
inline uint32_t DecodeFixed32(const char* ptr) {
  const uint8_t* const buffer = reinterpret_cast<const uint8_t*>(ptr);
  return (static_cast<uint32_t>(buffer[0])) | (static_cast<uint32_t>(buffer[1]) << 8) |
         (static_cast<uint32_t>(buffer[2]) << 16) | (static_cast<uint32_t>(buffer[3]) << 24);
}
uint32_t Hash(const char* data, size_t n, uint32_t seed) {
  const uint32_t m = 0xc6a4a793;
  const uint32_t r = 24;
  const char* limit = data + n;
  uint32_t h = seed ^ (n * m);
  while (limit - data >= 4) {
    uint32_t w = DecodeFixed32(data);
    data += 4;
    h += w;
    h *= m;
    h ^= (h >> 16);
  }
  switch (limit - data) {
    case 3:
      h += static_cast<uint8_t>(data[2]) << 16;
      [[fallthrough]];
    case 2:
      h += static_cast<uint8_t>(data[1]) << 8;
      [[fallthrough]];
    case 1:
      h += static_cast<uint8_t>(data[0]);
      h *= m;
      h ^= (h >> r);
      break;
  }
  return h;
}
static uint64_t s;
static inline uint64_t rnd() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
int main() {
  int n, k; unsigned long long seed;
  if (scanf("%d %d %llu", &n, &k, &seed) != 3) return 1;
  s = seed | 1;
  std::vector<char> buf(n + 64);
  for (auto& c : buf) c = static_cast<char>(rnd());
  uint32_t acc = 0;
  for (int i = 0; i < k; i++) {
    size_t len = n > 64 ? n - (i % 61) : n;
    acc = acc * 31 + Hash(buf.data() + (i % 64), len, static_cast<uint32_t>(i));
  }
  printf("%u\n", acc);
  return 0;
}
