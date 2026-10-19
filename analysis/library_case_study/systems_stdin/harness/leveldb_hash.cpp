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
int main() {
  int q; if (scanf("%d", &q) != 1) return 1;
  static char buf[1 << 20];
  for (int i = 0; i < q; i++) { if (scanf("%s", buf) != 1) return 1; printf("%u\n", Hash(buf, strlen(buf), 0xbc9f1d34)); }
  return 0;
}
