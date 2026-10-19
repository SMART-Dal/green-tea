// nginx src/core/ngx_murmurhash.c: ngx_murmur_hash2 (verbatim; u_char typedef inlined).
#include <cstdint>
#include <cstdio>
#include <cstddef>
#include <vector>
typedef unsigned char u_char;
uint32_t
ngx_murmur_hash2(u_char *data, size_t len)
{
    uint32_t  h, k;

    h = 0 ^ len;

    while (len >= 4) {
        k  = data[0];
        k |= data[1] << 8;
        k |= data[2] << 16;
        k |= data[3] << 24;

        k *= 0x5bd1e995;
        k ^= k >> 24;
        k *= 0x5bd1e995;

        h *= 0x5bd1e995;
        h ^= k;

        data += 4;
        len -= 4;
    }

    switch (len) {
    case 3:
        h ^= data[2] << 16;
        /* fall through */
    case 2:
        h ^= data[1] << 8;
        /* fall through */
    case 1:
        h ^= data[0];
        h *= 0x5bd1e995;
    }

    h ^= h >> 13;
    h *= 0x5bd1e995;
    h ^= h >> 15;

    return h;
}
static uint64_t st;
static inline uint64_t rnd() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
int main() {
  int n, k; unsigned long long seed;
  if (scanf("%d %d %llu", &n, &k, &seed) != 3) return 1;
  st = seed | 1;
  std::vector<u_char> buf(n + 64);
  for (auto& c : buf) c = static_cast<u_char>(rnd());
  uint32_t acc = 0;
  for (int i = 0; i < k; i++) acc = acc * 31 + ngx_murmur_hash2(buf.data() + (i % 64), 1 + (i * 37) % n);
  printf("%u\n", acc);
  return 0;
}
