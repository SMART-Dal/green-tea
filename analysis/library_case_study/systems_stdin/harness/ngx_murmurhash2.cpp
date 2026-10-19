// nginx src/core/ngx_murmurhash.c: ngx_murmur_hash2 (verbatim; u_char typedef inlined).
#include <cstdint>
#include <cstdio>
#include <cstring>
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
int main() {
  int q; if (scanf("%d", &q) != 1) return 1;
  static char buf[1 << 20];
  for (int i = 0; i < q; i++) { if (scanf("%s", buf) != 1) return 1; printf("%u\n", ngx_murmur_hash2((u_char*)buf, strlen(buf))); }
  return 0;
}
