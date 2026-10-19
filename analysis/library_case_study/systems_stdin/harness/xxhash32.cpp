// xxHash xxhash.h: XXH32 and its helpers (verbatim; scalar configuration, XXH32_ENDJMP = 0, no streaming).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
typedef uint8_t xxh_u8;
typedef uint32_t xxh_u32;
typedef uint32_t XXH32_hash_t;
typedef enum { XXH_aligned, XXH_unaligned } XXH_alignment;
#define XXH_FORCE_INLINE static inline
#define XXH_PUREF
#define XXH_RESTRICT __restrict
#define XXH_PUBLIC_API
#define XXH_ASSERT(c) ((void)0)
#define XXH_FALLTHROUGH [[fallthrough]]
#define XXH32_ENDJMP 0
#define XXH_FORCE_ALIGN_CHECK 0
#define XXH_rotl32(x,r) (((x) << (r)) | ((x) >> (32 - (r))))
static xxh_u32 XXH_readLE32_align(const void* ptr, XXH_alignment align) { (void)align; xxh_u32 v; memcpy(&v, ptr, 4); return v; }
#define XXH_PRIME32_1  0x9E3779B1U
#define XXH_PRIME32_2  0x85EBCA77U
#define XXH_PRIME32_3  0xC2B2AE3DU
#define XXH_PRIME32_4  0x27D4EB2FU
#define XXH_PRIME32_5  0x165667B1U
static xxh_u32 XXH32_round(xxh_u32 acc, xxh_u32 input)
{
    acc += input * XXH_PRIME32_2;
    acc  = XXH_rotl32(acc, 13);
    acc *= XXH_PRIME32_1;
    return acc;
}
static xxh_u32 XXH32_avalanche(xxh_u32 hash)
{
    hash ^= hash >> 15;
    hash *= XXH_PRIME32_2;
    hash ^= hash >> 13;
    hash *= XXH_PRIME32_3;
    hash ^= hash >> 16;
    return hash;
}
#define XXH_get32bits(p) XXH_readLE32_align(p, align)
XXH_FORCE_INLINE void
XXH32_initAccs(xxh_u32 *acc, xxh_u32 seed)
{
    XXH_ASSERT(acc != NULL);
    acc[0] = seed + XXH_PRIME32_1 + XXH_PRIME32_2;
    acc[1] = seed + XXH_PRIME32_2;
    acc[2] = seed + 0;
    acc[3] = seed - XXH_PRIME32_1;
}
XXH_FORCE_INLINE const xxh_u8 *
XXH32_consumeLong(
    xxh_u32 *XXH_RESTRICT acc,
    xxh_u8 const *XXH_RESTRICT input,
    size_t len,
    XXH_alignment align
)
{
    const xxh_u8* const bEnd = input + len;
    const xxh_u8* const limit = bEnd - 15;
    XXH_ASSERT(acc != NULL);
    XXH_ASSERT(input != NULL);
    XXH_ASSERT(len >= 16);
    do {
        acc[0] = XXH32_round(acc[0], XXH_get32bits(input)); input += 4;
        acc[1] = XXH32_round(acc[1], XXH_get32bits(input)); input += 4;
        acc[2] = XXH32_round(acc[2], XXH_get32bits(input)); input += 4;
        acc[3] = XXH32_round(acc[3], XXH_get32bits(input)); input += 4;
    } while (input < limit);
    return input;
}
XXH_FORCE_INLINE XXH_PUREF xxh_u32
XXH32_mergeAccs(const xxh_u32 *acc)
{
    XXH_ASSERT(acc != NULL);
    return XXH_rotl32(acc[0], 1)  + XXH_rotl32(acc[1], 7)
         + XXH_rotl32(acc[2], 12) + XXH_rotl32(acc[3], 18);
}
static XXH_PUREF xxh_u32
XXH32_finalize(xxh_u32 hash, const xxh_u8* ptr, size_t len, XXH_alignment align)
{
#define XXH_PROCESS1 do {                             \
    hash += (*ptr++) * XXH_PRIME32_5;                 \
    hash = XXH_rotl32(hash, 11) * XXH_PRIME32_1;      \
} while (0)
#define XXH_PROCESS4 do {                             \
    hash += XXH_get32bits(ptr) * XXH_PRIME32_3;       \
    ptr += 4;                                         \
    hash  = XXH_rotl32(hash, 17) * XXH_PRIME32_4;     \
} while (0)
    if (ptr==NULL) XXH_ASSERT(len == 0);
    len &= 15;
    while (len >= 4) {
        XXH_PROCESS4;
        len -= 4;
    }
    while (len > 0) {
        XXH_PROCESS1;
        --len;
    }
    return XXH32_avalanche(hash);
}
XXH_FORCE_INLINE XXH_PUREF xxh_u32
XXH32_endian_align(const xxh_u8* input, size_t len, xxh_u32 seed, XXH_alignment align)
{
    xxh_u32 h32;
    if (input==NULL) XXH_ASSERT(len == 0);
    if (len>=16) {
        xxh_u32 acc[4];
        XXH32_initAccs(acc, seed);
        input = XXH32_consumeLong(acc, input, len, align);
        h32 = XXH32_mergeAccs(acc);
    } else {
        h32  = seed + XXH_PRIME32_5;
    }
    h32 += (xxh_u32)len;
    return XXH32_finalize(h32, input, len&15, align);
}
XXH_PUBLIC_API XXH32_hash_t XXH32 (const void* input, size_t len, XXH32_hash_t seed)
{
    if (XXH_FORCE_ALIGN_CHECK) {
        if ((((size_t)input) & 3) == 0) {
            return XXH32_endian_align((const xxh_u8*)input, len, seed, XXH_aligned);
    }   }
    return XXH32_endian_align((const xxh_u8*)input, len, seed, XXH_unaligned);
}
int main() {
  int q; if (scanf("%d", &q) != 1) return 1;
  static char buf[1 << 20];
  for (int i = 0; i < q; i++) { if (scanf("%s", buf) != 1) return 1; printf("%u\n", XXH32(buf, strlen(buf), 0)); }
  return 0;
}
