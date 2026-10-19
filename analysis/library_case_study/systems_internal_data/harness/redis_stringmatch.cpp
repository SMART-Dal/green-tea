// Redis src/util.c: stringmatchlen glob matcher (verbatim; variable `not` renamed `negate`, a C++ keyword).
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>
static int stringmatchlen_impl(const char *pattern, int patternLen,
        const char *string, int stringLen, int nocase, int *skipLongerMatches, int nesting)
{
    /* Protection against abusive patterns. */
    if (nesting > 1000) return 0;

    while(patternLen && stringLen) {
        switch(pattern[0]) {
        case '*':
            while (patternLen && pattern[1] == '*') {
                pattern++;
                patternLen--;
            }
            if (patternLen == 1)
                return 1; /* match */
            while(stringLen) {
                if (stringmatchlen_impl(pattern+1, patternLen-1,
                            string, stringLen, nocase, skipLongerMatches, nesting+1))
                    return 1; /* match */
                if (*skipLongerMatches)
                    return 0; /* no match */
                string++;
                stringLen--;
            }
            /* There was no match for the rest of the pattern starting
             * from anywhere in the rest of the string. If there were
             * any '*' earlier in the pattern, we can terminate the
             * search early without trying to match them to longer
             * substrings. This is because a longer match for the
             * earlier part of the pattern would require the rest of the
             * pattern to match starting later in the string, and we
             * have just determined that there is no match for the rest
             * of the pattern starting from anywhere in the current
             * string. */
            *skipLongerMatches = 1;
            return 0; /* no match */
            break;
        case '?':
            string++;
            stringLen--;
            break;
        case '[':
        {
            int negate, match;

            pattern++;
            patternLen--;
            negate = patternLen && pattern[0] == '^';
            if (negate) {
                pattern++;
                patternLen--;
            }
            match = 0;
            while(1) {
                if (patternLen >= 2 && pattern[0] == '\\') {
                    pattern++;
                    patternLen--;
                    if (pattern[0] == string[0])
                        match = 1;
                } else if (patternLen == 0) {
                    pattern--;
                    patternLen++;
                    break;
                } else if (pattern[0] == ']') {
                    break;
                } else if (patternLen >= 3 && pattern[1] == '-') {
                    int start = pattern[0];
                    int end = pattern[2];
                    int c = string[0];
                    if (start > end) {
                        int t = start;
                        start = end;
                        end = t;
                    }
                    if (nocase) {
                        start = tolower(start);
                        end = tolower(end);
                        c = tolower(c);
                    }
                    pattern += 2;
                    patternLen -= 2;
                    if (c >= start && c <= end)
                        match = 1;
                } else {
                    if (!nocase) {
                        if (pattern[0] == string[0])
                            match = 1;
                    } else {
                        if (tolower((int)pattern[0]) == tolower((int)string[0]))
                            match = 1;
                    }
                }
                pattern++;
                patternLen--;
            }
            if (negate)
                match = !match;
            if (!match)
                return 0; /* no match */
            string++;
            stringLen--;
            break;
        }
        case '\\':
            if (patternLen >= 2) {
                pattern++;
                patternLen--;
            }
            /* fall through */
        default:
            if (!nocase) {
                if (pattern[0] != string[0])
                    return 0; /* no match */
            } else {
                if (tolower((int)pattern[0]) != tolower((int)string[0]))
                    return 0; /* no match */
            }
            string++;
            stringLen--;
            break;
        }
        pattern++;
        patternLen--;
        if (stringLen == 0) {
            while(patternLen && *pattern == '*') {
                pattern++;
                patternLen--;
            }
            break;
        }
    }
    if (patternLen == 0 && stringLen == 0)
        return 1;
    return 0;
}

int stringmatchlen(const char *pattern, int patternLen,
        const char *string, int stringLen, int nocase) {
    int skipLongerMatches = 0;
    return stringmatchlen_impl(pattern,patternLen,string,stringLen,nocase,&skipLongerMatches,0);
}
static uint64_t st;
static inline uint64_t rnd() { st ^= st << 13; st ^= st >> 7; st ^= st << 17; return st; }
int main() {
  int m, len; unsigned long long seed;
  if (scanf("%d %d %llu", &m, &len, &seed) != 3) return 1;
  st = seed | 1;
  const char* alpha = "abcab*?[]";
  std::vector<std::string> pats(m), strs(m);
  for (int i = 0; i < m; i++) {
    pats[i].resize(1 + rnd() % 12); for (auto& c : pats[i]) c = alpha[rnd() % 9];
    strs[i].resize(len / 2 + rnd() % len); for (auto& c : strs[i]) c = 'a' + rnd() % 3;
  }
  long long hits = 0;
  for (int i = 0; i < m; i++) for (int j = 0; j < m; j++)
    hits = hits * 3 + stringmatchlen(pats[i].data(), (int)pats[i].size(), strs[j].data(), (int)strs[j].size(), (i + j) & 1);
  printf("%lld\n", hits);
  return 0;
}
