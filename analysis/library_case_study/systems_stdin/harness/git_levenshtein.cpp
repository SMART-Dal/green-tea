// git levenshtein.c: levenshtein() (verbatim; ALLOC_ARRAY expanded to malloc).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <string>
#define ALLOC_ARRAY(x, alloc) (x) = (int *)malloc(sizeof(*(x)) * (alloc))
int levenshtein(const char *string1, const char *string2,
		int w, int s, int a, int d)
{
	int len1 = strlen(string1), len2 = strlen(string2);
	int *row0, *row1, *row2;
	int i, j;
	ALLOC_ARRAY(row0, len2 + 1);
	ALLOC_ARRAY(row1, len2 + 1);
	ALLOC_ARRAY(row2, len2 + 1);
	for (j = 0; j <= len2; j++)
		row1[j] = j * a;
	for (i = 0; i < len1; i++) {
		int *dummy;
		row2[0] = (i + 1) * d;
		for (j = 0; j < len2; j++) {
			/* substitution */
			row2[j + 1] = row1[j] + s * (string1[i] != string2[j]);
			/* swap */
			if (i > 0 && j > 0 && string1[i - 1] == string2[j] &&
					string1[i] == string2[j - 1] &&
					row2[j + 1] > row0[j - 1] + w)
				row2[j + 1] = row0[j - 1] + w;
			/* deletion */
			if (row2[j + 1] > row1[j + 1] + d)
				row2[j + 1] = row1[j + 1] + d;
			/* insertion */
			if (row2[j + 1] > row2[j] + a)
				row2[j + 1] = row2[j] + a;
		}
		dummy = row0;
		row0 = row1;
		row1 = row2;
		row2 = dummy;
	}
	i = row1[len2];
	free(row0);
	free(row1);
	free(row2);
	return i;
}
int main() {
  int q; if (scanf("%d", &q) != 1) return 1;
  static char a[1 << 16], b[1 << 16];
  for (int i = 0; i < q; i++) { if (scanf("%s %s", a, b) != 2) return 1; printf("%d\n", levenshtein(a, b, 0, 2, 1, 3)); }
  return 0;
}
