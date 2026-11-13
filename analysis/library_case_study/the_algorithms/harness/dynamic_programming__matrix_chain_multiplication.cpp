// TheAlgorithms/C-Plus-Plus dynamic_programming/matrix_chain_multiplication.cpp (verbatim, tests and main removed).
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <climits>
#include <iostream>
using namespace std;

#define MAX 10

// dp table to store the solution for already computed sub problems
int dp[MAX][MAX];

// Function to find the most efficient way to multiply the given sequence of
// matrices
int MatrixChainMultiplication(int dim[], int i, int j) {
    // base case: one matrix
    if (j <= i + 1)
        return 0;

    // stores minimum number of scalar multiplications (i.e., cost)
    // needed to compute the matrix M[i+1]...M[j] = M[i..j]
    int min = INT_MAX;

    // if dp[i][j] is not calculated (calculate it!!)

    if (dp[i][j] == 0) {
        // take the minimum over each possible position at which the
        // sequence of matrices can be split

        for (int k = i + 1; k <= j - 1; k++) {
            // recur for M[i+1]..M[k] to get a i x k matrix
            int cost = MatrixChainMultiplication(dim, i, k);

            // recur for M[k+1]..M[j] to get a k x j matrix
            cost += MatrixChainMultiplication(dim, k, j);

            // cost to multiply two (i x k) and (k x j) matrix
            cost += dim[i] * dim[k] * dim[j];

            if (cost < min)
                min = cost;  // store the minimum cost
        }
        dp[i][j] = min;
    }

    // return min cost to multiply M[j+1]..M[j]
    return dp[i][j];
}

// main function

int main() {
  int q; if (scanf("%d", &q) != 1) return 1;
  for (int t = 0; t < q; t++) {
    int n; scanf("%d", &n); int dim[MAX]; for (int i = 0; i < n; i++) scanf("%d", &dim[i]);
    memset(dp, 0, sizeof dp);
    printf("%d\n", MatrixChainMultiplication(dim, 0, n - 1));
  }
  return 0;
}
