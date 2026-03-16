// Simple test - just return factorial of 10
int fact(int n) {
    if (n == 1) return 1;
    return fact(n - 1) * n;
}

int __main() {
    int res = fact(10);
    return res;  // Should return 3628800
}
