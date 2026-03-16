// Correct test - NO infinite loop!
// This is how official RC Module examples work

int fact(int n) {
    if (n <= 1) return 1;
    return n * fact(n - 1);
}

int main() {
    int result = fact(10);  // 3628800
    return result;
}
