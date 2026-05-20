/* Runtime probe — actually link и run EML cblas_sgemm. Если libeml*.so
 * помечен под несовместимую ISA, dynamic loader падает на exec — try_run
 * получит ненулевой exit code и CMake переключится на OpenBLAS fallback.
 *
 * См. CMakeLists.txt: BLAS strategy.
 */
extern "C" void cblas_sgemm(int, int, int, int, int, int,
                            float, const float*, int,
                            const float*, int, float, float*, int);

int main(void) {
    float a[4] = {1, 2, 3, 4};
    float b[4] = {5, 6, 7, 8};
    float c[4] = {0, 0, 0, 0};
    /* CblasRowMajor=101, CblasNoTrans=111 */
    cblas_sgemm(101, 111, 111, 2, 2, 2, 1.0f, a, 2, b, 2, 0.0f, c, 2);
    return (c[0] != 0.0f) ? 0 : 1;
}
