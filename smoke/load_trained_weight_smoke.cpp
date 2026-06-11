#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
    const std::filesystem::path in_path =
        std::filesystem::path("build") / "smoke" / "trained_weight.txt";

    std::ifstream in(in_path);
    if (!in) {
        std::cerr << "FAILED: cannot open input file: " << in_path.string() << "\n";
        return 1;
    }

    double w = 0.0;
    in >> w;

    if (!in) {
        std::cerr << "FAILED: cannot parse trained weight from file\n";
        return 1;
    }

    if (!std::isfinite(w)) {
        std::cerr << "FAILED: trained weight is not finite\n";
        return 1;
    }

    const double target = 5.0;
    const double abs_error = std::abs(w - target);

    std::cout << "loaded_w  = " << std::setprecision(17) << w << "\n";
    std::cout << "target    = " << target << "\n";
    std::cout << "abs_error = " << abs_error << "\n";

    if (w < 4.90 || w > 5.01) {
        std::cerr << "FAILED: loaded weight is outside expected range [4.90, 5.01]\n";
        return 1;
    }

    std::cout << "PromeTorch trained weight load smoke OK\n";
    return 0;
}
