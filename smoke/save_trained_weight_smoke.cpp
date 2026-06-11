#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

int main() {
    float w = 1.0f;
    const float target = 5.0f;
    const float lr = 0.1f;
    const int steps = 20;

    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for (int i = 0; i < steps; ++i) {
        const float diff = w - target;
        const float loss = diff * diff;
        const float grad = 2.0f * diff;

        if (i == 0) {
            first_loss = loss;
        }

        last_loss = loss;
        w = w - lr * grad;
    }

    const std::filesystem::path out_path =
        std::filesystem::path("build") / "smoke" / "trained_weight.txt";

    std::filesystem::create_directories(out_path.parent_path());

    std::ofstream out(out_path, std::ios::trunc);
    if (!out) {
        std::cerr << "FAILED: cannot open output file: " << out_path.string() << "\n";
        return 1;
    }

    out << std::setprecision(9) << w << "\n";
    out.close();

    if (!out) {
        std::cerr << "FAILED: write failed: " << out_path.string() << "\n";
        return 1;
    }

    std::cout << "PromeTorch trained weight save smoke OK\n";
    std::cout << "first_loss = " << first_loss << "\n";
    std::cout << "last_loss  = " << last_loss << "\n";
    std::cout << "final w    = " << std::setprecision(9) << w << "\n";
    std::cout << "saved file = " << out_path.string() << "\n";

    return 0;
}
