#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main()
{
    try
    {
        const std::filesystem::path in_path =
            std::filesystem::path("build") / "smoke" / "linear_model_params.txt";

        std::ifstream in(in_path);
        if (!in) throw std::runtime_error("cannot open linear_model_params.txt for reading");

        double w = 0.0;
        double b = 0.0;

        in >> w;
        in >> b;

        if (!in) throw std::runtime_error("cannot parse w and b");
        if (!std::isfinite(w)) throw std::runtime_error("w is not finite");
        if (!std::isfinite(b)) throw std::runtime_error("b is not finite");

        if (std::fabs(w - 2.0) > 0.1) throw std::runtime_error("w is not close enough to 2");
        if (std::fabs(b - 1.0) > 0.1) throw std::runtime_error("b is not close enough to 1");

        std::cout << "PromeTorch linear model params load smoke OK" << std::endl;
        std::cout << "loaded w = " << std::setprecision(17) << w << std::endl;
        std::cout << "loaded b = " << std::setprecision(17) << b << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "PromeTorch linear model params load smoke FAILED: " << e.what() << std::endl;
        return 1;
    }
}
