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
        if (!in) throw std::runtime_error("cannot open linear_model_params.txt");

        double w = 0.0;
        double b = 0.0;

        in >> w;
        in >> b;

        if (!in) throw std::runtime_error("cannot parse w and b");

        const double x = 4.0;
        const double expected = 9.0;
        const double y_pred = x * w + b;
        const double abs_error = std::fabs(y_pred - expected);

        if (abs_error > 0.01)
        {
            throw std::runtime_error("prediction is not close enough to 9");
        }

        std::cout << "PromeTorch linear model inference smoke OK" << std::endl;
        std::cout << "loaded w = " << std::setprecision(17) << w << std::endl;
        std::cout << "loaded b = " << std::setprecision(17) << b << std::endl;
        std::cout << "x = " << x << std::endl;
        std::cout << "y_pred = " << std::setprecision(17) << y_pred << std::endl;
        std::cout << "expected = " << expected << std::endl;
        std::cout << "abs_error = " << abs_error << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "PromeTorch linear model inference smoke FAILED: " << e.what() << std::endl;
        return 1;
    }
}
