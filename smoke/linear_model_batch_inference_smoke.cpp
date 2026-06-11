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

        const double xs[] = {0.0, 1.0, 2.0, 3.0, 4.0};
        const double expected[] = {1.0, 3.0, 5.0, 7.0, 9.0};

        const double max_allowed_error = 0.01;
        double max_error = 0.0;

        std::cout << "PromeTorch linear model batch inference smoke" << std::endl;
        std::cout << "loaded w = " << std::setprecision(17) << w << std::endl;
        std::cout << "loaded b = " << std::setprecision(17) << b << std::endl;

        for (int i = 0; i < 5; ++i)
        {
            const double y_pred = xs[i] * w + b;
            const double abs_error = std::fabs(y_pred - expected[i]);

            if (abs_error > max_error)
            {
                max_error = abs_error;
            }

            std::cout
                << "x = " << xs[i]
                << " | y_pred = " << std::setprecision(17) << y_pred
                << " | expected = " << expected[i]
                << " | abs_error = " << abs_error
                << std::endl;

            if (abs_error > max_allowed_error)
            {
                throw std::runtime_error("batch inference error is too high");
            }
        }

        std::cout << "max_error = " << max_error << std::endl;
        std::cout << "PromeTorch linear model batch inference smoke OK" << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "PromeTorch linear model batch inference smoke FAILED: " << e.what() << std::endl;
        return 1;
    }
}
