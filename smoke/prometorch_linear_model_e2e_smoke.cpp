#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

#include "aten/src/ATen/ATen.h"
#include "torch/csrc/autograd/autograd.h"

static at::Tensor get_grad(const at::Tensor &tensor)
{
    auto *raw_meta = tensor.autograd_meta();
    if (raw_meta && raw_meta->grad_)
    {
        return at::Tensor(raw_meta->grad_);
    }
    return at::Tensor();
}

static at::Tensor scalar_tensor(float value, bool requires_grad)
{
    at::Tensor t = torch::ones({1});
    t.mutable_data_ptr<float>()[0] = value;
    t.set_requires_grad(requires_grad);
    return t;
}

int main()
{
    try
    {
        using namespace at;
        using namespace torch::autograd;

        float w_value = 0.0f;
        float b_value = 0.0f;

        const float lr = 0.01f;
        const int epochs = 200;

        const float xs[] = {0.0f, 1.0f, 2.0f, 3.0f};
        const float ys[] = {1.0f, 3.0f, 5.0f, 7.0f};

        float first_loss = -1.0f;
        float last_loss = -1.0f;

        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float epoch_loss = 0.0f;

            for (int i = 0; i < 4; ++i)
            {
                Tensor w = scalar_tensor(w_value, true);
                Tensor b = scalar_tensor(b_value, true);
                Tensor x = scalar_tensor(xs[i], false);
                Tensor y = scalar_tensor(ys[i], false);

                Tensor pred = add_autograd(mul_autograd(w, x), b);
                Tensor diff = sub_autograd(pred, y);
                Tensor loss = sum_autograd(mul_autograd(diff, diff));

                epoch_loss += loss.item<float>();

                tensor_backward(loss);

                Tensor grad_w = get_grad(w);
                Tensor grad_b = get_grad(b);

                if (!grad_w.defined()) throw std::runtime_error("grad_w is not defined");
                if (!grad_b.defined()) throw std::runtime_error("grad_b is not defined");

                w_value -= lr * grad_w.data_ptr<float>()[0];
                b_value -= lr * grad_b.data_ptr<float>()[0];
            }

            if (epoch == 0) first_loss = epoch_loss;
            last_loss = epoch_loss;
        }

        if (!(last_loss < first_loss)) throw std::runtime_error("loss did not decrease");
        if (std::fabs(w_value - 2.0f) > 0.1f) throw std::runtime_error("w is not close enough to 2");
        if (std::fabs(b_value - 1.0f) > 0.1f) throw std::runtime_error("b is not close enough to 1");

        const std::filesystem::path path =
            std::filesystem::path("build") / "smoke" / "linear_model_e2e_params.txt";

        std::filesystem::create_directories(path.parent_path());

        {
            std::ofstream out(path, std::ios::trunc);
            if (!out) throw std::runtime_error("cannot write params file");
            out << std::setprecision(9) << w_value << "\n";
            out << std::setprecision(9) << b_value << "\n";
        }

        double loaded_w = 0.0;
        double loaded_b = 0.0;

        {
            std::ifstream in(path);
            if (!in) throw std::runtime_error("cannot read params file");
            in >> loaded_w;
            in >> loaded_b;
            if (!in) throw std::runtime_error("cannot parse params file");
        }

        const double x = 4.0;
        const double expected = 9.0;
        const double y_pred = x * loaded_w + loaded_b;
        const double abs_error = std::fabs(y_pred - expected);

        if (abs_error > 0.01) throw std::runtime_error("inference error is too high");

        std::cout << "PromeTorch linear model e2e smoke OK" << std::endl;
        std::cout << "first_loss = " << first_loss << std::endl;
        std::cout << "last_loss = " << last_loss << std::endl;
        std::cout << "saved w = " << std::setprecision(17) << w_value << std::endl;
        std::cout << "saved b = " << std::setprecision(17) << b_value << std::endl;
        std::cout << "loaded w = " << loaded_w << std::endl;
        std::cout << "loaded b = " << loaded_b << std::endl;
        std::cout << "x = " << x << std::endl;
        std::cout << "y_pred = " << y_pred << std::endl;
        std::cout << "expected = " << expected << std::endl;
        std::cout << "abs_error = " << abs_error << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "PromeTorch linear model e2e smoke FAILED: " << e.what() << std::endl;
        return 1;
    }
}
