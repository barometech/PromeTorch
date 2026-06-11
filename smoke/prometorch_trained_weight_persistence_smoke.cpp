#include <iostream>
#include <stdexcept>
#include <string>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>

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

        float w_value = 1.0f;
        const float target_value = 5.0f;
        const float lr = 0.1f;
        const int steps = 20;

        float first_loss = -1.0f;
        float last_loss = -1.0f;

        for (int step = 0; step < steps; ++step)
        {
            Tensor w = scalar_tensor(w_value, true);
            Tensor target = scalar_tensor(target_value, false);

            Tensor diff = sub_autograd(w, target);
            Tensor sq = mul_autograd(diff, diff);
            Tensor loss = sum_autograd(sq);

            float loss_value = loss.item<float>();

            if (step == 0)
            {
                first_loss = loss_value;
            }

            tensor_backward(loss);

            Tensor grad_w = get_grad(w);
            if (!grad_w.defined())
            {
                throw std::runtime_error("grad_w is not defined");
            }

            float grad_value = grad_w.data_ptr<float>()[0];
            w_value = w_value - lr * grad_value;
            last_loss = loss_value;
        }

        if (!(last_loss < first_loss))
        {
            throw std::runtime_error("loss did not decrease");
        }

        if (std::fabs(w_value - target_value) > 0.1f)
        {
            throw std::runtime_error("w is not close enough to target");
        }

        const std::filesystem::path out_path =
            std::filesystem::path("build") / "smoke" / "trained_weight.txt";

        std::filesystem::create_directories(out_path.parent_path());

        std::ofstream out(out_path, std::ios::trunc);
        if (!out)
        {
            throw std::runtime_error("cannot open trained_weight.txt for writing");
        }

        out << std::setprecision(9) << w_value << std::endl;

        std::cout << "PromeTorch trained weight persistence smoke OK" << std::endl;
        std::cout << "first_loss = " << first_loss << std::endl;
        std::cout << "last_loss = " << last_loss << std::endl;
        std::cout << "final w = " << std::setprecision(9) << w_value << std::endl;
        std::cout << "saved file = " << out_path.string() << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "PromeTorch trained weight persistence smoke FAILED: " << e.what() << std::endl;
        return 1;
    }
}
