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

        float w0_value = 0.0f;
        float w1_value = 0.0f;
        float b_value = 0.0f;

        const float lr = 0.005f;
        const int max_epochs = 1000;
        const float early_stop_error = 0.005f;

        const int train_count = 6;
        const float x0s[train_count] = {1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 3.0f};
        const float x1s[train_count] = {0.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f};
        const float ys[train_count]  = {3.0f, -2.0f, 0.0f, 2.0f, -3.0f, 1.0f};

        float first_loss = -1.0f;
        float last_loss = -1.0f;
        float training_max_error = -1.0f;
        int stopped_epoch = -1;

        for (int epoch = 0; epoch < max_epochs; ++epoch)
        {
            Tensor w0 = scalar_tensor(w0_value, true);
            Tensor w1 = scalar_tensor(w1_value, true);
            Tensor b = scalar_tensor(b_value, true);

            Tensor total_loss;

            for (int i = 0; i < train_count; ++i)
            {
                Tensor x0 = scalar_tensor(x0s[i], false);
                Tensor x1 = scalar_tensor(x1s[i], false);
                Tensor y = scalar_tensor(ys[i], false);

                Tensor term0 = mul_autograd(w0, x0);
                Tensor term1 = mul_autograd(w1, x1);
                Tensor pred = add_autograd(add_autograd(term0, term1), b);
                Tensor diff = sub_autograd(pred, y);
                Tensor sq = mul_autograd(diff, diff);
                Tensor loss = sum_autograd(sq);

                if (i == 0)
                {
                    total_loss = loss;
                }
                else
                {
                    total_loss = add_autograd(total_loss, loss);
                }
            }

            const float batch_loss = total_loss.item<float>();

            if (epoch == 0)
            {
                first_loss = batch_loss;
            }

            last_loss = batch_loss;

            tensor_backward(total_loss);

            Tensor grad_w0 = get_grad(w0);
            Tensor grad_w1 = get_grad(w1);
            Tensor grad_b = get_grad(b);

            if (!grad_w0.defined()) throw std::runtime_error("grad_w0 is not defined");
            if (!grad_w1.defined()) throw std::runtime_error("grad_w1 is not defined");
            if (!grad_b.defined()) throw std::runtime_error("grad_b is not defined");

            w0_value = w0_value - lr * grad_w0.data_ptr<float>()[0];
            w1_value = w1_value - lr * grad_w1.data_ptr<float>()[0];
            b_value = b_value - lr * grad_b.data_ptr<float>()[0];

            training_max_error = 0.0f;

            for (int i = 0; i < train_count; ++i)
            {
                const float pred = x0s[i] * w0_value + x1s[i] * w1_value + b_value;
                const float abs_error = std::fabs(pred - ys[i]);

                if (abs_error > training_max_error)
                {
                    training_max_error = abs_error;
                }
            }

            if (training_max_error < early_stop_error)
            {
                stopped_epoch = epoch + 1;
                break;
            }
        }

        if (stopped_epoch < 0) throw std::runtime_error("early stopping did not trigger");
        if (!(last_loss < first_loss)) throw std::runtime_error("loss did not decrease");

        const std::filesystem::path out_path =
            std::filesystem::path("build") / "smoke" / "multifeature_linear_model_params.txt";

        std::filesystem::create_directories(out_path.parent_path());

        {
            std::ofstream out(out_path, std::ios::trunc);
            if (!out) throw std::runtime_error("cannot open params file for writing");

            out << std::setprecision(9) << w0_value << "\n";
            out << std::setprecision(9) << w1_value << "\n";
            out << std::setprecision(9) << b_value << "\n";
            out.close();

            if (!out) throw std::runtime_error("params file write failed");
        }

        double loaded_w0 = 0.0;
        double loaded_w1 = 0.0;
        double loaded_b = 0.0;

        {
            std::ifstream in(out_path);
            if (!in) throw std::runtime_error("cannot open params file for reading");

            in >> loaded_w0;
            in >> loaded_w1;
            in >> loaded_b;

            if (!in) throw std::runtime_error("cannot parse saved params");
        }

        const int test_count = 3;
        const double test_x0s[test_count] = {4.0, 0.0, 2.0};
        const double test_x1s[test_count] = {0.0, 4.0, 3.0};
        const double expected[test_count] = {9.0, -11.0, -4.0};

        double max_error = 0.0;

        for (int i = 0; i < test_count; ++i)
        {
            const double y_pred = test_x0s[i] * loaded_w0 + test_x1s[i] * loaded_w1 + loaded_b;
            const double abs_error = std::fabs(y_pred - expected[i]);

            if (abs_error > max_error)
            {
                max_error = abs_error;
            }

            if (abs_error > 0.01)
            {
                throw std::runtime_error("loaded params inference error is too high");
            }
        }

        std::cout << "PromeTorch multifeature params persistence smoke OK" << std::endl;
        std::cout << "first_loss = " << first_loss << std::endl;
        std::cout << "last_loss = " << last_loss << std::endl;
        std::cout << "stopped_epoch = " << stopped_epoch << std::endl;
        std::cout << "training_max_error = " << std::setprecision(9) << training_max_error << std::endl;
        std::cout << "saved w0 = " << std::setprecision(9) << w0_value << std::endl;
        std::cout << "saved w1 = " << std::setprecision(9) << w1_value << std::endl;
        std::cout << "saved b = " << std::setprecision(9) << b_value << std::endl;
        std::cout << "loaded w0 = " << std::setprecision(17) << loaded_w0 << std::endl;
        std::cout << "loaded w1 = " << std::setprecision(17) << loaded_w1 << std::endl;
        std::cout << "loaded b = " << std::setprecision(17) << loaded_b << std::endl;
        std::cout << "max_error = " << std::setprecision(17) << max_error << std::endl;
        std::cout << "saved file = " << out_path.string() << std::endl;

        return 0;
    }
    catch (const std::exception &e)
    {
        std::cerr << "PromeTorch multifeature params persistence smoke FAILED: " << e.what() << std::endl;
        return 1;
    }
}
