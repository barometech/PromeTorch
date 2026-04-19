// torch/vision/vision.h
// torchvision-compatible vision package for PromeTorch.
//
// This header:
//   1. Defines torch::vision::ImageFolder (dataset that walks class-subdirs
//      and loads images via either bundled stb_image.h or built-in PPM/PGM).
//   2. Aggregates the rest of the vision package by including
//        torch/vision/transforms.h  (Resize, CenterCrop, RandomCrop,
//                                    RandomHorizontalFlip, Normalize,
//                                    ToTensor, Compose)
//        torch/vision/models.h      (MobileNetV2)
//
// Layout for ImageFolder:
//   root/
//     class_a/img001.ppm
//     class_a/img002.ppm
//     class_b/img003.ppm
//
// Loaded sample:
//   data:   uint8 HWC tensor [H,W,C]    (run ToTensor() to get float CHW [0,1])
//   target: int64 scalar tensor (class index, sorted lexicographically)
//
// Image loading backend:
//   - PPM (P6 RGB) and PGM (P5 grayscale) — always supported, pure C++ stdlib.
//   - JPG / PNG / BMP / TGA — only if bundled third_party/stb_image.h is on
//     the include path AND -DPT_USE_STB_IMAGE is defined. STB is pure C and
//     compiles cleanly under LCC on Elbrus.
//
// CPU-only, header-only.
#pragma once

#include "torch/vision/transforms.h"
#include "torch/vision/models.h"

#include "torch/data/dataset.h"
#include "aten/src/ATen/ATen.h"

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(PT_USE_STB_IMAGE)
  #include "third_party/stb_image.h"
#endif

#if defined(__has_include)
  #if __has_include(<filesystem>)
    #include <filesystem>
    #define PT_VISION_HAS_FS 1
  #endif
#endif

#if !defined(PT_VISION_HAS_FS)
  #include <dirent.h>
  #include <sys/stat.h>
#endif

namespace torch {
namespace vision {

using at::Tensor;
using torch::data::Dataset;
using torch::data::Example;

namespace detail {

// Skip whitespace and PNM comments ('# ...' until newline).
inline void skip_ws_and_comments(std::istream& f) {
    while (true) {
        int c = f.peek();
        if (c == EOF) return;
        if (std::isspace(c)) { f.get(); continue; }
        if (c == '#') { std::string dummy; std::getline(f, dummy); continue; }
        return;
    }
}

// Load binary PPM (P6, RGB) or PGM (P5, gray). Returns HWC uint8 tensor.
inline Tensor load_pnm(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("ImageFolder: cannot open " + path);

    std::string magic; f >> magic;
    if (magic != "P5" && magic != "P6") {
        throw std::runtime_error("ImageFolder: unsupported PNM magic '" + magic +
                                 "' in " + path + " (only binary P5/P6)");
    }
    int channels = (magic == "P6") ? 3 : 1;

    int width = 0, height = 0, maxval = 0;
    skip_ws_and_comments(f); f >> width;
    skip_ws_and_comments(f); f >> height;
    skip_ws_and_comments(f); f >> maxval;
    if (!f || width <= 0 || height <= 0 || maxval <= 0)
        throw std::runtime_error("ImageFolder: bad PNM header in " + path);
    f.get();  // consume single whitespace after maxval
    if (maxval > 255)
        throw std::runtime_error("ImageFolder: 16-bit PNM not supported (" + path + ")");

    const int64_t total = static_cast<int64_t>(width) * height * channels;
    Tensor img = at::empty({height, width, channels},
        at::TensorOptions().dtype(c10::ScalarType::Byte));
    f.read(reinterpret_cast<char*>(img.mutable_data_ptr<uint8_t>()), total);
    if (!f) throw std::runtime_error("ImageFolder: short read in " + path);
    return img;
}

#if defined(PT_USE_STB_IMAGE)
inline Tensor load_stb(const std::string& path) {
    int w, h, ch;
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &ch, 0);
    if (!data) throw std::runtime_error("ImageFolder: stbi_load failed for " + path);
    Tensor img = at::empty({h, w, ch},
        at::TensorOptions().dtype(c10::ScalarType::Byte));
    std::memcpy(img.mutable_data_ptr<uint8_t>(), data,
                static_cast<size_t>(w) * h * ch);
    stbi_image_free(data);
    return img;
}
#endif

inline std::string to_lower(std::string s) {
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

inline bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

inline bool is_supported_image(const std::string& path) {
    std::string lo = to_lower(path);
    if (ends_with(lo, ".ppm") || ends_with(lo, ".pgm") || ends_with(lo, ".pnm"))
        return true;
#if defined(PT_USE_STB_IMAGE)
    if (ends_with(lo, ".jpg") || ends_with(lo, ".jpeg") ||
        ends_with(lo, ".png") || ends_with(lo, ".bmp") ||
        ends_with(lo, ".tga"))
        return true;
#endif
    return false;
}

inline Tensor load_image(const std::string& path) {
    std::string lo = to_lower(path);
    if (ends_with(lo, ".ppm") || ends_with(lo, ".pgm") || ends_with(lo, ".pnm"))
        return load_pnm(path);
#if defined(PT_USE_STB_IMAGE)
    return load_stb(path);
#else
    throw std::runtime_error(
        "ImageFolder: file '" + path + "' not supported. "
        "Bundle third_party/stb_image.h and compile with -DPT_USE_STB_IMAGE "
        "for JPG/PNG/BMP/TGA, or convert to PPM/PGM.");
#endif
}

#if defined(PT_VISION_HAS_FS)
inline std::vector<std::string> list_dir(const std::string& path, bool dirs_only) {
    std::vector<std::string> out;
    std::error_code ec;
    for (auto& entry : std::filesystem::directory_iterator(path, ec)) {
        if (ec) break;
        if (dirs_only ? entry.is_directory(ec) : entry.is_regular_file(ec))
            out.push_back(entry.path().filename().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}
inline std::string path_join(const std::string& a, const std::string& b) {
    return (std::filesystem::path(a) / b).string();
}
#else
inline std::vector<std::string> list_dir(const std::string& path, bool dirs_only) {
    std::vector<std::string> out;
    DIR* d = opendir(path.c_str());
    if (!d) return out;
    while (struct dirent* e = readdir(d)) {
        std::string name = e->d_name;
        if (name == "." || name == "..") continue;
        std::string full = path + "/" + name;
        struct stat st{};
        if (stat(full.c_str(), &st) != 0) continue;
        bool is_dir = S_ISDIR(st.st_mode);
        if (dirs_only ? is_dir : !is_dir) out.push_back(name);
    }
    closedir(d);
    std::sort(out.begin(), out.end());
    return out;
}
inline std::string path_join(const std::string& a, const std::string& b) {
    if (!a.empty() && a.back() == '/') return a + b;
    return a + "/" + b;
}
#endif

} // namespace detail

// ---------------------------------------------------------------------------
// ImageFolder dataset.
// ---------------------------------------------------------------------------
class ImageFolder : public Dataset<Tensor, Tensor> {
public:
    using ExampleType = Example<Tensor, Tensor>;
    using Loader = std::function<Tensor(const std::string&)>;
    using TransformFn = std::function<Tensor(const Tensor&)>;

    explicit ImageFolder(const std::string& root,
                         TransformFn transform = nullptr,
                         Loader loader = nullptr)
        : root_(root)
        , transform_(std::move(transform))
        , loader_(loader ? std::move(loader) : Loader(detail::load_image)) {
        scan_();
    }

    ExampleType get(size_t index) override {
        PT_CHECK_MSG(index < samples_.size(), "ImageFolder index out of range");
        const auto& s = samples_[index];
        Tensor img = loader_(s.first);
        if (transform_) img = transform_(img);
        Tensor target = at::empty({}, at::TensorOptions().dtype(c10::ScalarType::Long));
        *target.mutable_data_ptr<int64_t>() = static_cast<int64_t>(s.second);
        return ExampleType(img, target);
    }

    size_t size() const override { return samples_.size(); }

    const std::vector<std::string>& classes() const { return classes_; }
    const std::vector<std::pair<std::string, int>>& samples() const { return samples_; }

private:
    void scan_() {
        classes_ = detail::list_dir(root_, /*dirs_only=*/true);
        if (classes_.empty()) {
            throw std::runtime_error(
                "ImageFolder: no class subdirectories found in " + root_);
        }
        for (size_t ci = 0; ci < classes_.size(); ++ci) {
            const std::string& cls = classes_[ci];
            std::string cls_dir = detail::path_join(root_, cls);
            auto files = detail::list_dir(cls_dir, /*dirs_only=*/false);
            for (const auto& f : files) {
                if (!detail::is_supported_image(f)) continue;
                samples_.emplace_back(detail::path_join(cls_dir, f),
                                      static_cast<int>(ci));
            }
        }
        if (samples_.empty()) {
            throw std::runtime_error(
                "ImageFolder: no supported images found under " + root_);
        }
    }

    std::string root_;
    TransformFn transform_;
    Loader loader_;
    std::vector<std::string> classes_;
    std::vector<std::pair<std::string, int>> samples_;
};

} // namespace vision
} // namespace torch
