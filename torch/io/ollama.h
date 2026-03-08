#pragma once

#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cstdlib>

#ifdef _WIN32
#include <windows.h>
#endif

namespace torch {
namespace io {
namespace ollama {

// ============================================================================
// Resolve Ollama model name → GGUF blob file path
//
// Directory structure:
//   ~/.ollama/models/manifests/registry.ollama.ai/library/<name>/<tag>
//   ~/.ollama/models/blobs/<sha256-digest>
//
// Manifest is JSON with layers, the model layer has:
//   "mediaType": "application/vnd.ollama.image.model"
//   "digest": "sha256:..."
// ============================================================================

inline std::string get_ollama_home() {
    // Check OLLAMA_MODELS env var first
    const char* env = std::getenv("OLLAMA_MODELS");
    if (env && env[0]) {
        return std::string(env);
    }

#ifdef _WIN32
    // Windows: %USERPROFILE%\.ollama\models
    const char* home = std::getenv("USERPROFILE");
    if (!home) home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("Cannot determine home directory (USERPROFILE/HOME not set)");
    }
    return std::string(home) + "\\.ollama\\models";
#else
    const char* home = std::getenv("HOME");
    if (!home) {
        throw std::runtime_error("Cannot determine home directory (HOME not set)");
    }
    return std::string(home) + "/.ollama/models";
#endif
}

// Simple JSON string value extractor (no dependency on JSON library)
// Finds "key": "value" and returns value
inline std::string json_get_string(const std::string& json, const std::string& key) {
    std::string search = "\"" + key + "\"";
    size_t pos = json.find(search);
    if (pos == std::string::npos) return "";

    // Find the colon after the key
    pos = json.find(':', pos + search.size());
    if (pos == std::string::npos) return "";

    // Skip whitespace
    pos = json.find('"', pos + 1);
    if (pos == std::string::npos) return "";

    // Extract string value
    size_t start = pos + 1;
    size_t end = json.find('"', start);
    if (end == std::string::npos) return "";

    return json.substr(start, end - start);
}

// Find the model layer digest from a manifest JSON
inline std::string find_model_digest(const std::string& manifest_json) {
    // We need to find the layer with mediaType "application/vnd.ollama.image.model"
    // and extract its digest.
    //
    // Manifest structure:
    // { "layers": [ { "mediaType": "...", "digest": "sha256:...", "size": N }, ... ] }

    const std::string model_type = "application/vnd.ollama.image.model";
    size_t pos = 0;

    while (true) {
        // Find next "mediaType"
        pos = manifest_json.find("\"mediaType\"", pos);
        if (pos == std::string::npos) break;

        // Check if this is the model layer
        size_t type_start = manifest_json.find('"', pos + 11);  // skip past "mediaType"
        if (type_start == std::string::npos) break;
        type_start = manifest_json.find('"', type_start + 1);  // after colon
        if (type_start == std::string::npos) break;

        // Actually, let's find the value more robustly
        size_t colon = manifest_json.find(':', pos);
        if (colon == std::string::npos) break;
        size_t val_start = manifest_json.find('"', colon + 1);
        if (val_start == std::string::npos) break;
        val_start++;
        size_t val_end = manifest_json.find('"', val_start);
        if (val_end == std::string::npos) break;

        std::string media_type = manifest_json.substr(val_start, val_end - val_start);

        if (media_type == model_type) {
            // Found the model layer. Now find the digest in the same object.
            // Look for "digest" nearby (within the same layer object)
            size_t digest_pos = manifest_json.find("\"digest\"", pos);
            if (digest_pos != std::string::npos) {
                size_t d_colon = manifest_json.find(':', digest_pos + 8);
                if (d_colon != std::string::npos) {
                    size_t d_start = manifest_json.find('"', d_colon + 1);
                    if (d_start != std::string::npos) {
                        d_start++;
                        size_t d_end = manifest_json.find('"', d_start);
                        if (d_end != std::string::npos) {
                            return manifest_json.substr(d_start, d_end - d_start);
                        }
                    }
                }
            }
        }

        pos = val_end + 1;
    }

    return "";
}

// ============================================================================
// Resolve model name to GGUF blob path
// Supports formats: "gemma3:4b", "qwen3:4b", "gemma3" (defaults to "latest")
// ============================================================================

inline std::string resolve_model(const std::string& model_name) {
    // Parse name:tag
    std::string name = model_name;
    std::string tag = "latest";
    size_t colon = model_name.find(':');
    if (colon != std::string::npos) {
        name = model_name.substr(0, colon);
        tag = model_name.substr(colon + 1);
    }

    std::string ollama_home = get_ollama_home();

    // Read manifest
#ifdef _WIN32
    std::string manifest_path = ollama_home + "\\manifests\\registry.ollama.ai\\library\\" + name + "\\" + tag;
#else
    std::string manifest_path = ollama_home + "/manifests/registry.ollama.ai/library/" + name + "/" + tag;
#endif

    std::ifstream manifest_file(manifest_path);
    if (!manifest_file) {
        throw std::runtime_error(
            "Ollama: Model not found: " + model_name +
            "\nExpected manifest at: " + manifest_path +
            "\nMake sure the model is pulled: ollama pull " + model_name
        );
    }

    std::stringstream ss;
    ss << manifest_file.rdbuf();
    std::string manifest_json = ss.str();
    manifest_file.close();

    // Find model layer digest
    std::string digest = find_model_digest(manifest_json);
    if (digest.empty()) {
        throw std::runtime_error(
            "Ollama: Could not find model layer in manifest for: " + model_name
        );
    }

    // Convert digest to blob path
    // digest format: "sha256:abc123..."
    // blob filename: "sha256-abc123..."
    std::string blob_name = digest;
    size_t digest_colon = blob_name.find(':');
    if (digest_colon != std::string::npos) {
        blob_name[digest_colon] = '-';
    }

#ifdef _WIN32
    std::string blob_path = ollama_home + "\\blobs\\" + blob_name;
#else
    std::string blob_path = ollama_home + "/blobs/" + blob_name;
#endif

    // Verify blob exists
    std::ifstream blob_check(blob_path, std::ios::binary);
    if (!blob_check) {
        throw std::runtime_error(
            "Ollama: Blob file not found: " + blob_path +
            "\nDigest: " + digest
        );
    }

    // Get file size
    blob_check.seekg(0, std::ios::end);
    auto size = blob_check.tellg();
    blob_check.close();

    std::cout << "[Ollama] Resolved '" << model_name << "' → " << blob_path << std::endl;
    std::cout << "[Ollama] File size: " << (size / 1024 / 1024) << " MB" << std::endl;

    return blob_path;
}

// ============================================================================
// List available Ollama models
// ============================================================================

inline void list_models() {
    std::string ollama_home;
    try {
        ollama_home = get_ollama_home();
    } catch (...) {
        std::cout << "Cannot determine Ollama models directory." << std::endl;
        return;
    }

#ifdef _WIN32
    std::string manifests_dir = ollama_home + "\\manifests\\registry.ollama.ai\\library";
#else
    std::string manifests_dir = ollama_home + "/manifests/registry.ollama.ai/library";
#endif

    std::cout << "[Ollama] Models directory: " << manifests_dir << std::endl;
    std::cout << "[Ollama] Use 'ollama list' to see available models." << std::endl;
}

} // namespace ollama
} // namespace io
} // namespace torch
