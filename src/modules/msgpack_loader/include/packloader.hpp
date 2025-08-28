#pragma once

#include <filesystem>
#include <string>
#include <optional>
#include <vector>
#include <msgpack.hpp>

namespace trackie::loader {

class PackLoader {
public:
    explicit PackLoader(const std::filesystem::path& file_path);
    ~PackLoader();

    PackLoader(const PackLoader&) = delete;
    PackLoader& operator=(const PackLoader&) = delete;
    PackLoader(PackLoader&&) = delete;
    PackLoader& operator=(PackLoader&&) = delete;

    std::optional<std::string> getString(const std::string& key) const;
    std::optional<int64_t> getInt(const std::string& key) const;
    std::optional<bool> getBool(const std::string& key) const;

private:
    msgpack::object_handle* m_handle;
    std::vector<char> m_buffer;
};

} // namespace trackie::loader