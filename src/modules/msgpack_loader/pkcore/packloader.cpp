#include "packloader.hpp"
#include <fstream>
#include <stdexcept>

namespace trackie::loader {

PackLoader::PackLoader(const std::filesystem::path& file_path)
    : m_handle(new msgpack::object_handle()) {
    if (!std::filesystem::exists(file_path)) {
        delete m_handle;
        throw std::runtime_error("Config file not found: " + file_path.string());
    }

    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        delete m_handle;
        throw std::runtime_error("Failed to open file: " + file_path.string());
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    m_buffer.resize(size);
    if (!file.read(m_buffer.data(), size)) {
        delete m_handle;
        throw std::runtime_error("Failed to read file content: " + file_path.string());
    }

    try {
        msgpack::unpack(*m_handle, m_buffer.data(), m_buffer.size());
    } catch (const msgpack::parse_error& e) {
        delete m_handle;
        throw std::runtime_error("Invalid MessagePack format in " + file_path.string() + ": " + e.what());
    }

    if (m_handle->get().type != msgpack::type::MAP) {
        delete m_handle;
        throw std::runtime_error("Config file root must be a map.");
    }
}

PackLoader::~PackLoader() {
    delete m_handle;
}

static const msgpack::object* find_value_by_key(const msgpack::object_handle* handle, const std::string& key) {
    if (!handle || handle->get().type != msgpack::type::MAP) {
        return nullptr;
    }
    const auto& map = handle->get().via.map;
    for (uint32_t i = 0; i < map.size; ++i) {
        const auto& map_key = map.ptr[i].key;
        if (map_key.type == msgpack::type::STR && map_key.as<std::string>() == key) {
            return &map.ptr[i].val;
        }
    }
    return nullptr;
}

std::optional<std::string> PackLoader::getString(const std::string& key) const {
    const msgpack::object* val_obj = find_value_by_key(m_handle, key);
    if (val_obj && val_obj->type == msgpack::type::STR) {
        return val_obj->as<std::string>();
    }
    return std::nullopt;
}

std::optional<int64_t> PackLoader::getInt(const std::string& key) const {
    const msgpack::object* val_obj = find_value_by_key(m_handle, key);
    if (val_obj && (val_obj->type == msgpack::type::POSITIVE_INTEGER || val_obj->type == msgpack::type::NEGATIVE_INTEGER)) {
        return val_obj->as<int64_t>();
    }
    return std::nullopt;
}

std::optional<bool> PackLoader::getBool(const std::string& key) const {
    const msgpack::object* val_obj = find_value_by_key(m_handle, key);
    if (val_obj && val_obj->type == msgpack::type::BOOLEAN) {
        return val_obj->as<bool>();
    }
    return std::nullopt;
}

} // namespace trackie::loader