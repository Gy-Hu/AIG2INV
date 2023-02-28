

#include <cassert>
#include <unordered_map>
#include "version_manager.h"
#include <sstream>
#include <vector>
#include <array>
#include <cassert>
#include <string>

template <size_t MAX_SIZE>
size_t split(const std::string &s, char delim, std::array<std::string, MAX_SIZE> &elems)
{
    static_assert(MAX_SIZE > 0, "Size must be positive");
    
    std::stringstream ss(s);
    std::string item;
    size_t count = 0;

    while(std::getline(ss, item, delim))
    {
#ifdef DEBUG
        assert(count < MAX_SIZE);
#endif
        elems[count++] = item;
    }
    return count;
}

template <size_t N>
std::array<std::string, N> split_to(const std::string& s, char delim)
{
    std::array<std::string, N> arr_to_ret;
    size_t elements_inserted = 0;

    std::stringstream ss(s);
    std::string item;

    while(std::getline(ss, item, delim) && elements_inserted < N) {
        arr_to_ret[elements_inserted++] = item;
    }
    assert(elements_inserted == N);
    return arr_to_ret;
}

std::unordered_map<std::string, size_t> VersionManager::_copies_counter = {};


std::string VersionManager::new_version(const std::string &orig_name)
{
    if (orig_name.find(DELIM) == std::string::npos)
    {
        assert(_copies_counter.find(orig_name) == _copies_counter.end());
        _copies_counter[orig_name] = 0;
        return orig_name + std::string(1, DELIM) +"0";
    }
    else
    {
        std::array<std::string, 2> parts = split_to<2>(orig_name, DELIM);
        return parts[0] + std::string(1, DELIM) + std::to_string(++_copies_counter[parts[0]]);
    }
}

std::string VersionManager::new_version(const size_t orig_name) {
    return VersionManager::new_version(std::to_string(orig_name));
}

void VersionManager::reset() {
    _copies_counter.clear();
}

size_t VersionManager::next_version_number(const std::string &key) {
    assert(key.find(DELIM) == key.npos);
    if (_copies_counter.find(key) == _copies_counter.end()) {
        _copies_counter[key] = 0;
        return 0;
    } else {
        return ++_copies_counter[key];
    }
}

std::string VersionManager::next_version(const std::string &key) {
    assert(key.find(DELIM) == key.npos);
    if (_copies_counter.find(key) == _copies_counter.end())
    {
        _copies_counter[key] = 0;
        return (key +std::string(1, DELIM)+"0");
    }
    else
    {
        return (key + std::string(1, DELIM) + std::to_string(++_copies_counter[key]));
    }
}

std::string VersionManager::version_to_string(const size_t version) {
    return std::string("Abs") + std::string(1, DELIM) + std::to_string(version);
}
