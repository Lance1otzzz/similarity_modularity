#pragma once
#include <string>

// Ensure folder path ends with separator
std::string ensureFolderSeparator(const std::string &folder) {
    if (folder.empty()) return "./";
    char last = folder.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        return folder + "\\";
#else
        return folder + "/";
#endif
    }
    return folder;
}
