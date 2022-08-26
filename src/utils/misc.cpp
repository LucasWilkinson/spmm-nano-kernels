//
// Created by lwilkinson on 8/25/22.
//

#include "utils/misc.h"

#include <iostream>
#include <filesystem>

std::string resolve_path(std::string file, const std::vector<std::string>& search_dirs) {
  // Filepath construction priority
  //  1. if path is an absolute path use as is
  //  2. search of non-empty search paths
  std::string fullpath = file;
  std::vector<std::string> tested_paths;
  bool file_found = false;

  // Assume leading / means absolute path
  if (file[0] == '/') {
    file_found = std::filesystem::exists(fullpath);
  } else {
    for (const auto& search_dir : search_dirs) {
      if (search_dir.empty()) continue;
      fullpath = search_dir + "/" + file;
      tested_paths.push_back(fullpath);
      if (std::filesystem::exists(fullpath)) {
        file_found = true;
        break;
      }
    }
  }

  if (!file_found) {
    std::cerr << "Failed to resolve location of file " << file << " tested:" << std::endl;
    for (const auto& path : tested_paths)
      std::cerr << "  " << path << std::endl;
    exit(-1);
  }

  // Clean path string
  return std::filesystem::path(fullpath).lexically_normal();
}
