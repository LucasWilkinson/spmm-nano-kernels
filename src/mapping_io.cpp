//
// Created by lwilkinson on 8/25/22.
//

#include "mapping_io.h"
#include "utils/misc.h"

#include "MicroKernelBase.h"

#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <sstream>
#include <iostream>

namespace sop {

static std::string replace(
  std::string str,
  const std::string& from,
  const std::string& to
) {
  size_t start_pos = 0;
  while((start_pos = str.find(from, start_pos)) != std::string::npos) {
    str.replace(start_pos, from.length(), to);
    start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
  }
  return str;
}

static std::vector<std::string> split(std::string strToSplit, char delimeter)
{
  std::stringstream ss(strToSplit);
  std::string item;
  std::vector<std::string> splittedStrings;
  while (std::getline(ss, item, delimeter))
  {
    splittedStrings.push_back(item);
  }
  return splittedStrings;
}

std::shared_ptr<NanoKernelMapping> read_pattern_mapping(
    const std::string& id,
    const std::vector<std::string>& search_dirs
) {
  std::string filepath = resolve_path("mapping_" + id + ".txt", search_dirs);

  std::ifstream file(filepath);
  std::string line;

  if (!file.is_open()) {
    std::cout << "Failed to open " << filepath << std::endl;
    exit(-1);
  }

  std::getline(file, line);
  int M_r = std::stoi(line);

  auto pattern_mapping_ptr = std::make_shared<NanoKernelMapping>(1 << M_r);
  auto& pattern_mapping = *pattern_mapping_ptr;

  pattern_mapping[0].push_back(0);

  while (std::getline(file, line)) {
    // Cleanup
    line = replace(line, "[", "");
    line = replace(line, "]", "");

    auto line_split = split(line, ':');
    int pattern = std::stoi(line_split[0]);
    auto nano_kernel_strings = split(line_split[1], ',');

    // replace all 'x' to 'y'
    for (const auto& nano_kernel_string : nano_kernel_strings) {
      pattern_mapping[pattern].push_back(std::stoi(nano_kernel_string));
    }
  }

  file.close();
  return pattern_mapping_ptr;
}
}