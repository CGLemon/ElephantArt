#pragma once

#include <vector>
#include <string>

std::string connect_path(const std::string path_1, const std::string path_2);

// Creates directory at a given path. Throws exception if cannot.
// Returns silently if already exists.
void create_directory(const std::string& path);

// Returns true if the directory already exists.
bool is_directory_exist(const std::string& directory);

// Returns list of full paths of regular files in this directory.
// Silently returns empty vector on error.
std::vector<std::string> get_file_list(const std::string& directory);

// Returns list of full paths of directory in this directory.
// Silently returns empty vector on error.
std::vector<std::string> get_directory_list(const std::string& directory);

// Returns list of full paths of regular files in this directory tree.
// Silently returns empty vector on error.
std::vector<std::string> search_file_tree(const std::string& directory, size_t *counter = nullptr);

// Returns size of a file, 0 if file doesn't exist or can't be read.
uint64_t get_file_size(const std::string& filename);

// Returns modification time of a file, 0 if file doesn't exist or can't be read.
time_t get_file_time(const std::string& filename);
