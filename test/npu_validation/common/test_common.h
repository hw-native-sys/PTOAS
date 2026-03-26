#pragma once

#include <cstddef>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace PtoTestCommon {

inline bool ReadFile(const std::string &filePath, size_t &fileSize, void *buffer, size_t bufferSize) {
  struct stat sBuf;
  if (stat(filePath.c_str(), &sBuf) == -1) {
    return false;
  }
  if (!S_ISREG(sBuf.st_mode)) {
    return false;
  }

  std::ifstream file(filePath, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  std::filebuf *buf = file.rdbuf();
  size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
  if (size == 0 || size > bufferSize) {
    return false;
  }
  buf->pubseekpos(0, std::ios::in);
  buf->sgetn(static_cast<char *>(buffer), size);
  fileSize = size;
  return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size) {
  if (buffer == nullptr) {
    return false;
  }

  int fd = open(filePath.c_str(), O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWRITE);
  if (fd < 0) {
    return false;
  }

  ssize_t writeSize = write(fd, buffer, size);
  (void)close(fd);
  return writeSize == static_cast<ssize_t>(size);
}

}  // namespace PtoTestCommon
