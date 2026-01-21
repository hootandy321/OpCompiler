#include <cctype>
#include <chrono>
#include <cnrt.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <math.h>
#include <stdexcept>
#include <stdio.h>
#include <string>

#define G 1073741824
namespace ch {
using namespace std::chrono;
}

size_t parse_size_arg(const std::string &arg) {
  size_t multiplier = 1;
  std::string num_part;
  char suffix = '\0';

  for (char c : arg) {
    if (std::isdigit(c)) {
      num_part += c;
    } else {
      suffix = std::tolower(c);
      break;
    }
  }

  if (num_part.empty())
    throw std::invalid_argument("No numeric part in input");

  size_t base = std::stoull(num_part);

  switch (suffix) {
  case 'b':
  case '\0':
    multiplier = 1;
    break;
  case 'k':
    multiplier = 1024;
    break;
  case 'm':
    multiplier = 1024 * 1024;
    break;
  case 'g':
    multiplier = 1024 * 1024 * 1024;
    break;
  default:
    throw std::invalid_argument("Unknown size suffix");
  }

  return base * multiplier;
}

void test_memcpy(int len) {
  int warmupRounds = 100;
  int timingRounds = 200;

  cnrtQueue_t queue;
  cnrtSetDevice(0);

  float *host = (float *)malloc(len * sizeof(float));
  float *dev1;
  float *dev2;
  cnrtMalloc((void **)&dev1, len * sizeof(float));
  cnrtMalloc((void **)&dev2, len * sizeof(float));

  for (int i = 0; i < len; i++) {
    host[i] = i;
  }

  ch::time_point<ch::high_resolution_clock, ch::nanoseconds> time1, time2;

  // H2D: host -> dev1
  for (int i = 0; i < warmupRounds; i++) {
    cnrtMemcpy(dev1, host, len * sizeof(float), cnrtMemcpyHostToDev);
  }
  time1 = ch::high_resolution_clock::now();
  for (int i = 0; i < timingRounds; i++) {
    cnrtMemcpy(dev1, host, len * sizeof(float), cnrtMemcpyHostToDev);
  }
  time2 = ch::high_resolution_clock::now();

  float h2d_timeTotal =
      ch::duration_cast<ch::duration<float>>(time2 - time1).count();
  float h2d_bandwidth =
      len * sizeof(float) / (h2d_timeTotal / timingRounds) / G;

  // D2H: dev1 -> host
  for (int i = 0; i < warmupRounds; i++) {
    cnrtMemcpy(host, dev1, len * sizeof(float), cnrtMemcpyDevToHost);
  }
  time1 = ch::high_resolution_clock::now();
  for (int i = 0; i < timingRounds; i++) {
    cnrtMemcpy(host, dev1, len * sizeof(float), cnrtMemcpyDevToHost);
  }
  time2 = ch::high_resolution_clock::now();

  float d2h_timeTotal =
      ch::duration_cast<ch::duration<float>>(time2 - time1).count();
  float d2h_bandwidth =
      len * sizeof(float) / (d2h_timeTotal / timingRounds) / G;

  // D2D: dev1 -> dev2
  for (int i = 0; i < warmupRounds; i++) {
    cnrtMemcpy(dev2, dev1, len * sizeof(float), cnrtMemcpyDevToDev);
  }
  time1 = ch::high_resolution_clock::now();
  for (int i = 0; i < timingRounds; i++) {
    cnrtMemcpy(dev2, dev1, len * sizeof(float), cnrtMemcpyDevToDev);
  }
  time2 = ch::high_resolution_clock::now();

  float d2d_timeTotal =
      ch::duration_cast<ch::duration<float>>(time2 - time1).count();
  float d2d_bandwidth =
      len * sizeof(float) / (d2d_timeTotal / timingRounds) / G;

  std::cout << len * sizeof(float) << "," << h2d_bandwidth << ","
            << d2h_bandwidth << "," << d2d_bandwidth << std::endl;

  cnrtFree(dev1);
  cnrtFree(dev2);
  free(host);
}

int main(int argc, char *argv[]) {
  size_t max_bytes = 0;
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <max_size>, e.g., 10k, 16M, 1g"
              << std::endl;
    return 1;
  }

  try {
    max_bytes = parse_size_arg(argv[1]);
  } catch (const std::exception &e) {
    std::cerr << "Error parsing input: " << e.what() << std::endl;
    return 1;
  }

  std::cout << "Bytes Transferred,H2D Bandwidth (GB/s),D2H Bandwidth "
               "(GB/s),D2D Bandwidth (GB/s)"
            << std::endl;

  int len = 128; 
  while (len * sizeof(float) <= max_bytes) {
    test_memcpy(len);
    len *= 2;
  }
  test_memcpy(max_bytes / sizeof(float));

  return 0;
}
