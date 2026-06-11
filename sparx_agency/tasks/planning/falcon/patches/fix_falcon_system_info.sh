#!/bin/bash
# ============================================================
# fix_falcon_system_info.sh  (robust v2)
#
# Replaces FALCON's printSystemInfo() so it NEVER aborts the
# exploration_node when nvidia-smi is missing or returns a
# non-numeric string.
#
# Root cause it fixes:
#   exploration_utils/src/system_info.cpp calls std::stol() on the
#   output of `nvidia-smi --query-gpu=...`. On Jetson that query
#   returns an NVML error string; in a CPU-only / no-utility-cap
#   container it returns nothing. std::stol() then throws an
#   uncaught std::invalid_argument -> SIGABRT (roslaunch exit -6)
#   at startup, before ROS logging exists (so no .log is written).
#
# This version REPLACES the whole file rather than sed-patching a
# line, so it can't silently no-op when upstream formatting changes.
# It self-verifies at the end.
# ============================================================
set -euo pipefail

# Locate the file regardless of workspace layout.
SI="$(find /catkin_ws/src/FALCON -path '*exploration_utils*/system_info.cpp' | head -n1)"
if [ -z "${SI}" ]; then
  echo "[fix_system_info] ERROR: system_info.cpp not found under /catkin_ws/src/FALCON" >&2
  exit 1
fi
echo "[fix_system_info] Patching ${SI}"
cp "${SI}" "${SI}.orig.bak" 2>/dev/null || true

cat > "${SI}" <<'CPP_EOF'
#include "system_info.h"

// Crash-safe stol: never throws. Returns fallback on bad input.
static long safe_stol(const std::string &s, long fallback = 0) {
  try {
    return std::stol(s);
  } catch (...) {
    return fallback;
  }
}

void printSystemInfo(std::string &output) {
  std::stringstream ss;
  ss << "|---------------------------------- System Info ----------------------------------|"
            << std::endl;
  std::string line;
  std::string cpu_name, cpu_cores, cpu_threads, cpu_freq;
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo.is_open()) {
    while (getline(cpuinfo, line)) {
      if (line.find("model name") != std::string::npos) {
        cpu_name = line.substr(line.find(":") + 2);
      } else if (line.find("cpu cores") != std::string::npos) {
        cpu_cores = line.substr(line.find(":") + 2);
      } else if (line.find("siblings") != std::string::npos) {
        cpu_threads = line.substr(line.find(":") + 2);
      } else if (line.find("cpu MHz") != std::string::npos) {
        cpu_freq = line.substr(line.find(":") + 2);
      }
    }
    cpuinfo.close();
  } else {
    std::cerr << "Unable to open /proc/cpuinfo" << std::endl;
  }

  ss << "CPU Name: " << cpu_name << std::endl;
  ss << "CPU Cores: " << cpu_cores << std::endl;
  ss << "CPU Threads: " << cpu_threads << std::endl;
  ss << "CPU Frequency: " << cpu_freq << " MHz" << std::endl;

  std::ifstream meminfo("/proc/meminfo");
  std::string mem_total, mem_free;
  if (meminfo.is_open()) {
    while (getline(meminfo, line)) {
      if (line.find("MemTotal") != std::string::npos) {
        mem_total = line.substr(line.find(":") + 2);
      } else if (line.find("MemFree") != std::string::npos) {
        mem_free = line.substr(line.find(":") + 2);
      }
    }
    meminfo.close();
  } else {
    std::cerr << "Unable to open /proc/meminfo" << std::endl;
  }

  ss << "Memory Total: " << safe_stol(mem_total) / 1024.0 / 1024.0 << " GB" << std::endl;
  ss << "Memory Free: " << safe_stol(mem_free) / 1024.0 / 1024.0 << " GB" << std::endl;

  // gpu info -- robust: never abort if nvidia-smi is absent or returns junk.
  std::string gpu_name = "N/A", gpu_mem_total = "0", gpu_mem_free = "0";
  std::string command =
      "nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null";
  FILE *fp = popen(command.c_str(), "r");
  if (fp != NULL) {
    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), fp) != NULL) {
      std::string l(buffer);
      size_t pos = l.find(",");
      if (pos == std::string::npos) continue;   // malformed / error line
      gpu_name = l.substr(0, pos);
      l = l.substr(pos + 1);
      pos = l.find(",");
      if (pos == std::string::npos) continue;
      gpu_mem_total = l.substr(0, pos);
      gpu_mem_free = l.substr(pos + 1);
    }
    pclose(fp);
  } else {
    std::cerr << "Failed to run command nvidia-smi" << std::endl;
  }

  ss << "GPU Name: " << gpu_name;
  ss << "GPU Memory Total: " << safe_stol(gpu_mem_total) / 1024.0 << " GB" << std::endl;
  ss << "GPU Memory Free: " << safe_stol(gpu_mem_free) / 1024.0 << " GB" << std::endl;

  ss << "|---------------------------------------------------------------------------------|"
            << std::endl;

  output = ss.str();
}
CPP_EOF

# Verify the dangerous pattern is gone and the guard is present.
if grep -qE 'std::stol\(gpu_mem' "${SI}"; then
  echo "[fix_system_info] ERROR: bare std::stol on gpu_mem still present!" >&2
  exit 1
fi
if ! grep -q 'safe_stol' "${SI}"; then
  echo "[fix_system_info] ERROR: safe_stol guard missing after patch!" >&2
  exit 1
fi
echo "[fix_system_info] OK: printSystemInfo() is now crash-safe."