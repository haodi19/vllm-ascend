// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
  #include <malloc.h>
#else
  #include <dlfcn.h>
#endif

namespace host_sync {

// No ACL initialization, device reset, or context ownership is hidden here.
// The owner must stop all workers and device users before calling close().
class HostMemory {
 public:
  static constexpr size_t kPageAlignment = 4096;

  HostMemory(size_t size_bytes, const std::string& backend, int device_id, const std::string& acl_library)
      : backend_(backend), device_id_(device_id) {
    if (size_bytes == 0 || size_bytes > std::numeric_limits<size_t>::max() - 2 * kPageAlignment) {
      throw std::invalid_argument("host control memory size is zero or too large");
    }
    size_ = (size_bytes + kPageAlignment - 1) & ~(kPageAlignment - 1);
    if (backend_ != "cpu" && backend_ != "acl") {
      throw std::invalid_argument("memory backend must be 'cpu' or 'acl'");
    }
    if (backend_ == "acl" && device_id_ < 0) {
      throw std::invalid_argument("ACL requires a non-negative logical device_id");
    }
    try {
      if (backend_ == "cpu") {
        allocate_cpu();
      } else {
        allocate_acl(acl_library);
      }
    } catch (...) {
      cleanup_noexcept();
      throw;
    }
  }

  HostMemory(const HostMemory&) = delete;
  HostMemory& operator=(const HostMemory&) = delete;
  HostMemory(HostMemory&&) = delete;
  HostMemory& operator=(HostMemory&&) = delete;

  ~HostMemory() { cleanup_noexcept(); }

  void* data() const noexcept { return data_; }
  uintptr_t device_address() const noexcept { return reinterpret_cast<uintptr_t>(device_alias_); }
  size_t size() const noexcept { return size_; }

  // Called once on the worker before it runs callbacks. The caller retains
  // ownership of the context for the entire lifetime of this allocation.
  void enter_worker_context() {
    if (!data_) throw std::runtime_error("host control memory is closed");
    if (backend_ == "cpu") return;
    check(set_context_(context_), "aclrtSetCurrentContext(worker)");
    int32_t actual_device = -1;
    check(get_device_(&actual_device), "aclrtGetDevice(worker)");
    if (actual_device != device_id_) {
      throw std::runtime_error("worker context belongs to a different logical device");
    }
  }

  // Idempotent after success. On ACL cleanup failure, ownership is retained so
  // explicit close() can be retried. In particular, registered memory is never
  // freed when aclrtHostUnregister fails.
  void close() {
    if (backend_ == "cpu") {
      if (allocation_) {
#if defined(_WIN32)
        _aligned_free(allocation_);
#else
        std::free(allocation_);
#endif
        allocation_ = nullptr;
        data_ = nullptr;
      }
      return;
    }
    if (!allocation_) {
      close_library();
      return;
    }

    // Never release memory under an unrelated context. Temporarily switch only
    // if there is a valid previous context that we can subsequently restore.
    void* previous_context = nullptr;
    check(get_context_(&previous_context), "aclrtGetCurrentContext(close)");
    if (!previous_context) {
      throw std::runtime_error("close requires a live ACL context; close before torch_npu teardown");
    }
    const bool switched = previous_context != context_;
    if (switched) check(set_context_(context_), "aclrtSetCurrentContext(close)");

    std::string failure;
    if (registered_) {
      const int result = unregister_(data_);
      if (result != 0) {
        failure = error_message("aclrtHostUnregister", result);
      } else {
        registered_ = false;
        device_alias_ = nullptr;
      }
    }
    if (failure.empty()) {
      const int result = free_host_(allocation_);
      if (result != 0) {
        failure = error_message("aclrtFreeHost", result);
      } else {
        allocation_ = nullptr;
        data_ = nullptr;
      }
    }
    if (switched) {
      const int result = set_context_(previous_context);
      if (result != 0) {
        if (!failure.empty()) failure += "; ";
        failure += error_message("aclrtSetCurrentContext(restore)", result);
      }
    }
    if (!failure.empty()) throw std::runtime_error(failure);
    close_library();
  }

 private:
  // Signatures mirror the ACL C ABI; no CANN headers or link-time dependency
  // are needed by the CPU-only extension.
  using GetContext = int (*)(void**);
  using SetContext = int (*)(void*);
  using GetDevice = int (*)(int32_t*);
  using MallocHost = int (*)(void**, size_t);
  using FreeHost = int (*)(void*);
  using RegisterHost = int (*)(void*, uint64_t, int, void**);
  using UnregisterHost = int (*)(void*);
  // aclrtHostRegisterType::ACL_HOST_REGISTER_MAPPED, not the V2 bit flag.
  static constexpr int kRegisterMapped = 0;

  static std::string error_message(const char* operation, int result) {
    return std::string(operation) + " failed with ACL error " + std::to_string(result);
  }

  static void check(int result, const char* operation) {
    if (result != 0) throw std::runtime_error(error_message(operation, result));
  }

  template <typename Function>
  Function symbol(const char* name) {
#if defined(_WIN32)
    (void)name;
    throw std::runtime_error("ACL host memory is supported only on Linux");
#else
    dlerror();
    void* address = dlsym(library_, name);
    const char* error = dlerror();
    if (error || !address) {
      throw std::runtime_error(std::string("missing ACL symbol ") + name + ": " + (error ? error : "null symbol"));
    }
    return reinterpret_cast<Function>(address);
#endif
  }

  void allocate_cpu() {
#if defined(_WIN32)
    allocation_ = _aligned_malloc(size_, kPageAlignment);
    if (!allocation_) throw std::bad_alloc();
#else
    const int result = posix_memalign(&allocation_, kPageAlignment, size_);
    if (result != 0) {
      allocation_ = nullptr;
      throw std::runtime_error("posix_memalign failed with error " + std::to_string(result));
    }
#endif
    data_ = allocation_;
    std::memset(data_, 0, size_);
    // CPU simulation deliberately has no device alias.
  }

  void allocate_acl(const std::string& acl_library) {
#if defined(_WIN32)
    (void)acl_library;
    throw std::runtime_error("ACL host memory is supported only on Linux");
#else
    const std::string library_name = acl_library.empty() ? "libascendcl.so" : acl_library;
    library_ = dlopen(library_name.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!library_) {
      const char* error = dlerror();
      throw std::runtime_error("cannot load ACL library " + library_name + ": " +
                               (error ? error : "unknown loader error"));
    }
    get_context_ = symbol<GetContext>("aclrtGetCurrentContext");
    set_context_ = symbol<SetContext>("aclrtSetCurrentContext");
    get_device_ = symbol<GetDevice>("aclrtGetDevice");
    malloc_host_ = symbol<MallocHost>("aclrtMallocHost");
    free_host_ = symbol<FreeHost>("aclrtFreeHost");
    register_ = symbol<RegisterHost>("aclrtHostRegister");
    unregister_ = symbol<UnregisterHost>("aclrtHostUnregister");
    check(get_context_(&context_), "aclrtGetCurrentContext");
    if (!context_) {
      throw std::runtime_error("ACL has no current context; initialize torch_npu on the selected device first");
    }
    int32_t actual_device = -1;
    check(get_device_(&actual_device), "aclrtGetDevice");
    if (actual_device != device_id_) {
      throw std::runtime_error("requested logical device_id does not match the current ACL device");
    }

    // aclrtMallocHost does not promise 4 KiB alignment on every supported
    // version. Retain the original allocation for aclrtFreeHost and register
    // a page-aligned subregion wholly contained within it.
    check(malloc_host_(&allocation_, size_ + kPageAlignment - 1), "aclrtMallocHost");
    if (!allocation_) throw std::runtime_error("aclrtMallocHost returned a null pointer");
    const uintptr_t base = reinterpret_cast<uintptr_t>(allocation_);
    data_ = reinterpret_cast<void*>((base + kPageAlignment - 1) & ~(kPageAlignment - 1));
    std::memset(data_, 0, size_);
    check(register_(data_, static_cast<uint64_t>(size_), kRegisterMapped, &device_alias_), "aclrtHostRegister");
    registered_ = true;
    if (!device_alias_) {
      throw std::runtime_error("aclrtHostRegister returned a null device alias");
    }
#endif
  }

  void close_library() noexcept {
#if !defined(_WIN32)
    if (library_) {
      dlclose(library_);
      library_ = nullptr;
    }
#endif
  }

  void cleanup_noexcept() noexcept {
    try {
      close();
    } catch (const std::exception& error) {
      // Leaking on failed unregister is safer than freeing still-registered
      // memory. Explicit close() reports failures to Python before teardown.
      std::fprintf(stderr, "host_sync memory cleanup failed: %s; resources retained\n", error.what());
    } catch (...) {
      std::fprintf(stderr, "host_sync memory cleanup failed; resources retained\n");
    }
  }

  std::string backend_;
  int device_id_;
  size_t size_ = 0;
  void* allocation_ = nullptr;
  void* data_ = nullptr;
  void* device_alias_ = nullptr;
  bool registered_ = false;
  void* library_ = nullptr;
  void* context_ = nullptr;
  GetContext get_context_ = nullptr;
  SetContext set_context_ = nullptr;
  GetDevice get_device_ = nullptr;
  MallocHost malloc_host_ = nullptr;
  FreeHost free_host_ = nullptr;
  RegisterHost register_ = nullptr;
  UnregisterHost unregister_ = nullptr;
};

}  // namespace host_sync
