// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Generic accelerator stream-completion shim.
//
// Compiled once per vendor with DEVICE_RUNTIME_HEADER, STREAM_TYPE and
// LAUNCH_HOST_FUNC supplied by sglang_fl.runtime.stream_completion.jit.  The
// callback never touches Python, so it is safe to run on the driver's
// completion thread.

#include <cerrno>
#include <cstdint>
#include <sys/eventfd.h>

#include DEVICE_RUNTIME_HEADER

namespace {

void notify_eventfd(void* user_data) {
  const int event_fd = static_cast<int>(reinterpret_cast<intptr_t>(user_data));
  while (eventfd_write(event_fd, 1) != 0 && errno == EINTR) {
  }
}

}  // namespace

extern "C" int enqueue_eventfd_completion(uintptr_t stream_ptr, int event_fd) {
  return static_cast<int>(LAUNCH_HOST_FUNC(
      reinterpret_cast<STREAM_TYPE>(stream_ptr), notify_eventfd,
      reinterpret_cast<void*>(static_cast<intptr_t>(event_fd))));
}
