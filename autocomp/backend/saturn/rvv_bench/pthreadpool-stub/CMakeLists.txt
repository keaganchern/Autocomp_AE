cmake_minimum_required(VERSION 3.5)
project(pthreadpool C)

# Empty static lib. XNNPACK_BUILD_LIBRARY=OFF in the parent build means
# operator-run.c / runtime.c / subgraph.c are not compiled, so no
# pthreadpool_* symbol is ever referenced. This target exists only so
# that link commands that name `pthreadpool` resolve.
add_library(pthreadpool STATIC src/pthreadpool-stub.c)
target_include_directories(pthreadpool PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
