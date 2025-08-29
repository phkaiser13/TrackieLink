# Placeholder CMakeLists.txt for Meta Provider

add_library(meta_provider STATIC
    src/meta_provider.cpp
)

target_include_directories(meta_provider PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against the abstract AI core interface
target_link_libraries(meta_provider PUBLIC
    ai_core
)
