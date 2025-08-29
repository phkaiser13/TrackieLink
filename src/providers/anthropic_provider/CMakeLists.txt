# Placeholder CMakeLists.txt for Anthropic Provider

add_library(anthropic_provider STATIC
    src/anthropic_provider.cpp
)

target_include_directories(anthropic_provider PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against the abstract AI core interface
target_link_libraries(anthropic_provider PUBLIC
    ai_core
)
