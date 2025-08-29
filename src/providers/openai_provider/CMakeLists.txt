# Placeholder CMakeLists.txt for OpenAI Provider

add_library(openai_provider STATIC
    src/openai_provider.cpp
)

target_include_directories(openai_provider PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link against the abstract AI core interface
target_link_libraries(openai_provider PUBLIC
    ai_core
)
