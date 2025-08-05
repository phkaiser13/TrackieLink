set(RUST_CORE_SRC_DIR ${CMAKE_SOURCE_DIR}/src/core/rust_core)

if(WIN32)
    set(RUST_CORE_LIB ${RUST_CORE_SRC_DIR}/target/release/rust_core.lib)
else()
    set(RUST_CORE_LIB ${RUST_CORE_SRC_DIR}/target/release/librust_core.a)
endif()

add_custom_command(
        OUTPUT ${RUST_CORE_LIB}
        COMMAND cargo build --release --manifest-path ${RUST_CORE_SRC_DIR}/Cargo.toml
        WORKING_DIRECTORY ${RUST_CORE_SRC_DIR}
        COMMENT "Building Rust core library..."
        VERBATIM
)

add_custom_target(
        rust_core_builder ALL
        DEPENDS ${RUST_CORE_LIB}
)

add_library(rust_core STATIC IMPORTED GLOBAL)
set_property(TARGET rust_core PROPERTY IMPORTED_LOCATION ${RUST_CORE_LIB})
add_dependencies(rust_core rust_core_builder)

add_executable(TrackieLink
        Core/application.cpp
        Core/main.cpp
)

target_include_directories(TrackieLink PRIVATE
        ${CMAKE_SOURCE_DIR}/src/shared/ffi
        ${CMAKE_SOURCE_DIR}/src/shared/types
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${NLOHMANN_JSON_INCLUDE_DIR}
)

target_link_libraries(TrackieLink PRIVATE
        gemini_service_lib
        video_processing
        rust_core
        # audio_processing é transitiva, então não linka aqui
)

add_dependencies(TrackieLink rust_core)
