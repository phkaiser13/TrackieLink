# O cpp_core é o orquestrador principal, então ele se torna o executável final.
add_executable(TrackieLink
        Core/application.cpp
        Core/main.cpp
)

# Incluímos os cabeçalhos FFI (Foreign Function Interface) para comunicação com Rust,
# os tipos compartilhados, e os cabeçalhos do próprio core.
target_include_directories(TrackieLink PRIVATE
        ${CMAKE_SOURCE_DIR}/src/shared/ffi
        ${CMAKE_SOURCE_DIR}/src/shared/types
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# O executável final se vincula a todos os módulos e serviços da aplicação,
# bem como à biblioteca Rust e às threads.
target_link_libraries(TrackieLink PRIVATE
        gemini_service_lib
        video_processing
        vision_processing
        c_inference
        function_handler_lib
        msgpack_loader
        nlohmann_json
        rust_core_lib
        Threads::Threads
        spdlog::spdlog
)

if(BUILD_TESTING)
    message(STATUS "Configurando testes para o cpp_core...")

    add_executable(cpp_core_tests
        tests/sample_test.cpp
    )

    target_link_libraries(cpp_core_tests PRIVATE
        GTest::gtest
        GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(cpp_core_tests)
endif()
