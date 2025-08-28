# Define a biblioteca 'c_core'
# Esta será a base em C puro do nosso projeto, contendo utilitários
# e estruturas de dados fundamentais.

add_library(c_core STATIC
    src/vector.c
    src/hashmap.c
    src/linked_list.c
)

# Expõe o diretório 'include' para que outros módulos possam usar os cabeçalhos do c_core.
target_include_directories(c_core PUBLIC
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)

# O c_core deve ser o mais independente possível.
# Se forem necessárias dependências (como pthreads), elas serão adicionadas aqui.
# Exemplo:
# find_package(Threads REQUIRED)
# target_link_libraries(c_core PUBLIC Threads::Threads)

# --- Configuração de Testes ---
if(BUILD_TESTING)
    message(STATUS "Configurando testes para c_core...")
    add_executable(test_vector tests/test_vector.c)

    # O executável de teste precisa se vincular à biblioteca c_core para usar suas funções.
    target_link_libraries(test_vector PRIVATE c_core)

    # Adiciona o teste ao CTest.
    add_test(NAME c_core_vector_test COMMAND test_vector)
endif()
