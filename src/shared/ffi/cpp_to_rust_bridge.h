/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#ifndef CPP_TO_RUST_BRIDGE_H
#define CPP_TO_RUST_BRIDGE_H

#include <stdbool.h> // Para usar 'bool' em vez de '_Bool' em C/C++

// --- Guarda para compatibilidade C/C++ ---
// Garante que o C++ trate estas declarações como funções C puras.
#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Inicializa o cache de ativos global no lado Rust.
     *
     * Deve ser chamada uma vez durante a inicialização da aplicação para
     * preparar o estado interno do módulo Rust.
     */
    void velocity_init_cache();

    /**
     * @brief Libera a memória do cache de ativos global no lado Rust.
     *
     * Deve ser chamada uma vez durante a limpeza da aplicação para garantir
     * que não haja vazamentos de memória.
     */
    void velocity_destroy_cache();

    /**
     * @brief Verifica se um ativo está no cache Rust e o adiciona se não estiver.
     *
     * @param asset_path_ptr Um ponteiro para uma string C terminada em nulo
     *                       representando o caminho do ativo.
     * @return 'true' se o ativo já estava no cache (cache hit),
     *         'false' caso contrário (cache miss).
     */
    bool velocity_check_and_cache_asset(const char* asset_path_ptr);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // CPP_TO_RUST_BRIDGE_H