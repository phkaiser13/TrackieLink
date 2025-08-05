/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

//This file is a Placeholder-like FIle

#ifndef RUST_TO_CPP_BRIDGE_H
#define RUST_TO_CPP_BRIDGE_H

// Este arquivo define a interface para funções implementadas em C++
// que devem ser chamáveis a partir do Rust.

// Exemplo: Se o Rust precisasse logar uma mensagem usando um sistema de log C++.

#ifdef __cplusplus
extern "C" {
#endif

    /**
     * @brief Função de exemplo que seria implementada em C++ e chamada pelo Rust.
     *
     * O código Rust declararia esta função em um bloco `extern "C" {}` e a chamaria
     * para passar uma mensagem de log para o lado C++.
     *
     * @param message A mensagem a ser logada, como uma string C.
     */
    void log_message_from_rust(const char* message);


#ifdef __cplusplus
} // extern "C"
#endif

#endif // RUST_TO_CPP_BRIDGE_H