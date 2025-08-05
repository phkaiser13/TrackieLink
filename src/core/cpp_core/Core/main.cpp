/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "application.hpp"
#include "logger.hpp"
#include <iostream>
#include <string>
#include <vector>

/**
 * @brief O ponto de entrada principal para a aplicação TrackieLink.
 *
 * Esta função analisa os argumentos da linha de comando, instancia a classe
 * principal da aplicação e a executa, capturando quaisquer exceções de
 * nível superior.
 */
int main(int argc, char* argv[]) {
    // Análise simples de argumentos da linha de comando
    std::string mode = "camera";
    bool show_preview = false;

    std::vector<std::string> args(argv + 1, argv + argc);
    for (size_t i = 0; i < args.size(); ++i) {
        if (args[i] == "--mode" && i + 1 < args.size()) {
            mode = args[++i];
        } else if (args[i] == "--show_preview") {
            show_preview = true;
        } else if (args[i] == "--help" || args[i] == "-h") {
            std::cout << "Uso: TrackieLink [--mode <camera|screen>] [--show_preview]\n";
            return 0;
        }
    }

    try {
        // Cria a aplicação na pilha. O RAII garante que o destrutor será chamado.
        trackie::core::Application app(mode, show_preview);

        // Executa o loop principal da aplicação.
        return app.run();

    } catch (const std::exception& e) {
        // Usa nosso logger para o erro final.
        trackie::log::TLog(trackie::log::LogLevel::ERROR, "Aplicação terminada por uma exceção: ", e.what());
        return 1;
    } catch (...) {
        trackie::log::TLog(trackie::log::LogLevel::ERROR, "Aplicação terminada por uma exceção desconhecida.");
        return 1;
    }

    return 0;
}