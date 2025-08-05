/*
* Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#pragma once

#include <filesystem>
#include <nlohmann/json.hpp>

namespace trackie::loader {

    /**
     * @brief Carrega um arquivo MessagePack do disco e o converte para um objeto JSON.
     *
     * Esta função lê todo o conteúdo de um arquivo binário .msgpack,
     * faz o parsing dos dados e os retorna como um objeto nlohmann::json.
     * Isso fornece uma interface flexível (semelhante a um dicionário Python)
     * para acessar os dados de configuração em toda a aplicação.
     *
     * @param file_path O caminho para o arquivo .msgpack a ser carregado.
     *        Usa std::filesystem::path para um manuseio de caminhos robusto e moderno.
     * @return Um objeto nlohmann::json representando os dados do arquivo.
     * @throws std::runtime_error se o arquivo não puder ser encontrado, aberto ou lido.
     * @throws nlohmann::json::parse_error se os dados no arquivo não forem
     *         um MessagePack válido que possa ser convertido para JSON.
     */
    nlohmann::json loadMsgpackFile(const std::filesystem::path& file_path);

} // namespace trackie::loader