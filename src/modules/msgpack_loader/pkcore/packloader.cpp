/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

#include "packloader.hpp"
#include <fstream>
#include <vector>
#include <stdexcept>

namespace trackie::loader {

nlohmann::json loadMsgpackFile(const std::filesystem::path& file_path) {
    // 1. Verificar se o arquivo existe antes de tentar abri-lo.
    //    Isso nos permite fornecer uma mensagem de erro mais clara.
    if (!std::filesystem::exists(file_path)) {
        throw std::runtime_error("Arquivo de configuração não encontrado em: " + file_path.string());
    }

    // 2. Abrir o arquivo em modo binário.
    //    'std::ios::ate' posiciona o cursor no final do arquivo,
    //    o que nos permite descobrir seu tamanho facilmente.
    std::ifstream file(file_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        throw std::runtime_error("Falha ao abrir o arquivo: " + file_path.string());
    }

    // 3. Obter o tamanho do arquivo e alocar um buffer.
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg); // Voltar para o início do arquivo.

    // Usamos um std::vector<uint8_t> como nosso buffer. É seguro,
    // gerencia sua própria memória e é compatível com as funções de parsing.
    std::vector<uint8_t> buffer(size);

    // 4. Ler todo o conteúdo do arquivo para o buffer.
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        throw std::runtime_error("Falha ao ler o conteúdo do arquivo: " + file_path.string());
    }

    // 5. Fazer o parsing do buffer de MessagePack para um objeto JSON.
    //    A biblioteca nlohmann/json tem uma função estática 'from_msgpack'
    //    que faz exatamente isso. Ela lançará uma exceção em caso de dados
    //    corrompidos ou malformados.
    return nlohmann::json::from_msgpack(buffer);
}

} // namespace trackie::loader