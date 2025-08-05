/*
 * Author: Pedro h. Garcia <phkaiser13>.
 * Licensed under the vyAI Social Commons License 1.0
 * See the LICENSE file in the project root.
 *
 * You are free to use, modify, and share this file under the terms of the license,
 * provided proper attribution and open distribution are maintained.
 */

use std::collections::HashSet;
use std::ffi::{CStr};
use std::sync::Mutex;
use libc::c_char;

// --- Estado Global Gerenciado pelo Rust ---
//
// Usamos um Mutex para garantir acesso thread-safe a um estado global.
// O `lazy_static` (ou `once_cell` em edições mais novas) seria ideal para
// inicialização global, mas para manter as dependências mínimas, vamos
// usar uma abordagem com `Box::leak` para criar um estado estático mutável.
// Este `ASSET_CACHE` simulará um cache de ativos já carregados.
static CACHE_MUTEX: Mutex<Option<Box<HashSet<String>>>> = Mutex::new(None);

/// Inicializa o cache de ativos global.
/// Deve ser chamada uma vez pela aplicação C++ durante a inicialização.
#[no_mangle]
pub extern "C" fn velocity_init_cache() {
    let mut cache_guard = CACHE_MUTEX.lock().unwrap();
    if cache_guard.is_none() {
        println!("[Rust] Inicializando o cache de ativos Velocity.");
        *cache_guard = Some(Box::new(HashSet::new()));
    }
}

/// Libera a memória do cache de ativos global.
/// Deve ser chamada uma vez pela aplicação C++ durante a limpeza.
#[no_mangle]
pub extern "C" fn velocity_destroy_cache() {
    let mut cache_guard = CACHE_MUTEX.lock().unwrap();
    if let Some(_cache_box) = cache_guard.take() {
        // O `cache_box` é consumido e sua memória é liberada quando sai de escopo.
        println!("[Rust] Cache de ativos Velocity destruído.");
    }
}

/// Verifica se um ativo está no cache e o adiciona se não estiver.
///
/// Recebe uma string C do chamador, a converte para uma string Rust,
/// interage com o HashSet e retorna um booleano indicando se foi um "cache hit".
///
/// # Safety
/// O ponteiro `asset_path_ptr` deve ser um ponteiro válido para uma string C
/// terminada em nulo.
#[no_mangle]
pub extern "C" fn velocity_check_and_cache_asset(asset_path_ptr: *const c_char) -> bool {
    if asset_path_ptr.is_null() {
        return false;
    }

    // Converte a string C para um CStr Rust de forma segura.
    let c_str = unsafe { CStr::from_ptr(asset_path_ptr) };

    // Converte o CStr para uma String Rust.
    let asset_path = match c_str.to_str() {
        Ok(s) => s.to_owned(),
        Err(_) => return false, // Retorna false se a string não for UTF-8 válida.
    };

    let mut cache_guard = CACHE_MUTEX.lock().unwrap();

    if let Some(ref mut cache) = *cache_guard {
        // Se o ativo já existe no cache, retorna true (cache hit).
        if cache.contains(&asset_path) {
            true
        } else {
            // Se não, o insere e retorna false (cache miss).
            println!("[Rust] Cache miss para '{}'. Adicionando ao cache.", asset_path);
            cache.insert(asset_path);
            false
        }
    } else {
        // O cache não foi inicializado.
        eprintln!("[Rust] ERRO: velocity_check_and_cache_asset chamado antes de velocity_init_cache.");
        false
    }
}