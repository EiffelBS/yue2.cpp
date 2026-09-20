#pragma once
// model-store.h: centralised ownership of GGML modules
//
// VRAM policy doctrine. READ THIS BEFORE CHANGING ANYTHING IN THIS FILE.
//
//   --keep-loaded (EVICT_NEVER)
//       Everything stays in VRAM. No reload, ever. The user is telling us
//       they have the budget for the full working set. Do not second-guess
//       them by adding smart eviction rules.
//
//   default (EVICT_STRICT)
//       Maximum VRAM optimisation. At most one coexistence group resident
//       at a time. The AR weights and the NAR weights never coexist by
//       construction, because they live in different groups.
//
//   coexistence groups
//       The pipeline interleaves some modules at a granularity where
//       evicting between them would thrash, so eviction operates on
//       groups, not single modules:
//         AR         = { LM }        the score and semantic stages
//         SYNTH      = { NAR, VAE }  the flow matching and the decode, per song
//         TRANSCRIBE = { SS2 }       the audio to score transcriber
//       Modules in the same group coexist freely. A require from another
//       group evicts every idle module of the resident group.
//
//   the KV cache is not a module
//       The AR fills it and the NAR reads it, so it belongs to the pipeline
//       and survives every eviction. That is what lets the two halves of one
//       GGUF trade places in VRAM around it.
//
//   invariant held under BOTH policies
//       Exactly ONE instance per Yue2ModelKey for the whole process. Two
//       requires with the same key return the same pointer.
//
// A Yue2ModelStore holds the GGML module instances that the pipeline needs
// (AR half, NAR half, VAE). The pipeline asks the store for a module by key
// and returns it when done. The store decides what stays in VRAM and what
// gets evicted, following the policy above set at creation time.
//
// Keys
//   A module is uniquely identified by (kind, path). Two requires with the
//   same key return the same instance.
//
// Refcounting
//   Each module has a refcount. require increments it, release decrements.
//   In EVICT_STRICT, a module with refcount > 0 in another group cannot be
//   evicted: a conflicting require is a programming error (aborts). This
//   catches accidental overlap between groups that must not coexist.
//
// Threading
//   The store has no lock: a single worker thread owns it. The server
//   serializes GPU work on one worker; HTTP handlers never touch the
//   store. Adding a second worker requires adding a mutex here first.

#include "bpe.h"
#include "nar.h"
#include "qwen3-lm.h"
#include "sheetsage.h"
#include "vae.h"

#include <cstddef>
#include <string>

struct Yue2ModelStore;

enum ModelKind {
    MODEL_LM,   // Yue2Qwen3LM  the AR half of the backbone GGUF
    MODEL_NAR,  // Yue2NAR  the NAR half of the same GGUF
    MODEL_VAE,  // VAEGGML  from the VAE GGUF
    MODEL_SS2,  // SheetSage2 from the transcriber GGUF
};

struct Yue2ModelKey {
    ModelKind   kind;
    std::string path;  // GGUF path the module is loaded from
};

enum EvictPolicy {
    EVICT_STRICT,  // default: at most one coexistence group resident
    EVICT_NEVER,   // --keep-loaded: never evict, accumulate
};

Yue2ModelStore * yue2_store_create(EvictPolicy policy);
void         yue2_store_free(Yue2ModelStore * s);
EvictPolicy  yue2_store_policy(const Yue2ModelStore * s);

// Typed GPU module accessors. Each returns a pointer owned by the store;
// never free it yourself. Returns NULL on load failure.
//
// After require, the module stays resident with a refcount > 0 until the
// matching release. In EVICT_STRICT, require evicts every module outside
// its coexistence group whose refcount is zero; if any conflicting module
// has refcount > 0 the store aborts (a programming error in the caller).
Yue2Qwen3LM *    yue2_store_require_lm(Yue2ModelStore * s, const Yue2ModelKey & k);
Yue2NAR *    yue2_store_require_nar(Yue2ModelStore * s, const Yue2ModelKey & k);
VAEGGML *    yue2_store_require_vae(Yue2ModelStore * s, const Yue2ModelKey & k);
SheetSage2 * yue2_store_require_ss2(Yue2ModelStore * s, const Yue2ModelKey & k);

// Release decrements the refcount for the module behind this handle.
// Pass exactly the pointer returned by require. After release, the pointer
// must not be used: in EVICT_STRICT it may be unloaded immediately.
void yue2_store_release(Yue2ModelStore * s, void * handle);

// CPU-resident accessor. Loaded on first call, kept forever, never
// evicted: the tokenizer travels with the backbone GGUF metadata (a few
// MB). Returns NULL on load failure.
BPETokenizer * yue2_store_bpe(Yue2ModelStore * s, const char * lm_path);

// Observability: sum of currently resident GPU module weight buffers, and
// the count of loaded GPU modules.
size_t yue2_store_vram_bytes(const Yue2ModelStore * s);
int    yue2_store_gpu_module_count(const Yue2ModelStore * s);

// RAII helper. Builds on top of yue2_store_release, nothing else.
struct Yue2ModelHandle {
    Yue2ModelStore * store;
    void *       ptr;

    Yue2ModelHandle(Yue2ModelStore * s, void * p) : store(s), ptr(p) {}

    ~Yue2ModelHandle() {
        if (store && ptr) {
            yue2_store_release(store, ptr);
        }
    }

    // non-copyable, movable
    Yue2ModelHandle(const Yue2ModelHandle &)             = delete;
    Yue2ModelHandle & operator=(const Yue2ModelHandle &) = delete;

    Yue2ModelHandle(Yue2ModelHandle && o) noexcept : store(o.store), ptr(o.ptr) {
        o.store = nullptr;
        o.ptr   = nullptr;
    }
};
