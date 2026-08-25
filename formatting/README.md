# MATAR code formatting

Formatting is a two-stage pipeline:

1. **clang-format** (config: `.clang-format` at the repo root) — Google style with
   MATAR conventions: 4-space indents, 100-column limit, aligned consecutive
   assignments, and macro-awareness for `MATAR_*` statement macros and
   `KOKKOS_*` attribute macros.
2. **`matar-format.py`** — post-processor that rewrites MATAR parallel-macro
   calls (`FOR_ALL`, `DO_ALL`, `FOR_REDUCE_*`, `DO_REDUCE_*`, their `_CLASS`
   variants, and the hierarchical `FOR_FIRST`/`FOR_SECOND`/`FOR_THIRD`,
   `DO_FIRST`/..., `*_REDUCE_*_SECOND`/`_THIRD` forms) into the canonical
   layout clang-format cannot produce (it has no notion of argument groups):

   ```c++
   DO_REDUCE_MAX(k, 0, 10,
                 j, 0, 10,
                 i, 0, 10,
                 loc_max, {
       if (loc_max < arr3D(i, j, k)) {
           loc_max = arr3D(i, j, k);
       }
   }, result);
   ```

   One index triple per line aligned under the first argument, reduction
   variables on their own line, body at macro indent + 4, trailing arguments
   joined onto the closing line. Bodies keep clang-format's internal
   formatting; only their indentation is shifted. Macros nested inside another
   macro's body (e.g. a `FOR_REDUCE_SUM_SECOND` inside a `FOR_FIRST`) are
   reflowed recursively into the same expanded layout.
   The pass skips any call it cannot parse confidently, and comments/strings
   are masked so braces inside them cannot confuse the matcher.

## Usage

Always run both stages as a pair, clang-format first and the reflow second —
clang-format alone will re-pack the macro headers, and the reflow restores
them. Applied as a pair the result is stable (repeated runs are byte-identical).

```bash
clang-format -i src/include/*.h solvers/*.hpp
python3 formatting/matar-format.py src/include/*.h solvers/*.hpp
```

Files clang-format must never touch are listed in `.clang-format-ignore`
(currently `src/include/macros.h`, whose `#define \` continuation tables
clang-format mangles).
