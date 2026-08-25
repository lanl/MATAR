# MATAR code formatting

Formatting is a two-stage pipeline:

1. **clang-format** (config: `.clang-format` at the repo root) — Google style with
   MATAR conventions: 4-space indents, 100-column limit, aligned consecutive
   assignments, and macro-awareness for `MATAR_*` statement macros and
   `KOKKOS_*` attribute macros.
2. **`matar-format.py`** — post-processor that rewrites MATAR parallel-macro
   calls (`FOR_ALL`, `DO_ALL`, `FOR_REDUCE_*`, `DO_REDUCE_*`, and their `_CLASS`
   variants) into the canonical layout clang-format cannot produce
   (it has no notion of argument groups):

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
   formatting; only their indentation is shifted. The pass is idempotent and
   skips any call it cannot parse confidently.

## Usage

Always run both stages, clang-format first:

```bash
clang-format -i src/include/*.h solvers/*.hpp
python3 formatting/matar-format.py src/include/*.h solvers/*.hpp
```
