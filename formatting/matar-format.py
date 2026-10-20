#!/usr/bin/env python3
"""Post-processor for clang-format: canonical layout for MATAR parallel macros.

clang-format treats FOR_ALL(k, 0, n, j, 0, n, i, 0, n, {...}) as a call with
nine scalar arguments and packs or splits them arbitrarily. This pass rewrites
each call into MATAR's canonical layout:

    DO_REDUCE_MAX(k, 0, 10,
                  j, 0, 10,
                  i, 0, 10,
                  loc_max, {
        <body — clang-format's line layout, re-indented to macro indent + 4>
    }, result);

Bodies keep clang-format's internal formatting; only their indentation shifts.
Comments and string literals are masked out before any paren/brace matching,
so braces inside comments (including commented-out code) cannot confuse it.

Usage: matar-format.py <file> [<file> ...]   (edits in place; the pipeline
clang-format-then-reflow is idempotent as a pair)
"""
import re
import sys

# Longer names must precede their prefixes (regex alternation is first-match):
# e.g. FOR_REDUCE_SUM_SECOND before FOR_REDUCE_SUM.
MACROS = (
    # host-side twins first: they are longer names sharing a prefix with the
    # device macros below
    "FOR_ALL_HOST_CLASS", "DO_ALL_HOST_CLASS",
    "FOR_ALL_HOST", "DO_ALL_HOST",
    "FOR_REDUCE_SUM_HOST_CLASS", "FOR_REDUCE_MAX_HOST_CLASS",
    "FOR_REDUCE_MIN_HOST_CLASS", "FOR_REDUCE_PRODUCT_HOST_CLASS",
    "FOR_REDUCE_SUM_HOST", "FOR_REDUCE_MAX_HOST",
    "FOR_REDUCE_MIN_HOST", "FOR_REDUCE_PRODUCT_HOST",
    "DO_REDUCE_SUM_HOST", "DO_REDUCE_MAX_HOST", "DO_REDUCE_MIN_HOST",
    "RUN_HOST_CLASS", "RUN_HOST",
    "FOR_ALL_CLASS", "DO_ALL_CLASS",
    "FOR_ALL", "DO_ALL",
    "FOR_REDUCE_SUM_CLASS", "FOR_REDUCE_MAX_CLASS", "FOR_REDUCE_MIN_CLASS",
    "FOR_REDUCE_PRODUCT_CLASS",
    "FOR_REDUCE_SUM_SECOND", "FOR_REDUCE_SUM_THIRD",
    "FOR_REDUCE_MAX_SECOND", "FOR_REDUCE_MIN_SECOND",
    "DO_REDUCE_SUM_SECOND", "DO_REDUCE_SUM_THIRD",
    "DO_REDUCE_MAX_THIRD", "DO_REDUCE_MIN_THIRD",
    "FOR_REDUCE_SUM", "FOR_REDUCE_MAX", "FOR_REDUCE_MIN", "FOR_REDUCE_PRODUCT",
    "DO_REDUCE_SUM", "DO_REDUCE_MAX", "DO_REDUCE_MIN",
    "FOR_FIRST", "FOR_SECOND", "FOR_THIRD",
    "DO_FIRST", "DO_SECOND", "DO_THIRD",
)
MACRO_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(MACROS) + r")\s*\(")
INDENT = "    "


def build_mask(text):
    """mask[i]: inside a comment or string. cmask[i]: inside a comment only.

    The two differ where it matters: a trailing kernel-name argument is a
    string literal, so the scan that skips past whitespace/commas/comments
    between the body and the next argument must not skip strings.
    """
    mask = [False] * len(text)
    cmask = [False] * len(text)
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                mask[k] = True
                cmask[k] = True
            i = j
        elif c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            for k in range(i, j):
                mask[k] = True
                cmask[k] = True
            i = j
        elif c in "\"'":
            quote, j = c, i + 1
            while j < n and text[j] != quote:
                j += 2 if text[j] == "\\" else 1
            j += 1
            for k in range(i, min(j, n)):
                mask[k] = True
            i = j
        else:
            i += 1
    return mask, cmask


def split_top_level(text, mask, offset):
    """Split on commas at paren/bracket/brace depth zero, outside comments/strings."""
    args, depth, start = [], 0, 0
    for i, c in enumerate(text):
        if mask[offset + i]:
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def parse_call(text, mask, open_paren):
    """Return (body_open, body_close, call_close) indices, or None.

    body_open/body_close bracket the '{...}' body argument; call_close is the
    index of the macro's closing ')'.
    """
    depth, i, body_open, body_close = 0, open_paren, None, None
    n = len(text)
    while i < n:
        c = text[i]
        if mask[i]:
            i += 1
            continue
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return (body_open, body_close, i) if body_close else None
        elif c == "{" and depth == 1 and body_open is None:
            body_open = i
            b, j = 0, i
            while j < n:
                if not mask[j]:
                    if text[j] == "{":
                        b += 1
                    elif text[j] == "}":
                        b -= 1
                        if b == 0:
                            body_close = j
                            break
                j += 1
            if body_close is None:
                return None
            i = body_close
        i += 1
    return None


def reindent_body(body_inner, target):
    """Shift the body's lines so they sit at `target` indentation."""
    lines = body_inner.split("\n")
    if len(lines) == 1:
        stmt = lines[0].strip()
        return "\n" + target + INDENT + stmt + "\n" + target if stmt else ""
    # First line may carry content (e.g. a nested macro the recursion just
    # expanded out of a single packed line) — it must not be dropped.
    head = lines[0].strip()
    rest = lines[1:]
    content = [ln for ln in rest if ln.strip()]
    if not content and not head:
        return "\n" + target
    current = min(len(ln) - len(ln.lstrip()) for ln in content) if content else 0
    shifted = []
    if head:
        shifted.append(target + INDENT + head)
    for ln in rest:
        shifted.append(target + INDENT + ln[current:] if ln.strip() else "")
    while shifted and not shifted[-1]:
        shifted.pop()
    return "\n" + "\n".join(shifted) + "\n" + target


def reflow(source):
    mask, cmask = build_mask(source)
    out, pos = [], 0
    for m in MACRO_RE.finditer(source):
        if m.start() < pos or mask[m.start()]:
            continue  # already handled, or macro name inside a comment/string
        open_paren = m.end() - 1
        parsed = parse_call(source, mask, open_paren)
        if parsed is None:
            continue
        body_open, body_close, call_close = parsed
        line_start = source.rfind("\n", 0, m.start()) + 1
        indent = source[line_start:m.start()]
        if indent.strip():
            continue  # macro is not first on its line; don't touch
        header_text = source[open_paren + 1:body_open].rstrip().rstrip(",")
        header_args = split_top_level(header_text, mask, open_paren + 1)
        t = body_close + 1
        while t < call_close and (source[t] in " \t\n," or cmask[t]):
            t += 1
        trailing_args = split_top_level(source[t:call_close], mask, t)
        # recurse: nested parallel macros (hierarchical FOR_SECOND / REDUCE_*_SECOND
        # inside FOR_FIRST bodies) always get the expanded canonical layout too
        body_text = reflow(source[body_open + 1:body_close])
        body = reindent_body(body_text, indent)

        align = " " * (len(indent) + len(m.group(1)) + 1)
        n_triples = len(header_args) // 3
        header_lines = [", ".join(header_args[t * 3:t * 3 + 3]) for t in range(n_triples)]
        header_lines.extend(header_args[n_triples * 3:])
        header = (",\n" + align).join(header_lines)
        # RUN_HOST and friends take only a body: no leading argument list, so
        # no separating comma before the brace.
        header_prefix = header + ", " if header_lines else ""

        tail = "".join(", " + a for a in trailing_args)
        out.append(source[pos:m.start()])
        out.append(m.group(1) + "(" + header_prefix + "{" + body + "}" + tail + ")")
        pos = call_close + 1
    out.append(source[pos:])
    return "".join(out)


def main():
    for path in sys.argv[1:]:
        # surrogateescape: pass through non-UTF-8 bytes (stray Latin-1 chars
        # in old copyright headers) unchanged instead of erroring
        with open(path, encoding="utf-8", errors="surrogateescape") as f:
            original = f.read()
        # Iterate to a fixed point: deeply nested macros settle one level per
        # pass (nested regeneration recomputes its alignment from the previous
        # pass's placement).
        formatted = original
        for _ in range(10):
            again = reflow(formatted)
            if again == formatted:
                break
            formatted = again
        formatted = re.sub(r"[ \t]+\n", "\n", formatted)
        if formatted != original:
            with open(path, "w", encoding="utf-8", errors="surrogateescape") as f:
                f.write(formatted)
            print(f"reflowed: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
