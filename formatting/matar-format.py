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
Usage: matar-format.py <file> [<file> ...]   (edits in place, idempotent)
"""
import re
import sys

MACROS = (
    "FOR_ALL_CLASS", "DO_ALL_CLASS",
    "FOR_ALL", "DO_ALL",
    "FOR_REDUCE_SUM_CLASS", "FOR_REDUCE_MAX_CLASS", "FOR_REDUCE_MIN_CLASS",
    "FOR_REDUCE_PRODUCT_CLASS",
    "FOR_REDUCE_SUM", "FOR_REDUCE_MAX", "FOR_REDUCE_MIN", "FOR_REDUCE_PRODUCT",
    "DO_REDUCE_SUM", "DO_REDUCE_MAX", "DO_REDUCE_MIN",
)
MACRO_RE = re.compile(r"(?<![A-Za-z0-9_])(" + "|".join(MACROS) + r")\s*\(")
INDENT = "    "


def skip_string(text, i):
    quote = text[i]
    i += 1
    while i < len(text) and text[i] != quote:
        i += 2 if text[i] == "\\" else 1
    return i + 1


def split_top_level(text):
    """Split on commas at paren/bracket/brace depth zero, respecting strings."""
    args, depth, start, i = [], 0, 0, 0
    while i < len(text):
        c = text[i]
        if c in "\"'":
            i = skip_string(text, i)
            continue
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            args.append(text[start:i].strip())
            start = i + 1
        i += 1
    tail = text[start:].strip()
    if tail:
        args.append(tail)
    return args


def parse_call(text, open_paren):
    """Return (body_open, body_close, call_close) indices, or None.

    body_open/body_close bracket the '{...}' body argument; call_close is the
    index of the macro's closing ')'.
    """
    depth, i, body_open, body_close = 0, open_paren, None, None
    while i < len(text):
        c = text[i]
        if c in "\"'":
            i = skip_string(text, i)
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
            while j < len(text):
                cj = text[j]
                if cj in "\"'":
                    j = skip_string(text, j)
                    continue
                if cj == "{":
                    b += 1
                elif cj == "}":
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
    content = [ln for ln in lines[1:] if ln.strip()]
    if not content:
        return "\n" + target
    current = min(len(ln) - len(ln.lstrip()) for ln in content)
    shifted = []
    for ln in lines[1:]:
        shifted.append(target + INDENT + ln[current:] if ln.strip() else "")
    while shifted and not shifted[-1]:
        shifted.pop()
    return "\n" + "\n".join(shifted) + "\n" + target


def reflow(source):
    out, pos = [], 0
    for m in MACRO_RE.finditer(source):
        if m.start() < pos:
            continue
        open_paren = m.end() - 1
        parsed = parse_call(source, open_paren)
        if parsed is None:
            continue
        body_open, body_close, call_close = parsed
        line_start = source.rfind("\n", 0, m.start()) + 1
        indent = source[line_start:m.start()]
        if indent.strip():
            continue  # macro is not first on its line; don't touch
        header_args = split_top_level(source[open_paren + 1:body_open].rstrip().rstrip(","))
        trailing_args = split_top_level(source[body_close + 1:call_close].lstrip().lstrip(","))
        body = reindent_body(source[body_open + 1:body_close], indent)

        align = " " * (len(indent) + len(m.group(1)) + 1)
        n_triples = len(header_args) // 3
        header_lines = [", ".join(header_args[t * 3:t * 3 + 3]) for t in range(n_triples)]
        header_lines.extend(header_args[n_triples * 3:])
        header = (",\n" + align).join(header_lines)

        tail = "".join(", " + a for a in trailing_args)
        out.append(source[pos:m.start()])
        out.append(m.group(1) + "(" + header + ", {" + body + "}" + tail + ")")
        pos = call_close + 1
    out.append(source[pos:])
    return "".join(out)


def main():
    for path in sys.argv[1:]:
        with open(path, encoding="utf-8") as f:
            original = f.read()
        formatted = reflow(original)
        if formatted != original:
            with open(path, "w", encoding="utf-8") as f:
                f.write(formatted)
            print(f"reflowed: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
