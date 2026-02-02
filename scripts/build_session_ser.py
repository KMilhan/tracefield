#!/usr/bin/env python3
"""Build a single session SER file by concatenating per-sub SERs.

This preserves the header from the first SER and appends frame data from
each subsequent file (skipping their headers). Frame count is left as-is
because analyzeFileRange derives the count from file size.
"""

from pathlib import Path


def main() -> int:
    base = Path("testdata/m42/secondary")
    out = base / "guide2.ser"
    subs = sorted(base.glob("guide_sub_*.ser"))
    if not subs:
        raise SystemExit("no per-sub SER files found")

    header = subs[0].read_bytes()[:178]
    out.write_bytes(header)
    with out.open("ab") as w:
        for path in subs:
            data = path.read_bytes()
            if len(data) < 178:
                raise SystemExit(f"short SER: {path}")
            w.write(data[178:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
