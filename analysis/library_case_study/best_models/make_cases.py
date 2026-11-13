import json, subprocess, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
SETS = {"stdin": "systems_stdin", "internal": "systems_internal_data", "alg": "algorithm_libraries", "talg": "the_algorithms"}
TEXT = {"stdin/leveldb_hash", "stdin/zlib_adler32", "stdin/xxhash32", "stdin/ngx_murmurhash2", "stdin/redis_siphash", "stdin/git_levenshtein",
        "stdin/llvm_edit_distance", "stdin/redis_stringmatch", "alg/acl_z_algorithm", "alg/acl_suffix_array", "alg/kactl_kmp"}
hi = lambda s: "".join(chr(0xC0 + ord(c) % 64) if c.isascii() and c.isalpha() else c for c in s)
C = {}
for tag, d in SETS.items():
    for k, c in json.load(open(HERE.parent / d / "cases.json")).items():
        key = f"{tag}/{k}"
        if key in TEXT:  # one added non-ASCII test, answered by the original program
            with tempfile.TemporaryDirectory() as t:
                src, b = Path(t) / "o.cpp", Path(t) / "o.bin"; src.write_text(c["code"])
                subprocess.run(["g++", "-O3", "-std=c++17", "-static", str(src), "-o", str(b)], check=True, capture_output=True)
                inp = hi(c["test_inputs"][5]); r = subprocess.run([str(b)], input=inp.encode("latin-1"), capture_output=True, timeout=60)
                if r.returncode or not r.stdout.strip(): print("non-ASCII input outside the function's domain:", key)
                else: c["test_inputs"] = c["test_inputs"] + [inp]; c["test_outputs"] = c["test_outputs"] + [r.stdout.decode("latin-1").strip()]
        C[key] = c
json.dump(C, open(HERE / "cases.json", "w"))
print(len(C), "functions,", sum(len(c["test_inputs"]) == 11 for c in C.values()), "with a non-ASCII test")
