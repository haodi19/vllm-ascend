# SPDX-License-Identifier: Apache-2.0
"""Compare direct and ordinary load streams with the same late-wait MTE path."""

from run_cpp_mte_profile_compare import main

if __name__ == "__main__":
    raise SystemExit(main(stream_compare=True))
