# SPDX-License-Identifier: Apache-2.0
"""Compare callback, default MTE and combined worker policies with profiles."""

from run_cpp_mte_profile_compare import main

if __name__ == "__main__":
    raise SystemExit(main(handoff_compare=True))
