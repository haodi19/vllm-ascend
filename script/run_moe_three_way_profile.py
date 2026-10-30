# SPDX-License-Identifier: Apache-2.0
"""Compare callback and both MTE load stream modes with separate profiles."""

from run_cpp_mte_profile_compare import main

if __name__ == "__main__":
    raise SystemExit(main(three_way=True))
