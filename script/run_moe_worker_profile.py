# SPDX-License-Identifier: Apache-2.0
"""Compare individual worker policies on the ordinary-stream late-wait MTE baseline."""

from run_cpp_mte_profile_compare import main

if __name__ == "__main__":
    raise SystemExit(main(worker_compare=True))
