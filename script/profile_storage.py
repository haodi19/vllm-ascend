"""采集和离线解析前检查磁盘；不删除用户文件。"""

import shutil
from pathlib import Path

GIB = 1 << 30
PROFILE_RESERVE_GIB = 20


def require_space(path, minimum_gib=PROFILE_RESERVE_GIB):
    existing = Path(path).resolve()
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free / GIB
    print(f"STORAGE: {existing}; free={free:.2f} GiB; required={minimum_gib:.2f} GiB", flush=True)
    if free < minimum_gib:
        raise RuntimeError(
            f"磁盘空间不足：{existing} 剩余 {free:.2f} GiB，需要至少 {minimum_gib:.2f} GiB。"
            "请自行清理不需要的结果，或将 --output-dir 指向有空间的磁盘；不要继续复制/解析残缺采集。"
        )


def require_raw_data(worker):
    profiles = list(Path(worker).glob("PROF_*"))
    has_data = any(
        entry.is_file() and entry.stat().st_size > 0
        for profile in profiles
        for directory in profile.glob("device_*/data")
        for entry in directory.rglob("*")
    )
    has_start = any(list(profile.glob("device_*/start_info.*")) for profile in profiles)
    if not has_data or not has_start:
        raise ValueError(f"{worker.name}: 缺少设备原始数据或 start_info，重新解析无法补回，需重新采集。")
