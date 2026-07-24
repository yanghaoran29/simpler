# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Query the runtime-visible AICore cluster count (block_dim) for ST/orch sizing.

Mirrors DeviceRunnerBase::query_max_block_dim / sim PLATFORM_MAX_BLOCKDIM:
onboard uses ACL stream cube-core limit (capped by arch ceiling); sim uses the
arch ceiling. Use this helper when golden / buffer sizing must match orch's
rt_available_cluster_count(). DeviceRunner always resolves block_dim from
ACL / PLATFORM_MAX_BLOCKDIM — CallConfig no longer carries it.
"""

from __future__ import annotations

from .platform_info import parse_platform

# Matches src/{arch}/platform/include/common/platform_config.h
_PLATFORM_MAX_BLOCKDIM = {
    "a2a3": 24,
    "a5": 36,
}

# aclrtGetStreamResLimit type: ACL_RT_DEV_RES_CUBE_CORE
_ACL_RT_DEV_RES_CUBE_CORE = 0

_acl_inited = False


def platform_max_block_dim(platform: str) -> int:
    """Compile-time ceiling for the arch (not the live ACL limit)."""
    arch, _ = parse_platform(platform)
    return _PLATFORM_MAX_BLOCKDIM[arch]


def available_block_dim(platform: str, device_id: int = 0) -> int:
    """User-visible cluster count for this platform / device.

    - sim: PLATFORM_MAX_BLOCKDIM for the arch
    - onboard: min(ACL cube cores, PLATFORM_MAX_BLOCKDIM)
    """
    arch, variant = parse_platform(platform)
    ceiling = _PLATFORM_MAX_BLOCKDIM[arch]
    if variant == "sim":
        return ceiling

    global _acl_inited
    try:
        import acl  # type: ignore
    except ImportError:
        return ceiling

    if not _acl_inited:
        ret = acl.init()
        if ret != 0:
            return ceiling
        _acl_inited = True

    if acl.rt.set_device(device_id) != 0:
        return ceiling
    stream, ret = acl.rt.create_stream()
    if ret != 0:
        return ceiling
    try:
        cube, ret = acl.rt.get_stream_res_limit(stream, _ACL_RT_DEV_RES_CUBE_CORE)
        if ret != 0 or cube is None or int(cube) < 1:
            return ceiling
        return min(int(cube), ceiling)
    finally:
        acl.rt.destroy_stream(stream)
