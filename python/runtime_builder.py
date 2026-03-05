import importlib.util
import logging
import os
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from runtime_compiler import RuntimeCompiler
from kernel_compiler import KernelCompiler

logger = logging.getLogger(__name__)


class RuntimeBuilder:
    """Discovers and builds runtime implementations from src/runtime/.

    Accepts a platform selection to provide correctly configured
    RuntimeCompiler and KernelCompiler instances. Runtime and platform
    are orthogonal — the same runtime (e.g., host_build_graph) can
    be compiled for any platform (e.g., a2a3, a2a3sim).
    """

    def __init__(self, platform: str = "a2a3"):
        """
        Initialize RuntimeBuilder with platform selection.

        Args:
            platform: Target platform ("a2a3" or "a2a3sim")
        """
        self.platform = platform

        runtime_root = Path(__file__).parent.parent
        self.runtime_root = runtime_root
        self.runtime_dir = runtime_root / "src" / "runtime"

        # Discover available runtime implementations
        self._runtimes = {}
        if self.runtime_dir.is_dir():
            for entry in sorted(self.runtime_dir.iterdir()):
                config_path = entry / "build_config.py"
                if entry.is_dir() and config_path.is_file():
                    self._runtimes[entry.name] = config_path

        # Create platform-configured compilers
        self._runtime_compiler = RuntimeCompiler.get_instance(platform=platform)
        self._kernel_compiler = KernelCompiler(platform=platform)

        # Pending static objects for a2a3sim builds (set via set_kernel_objects())
        self._pending_extra_objects: list = []
        self._pending_dispatch_source: str = ""

    def get_runtime_compiler(self) -> RuntimeCompiler:
        """Return the RuntimeCompiler configured for this platform."""
        return self._runtime_compiler

    def get_kernel_compiler(self) -> KernelCompiler:
        """Return the KernelCompiler configured for this platform."""
        return self._kernel_compiler

    def list_runtimes(self) -> list:
        """Return names of discovered runtime implementations."""
        return list(self._runtimes.keys())

    def set_kernel_objects(
        self,
        extra_obj_paths: list,
        dispatch_source_path: str,
    ) -> None:
        """Register pre-compiled kernel/orch objects for the next a2a3sim build.

        For a2a3sim static linking: the orchestration .o and kernel_N.o files
        must be available before build() is called. This method stores their
        paths so that the host CMake build can link them in.

        Args:
            extra_obj_paths: List of .o file paths (orch.o + kernel_N.o)
            dispatch_source_path: Path to generated kernel_dispatch.cpp
        """
        self._pending_extra_objects = list(extra_obj_paths)
        self._pending_dispatch_source = dispatch_source_path

    def build(self, name: str) -> tuple:
        """
        Build a specific runtime implementation by name.

        For a2a3sim: builds sequentially (aicore.a → aicpu.a → host.so).
        The host.so links in the static archives plus any objects previously
        registered via set_kernel_objects().

        For other platforms: compiles all three targets in parallel.

        Args:
            name: Name of the runtime implementation (e.g. 'host_build_graph')

        Returns:
            Tuple of (host_binary, aicpu_binary, aicore_binary) as bytes

        Raises:
            ValueError: If the named runtime is not found
        """
        if name not in self._runtimes:
            available = ", ".join(self._runtimes.keys()) or "(none)"
            raise ValueError(
                f"Runtime '{name}' not found. Available runtimes: {available}"
            )

        config_path = self._runtimes[name]
        config_dir = config_path.parent

        # Load build_config.py
        spec = importlib.util.spec_from_file_location("build_config", config_path)
        build_config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(build_config_module)
        build_config = build_config_module.BUILD_CONFIG

        compiler = self._runtime_compiler

        # Prepare configs for all three targets
        aicore_cfg = build_config["aicore"]
        aicore_include_dirs = [str((config_dir / p).resolve()) for p in aicore_cfg["include_dirs"]]
        aicore_source_dirs = [str((config_dir / p).resolve()) for p in aicore_cfg["source_dirs"]]

        aicpu_cfg = build_config["aicpu"]
        aicpu_include_dirs = [str((config_dir / p).resolve()) for p in aicpu_cfg["include_dirs"]]
        aicpu_source_dirs = [str((config_dir / p).resolve()) for p in aicpu_cfg["source_dirs"]]

        host_cfg = build_config["host"]
        host_include_dirs = [str((config_dir / p).resolve()) for p in host_cfg["include_dirs"]]
        host_source_dirs = [str((config_dir / p).resolve()) for p in host_cfg["source_dirs"]]

        if self.platform == "a2a3sim":
            return self._build_a2a3sim(
                compiler,
                aicore_include_dirs, aicore_source_dirs,
                aicpu_include_dirs, aicpu_source_dirs,
                host_include_dirs, host_source_dirs,
            )

        # Non-a2a3sim: compile all three targets in parallel
        logger.info("Compiling AICore, AICPU, Host in parallel...")

        with ThreadPoolExecutor(max_workers=3) as executor:
            fut_aicore = executor.submit(compiler.compile, "aicore", aicore_include_dirs, aicore_source_dirs)
            fut_aicpu = executor.submit(compiler.compile, "aicpu", aicpu_include_dirs, aicpu_source_dirs)
            fut_host = executor.submit(compiler.compile, "host", host_include_dirs, host_source_dirs)

            aicore_binary = fut_aicore.result()
            aicpu_binary = fut_aicpu.result()
            host_binary = fut_host.result()

        logger.info("Build complete!")
        return (host_binary, aicpu_binary, aicore_binary)

    def _build_a2a3sim(
        self,
        compiler,
        aicore_include_dirs, aicore_source_dirs,
        aicpu_include_dirs, aicpu_source_dirs,
        host_include_dirs, host_source_dirs,
    ) -> tuple:
        """Sequential build for a2a3sim static linking.

        Builds aicore.a and aicpu.a first, writes them to persistent temp files,
        then builds host.so linking in the static archives plus any registered
        kernel/orch objects.

        Returns:
            Tuple of (host_binary, aicpu_binary, aicore_binary) as bytes
        """
        logger.info("a2a3sim: Building AICore static library...")
        aicore_binary = compiler.compile("aicore", aicore_include_dirs, aicore_source_dirs)

        logger.info("a2a3sim: Building AICPU static library...")
        aicpu_binary = compiler.compile("aicpu", aicpu_include_dirs, aicpu_source_dirs)

        # Write .a files to persistent temp locations for the host CMake build
        aicore_fd, aicore_lib_path = tempfile.mkstemp(prefix="libaicore_", suffix=".a")
        aicpu_fd, aicpu_lib_path = tempfile.mkstemp(prefix="libaicpu_", suffix=".a")

        try:
            os.write(aicore_fd, aicore_binary)
            os.close(aicore_fd)
            os.write(aicpu_fd, aicpu_binary)
            os.close(aicpu_fd)

            # Configure static build extras for the host compile step
            extras = {
                "aicpu_lib": aicpu_lib_path,
                "aicore_lib": aicore_lib_path,
            }
            if self._pending_extra_objects:
                extras["extra_objects"] = self._pending_extra_objects
            if self._pending_dispatch_source:
                extras["dispatch_source"] = self._pending_dispatch_source

            compiler.set_static_build_extras(extras)

            logger.info("a2a3sim: Building Host shared library (with static deps)...")
            host_binary = compiler.compile("host", host_include_dirs, host_source_dirs)
        finally:
            if os.path.exists(aicore_lib_path):
                os.unlink(aicore_lib_path)
            if os.path.exists(aicpu_lib_path):
                os.unlink(aicpu_lib_path)
            # Clear static build extras after host build
            compiler.set_static_build_extras({})
            self._pending_extra_objects = []
            self._pending_dispatch_source = ""

        logger.info("a2a3sim: Build complete!")
        return (host_binary, aicpu_binary, aicore_binary)
