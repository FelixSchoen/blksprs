{
  description = "Python development environment with optional GPU support";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs =
    { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      # On NixOS, the NVIDIA driver is managed by the system configuration.
      # We use /run/opengl-driver which is symlinked by NixOS to the correct driver.
      # This avoids driver version mismatches between the flake and system.
      nixosDriverPath = "/run/opengl-driver";

      # CUDA toolkit
      cudaToolkit = pkgs.cudaPackages.cudatoolkit;

      # Base library dependencies (always included)
      baseLibs = with pkgs; [
        stdenv.cc.cc.lib
        zlib
        zstd
        openssl
        curl
        bzip2
        xz
        libxml2
        util-linux
        systemd
        ncurses
        attr
        libssh
        acl
        libsodium
      ];

      # GPU libraries (CUDA + cuDNN + Graphics/X11)
      gpuLibs = with pkgs; [
        # Graphics/X11
        libGL
        libGLU
        xorg.libX11
        xorg.libXext
        xorg.libXrender
        xorg.libXrandr
        xorg.libXi
        xorg.libXcursor
        xorg.libXfixes
        xorg.libXmu
        xorg.libXv
        libxkbcommon
        freeglut
      ];

      # Configurable Python shell builder
      makePythonShell =
        {
          python ? pkgs.python313,
          withGpu ? false,
        }:
        let
          pythonEnv = python.withPackages (
            ps: with ps; [
              pip
              virtualenv
            ]
          );

          # Build package list based on options
          # Note: We do NOT include a driver package - we use the system driver
          gpuPackages =
            if withGpu then
              [
                cudaToolkit
                pkgs.cudaPackages.cudnn
              ]
              ++ gpuLibs
            else
              [ ];

          # Build library path - include NixOS driver path for GPU
          libPath = pkgs.lib.makeLibraryPath (baseLibs ++ (if withGpu then gpuLibs else [ ]));

          # Shell hook for GPU/CUDA configuration
          gpuShellHook =
            if withGpu then
              ''
                # CUDA configuration
                export CUDA_PATH="${cudaToolkit}"
                export CUDA_HOME="${cudaToolkit}"
                export CUDA_DEVICE_ORDER="PCI_BUS_ID"
                export CUDA_LAUNCH_BLOCKING=0

                # Use NixOS system driver via /run/opengl-driver
                # This is the correct way to access GPU drivers on NixOS
                export LD_LIBRARY_PATH="${nixosDriverPath}/lib:${cudaToolkit}/lib:${cudaToolkit}/lib64:${pkgs.cudaPackages.cudnn}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

                # Triton-specific configuration for NixOS
                # Point to system driver for libcuda.so
                export TRITON_LIBCUDA_PATH="${nixosDriverPath}/lib"
                export TRITON_PTXAS_PATH="${cudaToolkit}/bin/ptxas"
                export TRITON_CUOBJDUMP_PATH="${cudaToolkit}/bin/cuobjdump"
                export TRITON_NVDISASM_PATH="${cudaToolkit}/bin/nvdisasm"

                # Triton cache and compatibility
                export TRITON_CACHE_DIR="/tmp/triton-cache-$UID"
                mkdir -p $TRITON_CACHE_DIR
                export TRITON_IGNORE_UNKNOWN_PARAMETERS=1
                export TRITON_PRINT_AUTOTUNING=0  # Set to 1 for debugging
              ''
            else
              "";

          gpuStatus = if withGpu then "✓ GPU stack enabled (CUDA + cuDNN + Graphics)" else "✗ GPU disabled";

        in
        pkgs.mkShell {
          name = "python-dev";

          packages = [
            pythonEnv

            # Build tools
            pkgs.gcc
            pkgs.gnumake
            pkgs.cmake
            pkgs.pkg-config
            pkgs.binutils

            # Version control
            pkgs.git
          ]
          ++ baseLibs
          ++ gpuPackages;

          shellHook = ''
            # Library paths
            export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

            # Compiler configuration
            export CC="${pkgs.gcc}/bin/gcc"
            export CXX="${pkgs.gcc}/bin/g++"

            ${gpuShellHook}

            # Auto-activate venv if it exists
            if [ -d ".venv" ]; then
              source .venv/bin/activate
            fi

            # Environment info
            echo ""
            echo "🐍 $(python --version) development environment"
            echo ""
            echo "📦 Virtual environment:"
            if [ -d ".venv" ]; then
              echo "   ✓ .venv activated"
            else
              echo "   ✗ No .venv found. Run: python -m venv .venv && source .venv/bin/activate"
            fi
            echo ""
            echo "🔧 Features:"
            echo "   ${gpuStatus}"
            echo ""
          '';
        };

    in
    {
      devShells.${system} = {
        # Default: basic Python, no GPU
        default = makePythonShell { };

        # GPU: full stack (CUDA + cuDNN + Graphics)
        gpu = makePythonShell { withGpu = true; };

        # Python version variants - default (no GPU)
        py312 = makePythonShell { python = pkgs.python312; };
        py313 = makePythonShell { python = pkgs.python313; };
        py314 = makePythonShell { python = pkgs.python314; };

        # Python version variants - GPU
        gpu-py312 = makePythonShell {
          python = pkgs.python312;
          withGpu = true;
        };
        gpu-py313 = makePythonShell {
          python = pkgs.python313;
          withGpu = true;
        };
        gpu-py314 = makePythonShell {
          python = pkgs.python314;
          withGpu = true;
        };
      };
    };
}
