{
  description = "Development environment for blksprs with optional GPU support";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-26.05";
  };

  outputs =
    { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgsBase = import nixpkgs {
        inherit system;
      };
      pkgsGpu = import nixpkgs {
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
      cudaToolkit = pkgsGpu.cudaPackages.cudatoolkit;

      # Base library dependencies (always included)
      baseLibs =
        pkgs: with pkgs; [
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

      # GPU libraries used by optional graphics-dependent tooling. PyTorch
      # wheels provide their own CUDA runtime libraries and cuDNN.
      gpuLibs = with pkgsGpu; [
        # Graphics/X11
        libGL
        libGLU
        libx11
        libxext
        libxrender
        libxrandr
        libxi
        libxcursor
        libxfixes
        libxmu
        libxv
        libxkbcommon
        freeglut
      ];

      # Configurable Python shell builder
      makePythonShell =
        pkgs:
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

          basePackages = baseLibs pkgs;

          # Build package list based on options. We include the toolkit for
          # Triton's compiler tools, but deliberately omit cuDNN and CUDA
          # runtime library paths: binary PyTorch wheels must load the matching
          # versions they bundle themselves.
          gpuPackages =
            if withGpu then
              [
                cudaToolkit
              ]
              ++ gpuLibs
            else
              [ ];

          # Build library path - include NixOS driver path for GPU
          libPath = pkgs.lib.makeLibraryPath (basePackages ++ (if withGpu then gpuLibs else [ ]));

          # Shell hook for GPU/CUDA configuration
          gpuShellHook =
            if withGpu then
              ''
                # CUDA configuration
                export CUDA_PATH="${cudaToolkit}"
                export CUDA_HOME="${cudaToolkit}"
                export CUDA_DEVICE_ORDER="PCI_BUS_ID"
                export CUDA_LAUNCH_BLOCKING=0

                # Expose only the system NVIDIA driver. Prepending the Nix CUDA
                # runtime or cuDNN here can make them override the versions
                # bundled with PyTorch and break otherwise valid CUDA calls.
                export LD_LIBRARY_PATH="${nixosDriverPath}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

                # Triton-specific configuration for NixOS
                # Point to system driver for libcuda.so
                export TRITON_LIBCUDA_PATH="${nixosDriverPath}/lib"
                export TRITON_PTXAS_PATH="${cudaToolkit}/bin/ptxas"
                export TRITON_CUOBJDUMP_PATH="${cudaToolkit}/bin/cuobjdump"
                export TRITON_NVDISASM_PATH="${cudaToolkit}/bin/nvdisasm"

                # Triton cache and compatibility
                export TRITON_CACHE_DIR="/var/tmp/triton-cache-$UID"
                mkdir -p "$TRITON_CACHE_DIR"
                export TRITON_IGNORE_UNKNOWN_PARAMETERS=1
                export TRITON_PRINT_AUTOTUNING=0  # Set to 1 for debugging
              ''
            else
              "";

          gpuStatus =
            if withGpu then "✓ GPU tooling enabled (system driver + CUDA compiler tools)" else "✗ GPU disabled";

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

            # Media helpers used by music/data workflows
            pkgs.ffmpeg
            pkgs.fluidsynth
          ]
          ++ basePackages
          ++ gpuPackages;

          shellHook = ''
            # Library paths
            export LD_LIBRARY_PATH="${libPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            nix_python_version="$("${pythonEnv}/bin/python" --version 2>&1)"

            # Compiler configuration
            export CC="${pkgs.gcc}/bin/gcc"
            export CXX="${pkgs.gcc}/bin/g++"

            ${gpuShellHook}

            # Auto-activate the local venv only when its interpreter works.
            venv_path="$PWD/.venv"
            venv_bin="$venv_path/bin"
            venv_status="missing"
            if [ -x "$venv_bin/python" ] && "$venv_bin/python" -c 'import sys' >/dev/null 2>&1; then
              venv_status="usable"
              # The shared zsh prompt already renders the active venv name.
              # Keep Python's activate script from prepending its own prompt
              # fragment and avoid re-sourcing when direnv reloads the same venv.
              export VIRTUAL_ENV_DISABLE_PROMPT=1
              if [ "''${VIRTUAL_ENV:-}" != "$venv_path" ]; then
                source "$venv_path/bin/activate"
              fi

              # If nix develop inherits VIRTUAL_ENV from direnv, activation is
              # skipped above; keep the local venv ahead of Nix's Python tools.
              case "$PATH" in
                "$venv_path/bin":*) ;;
                *) export PATH="$venv_path/bin:$PATH" ;;
              esac
            elif [ -d "$venv_path" ]; then
              venv_status="stale"

              # Do not inherit a dead local environment from direnv or a
              # parent shell. Remove only this repository's venv from PATH.
              if [ "''${VIRTUAL_ENV:-}" = "$venv_path" ]; then
                unset VIRTUAL_ENV
              fi
              filtered_path=""
              previous_ifs="$IFS"
              IFS=:
              for path_entry in $PATH; do
                if [ "$path_entry" != "$venv_bin" ]; then
                  if [ -z "$filtered_path" ]; then
                    filtered_path="$path_entry"
                  else
                    filtered_path="$filtered_path:$path_entry"
                  fi
                fi
              done
              IFS="$previous_ifs"
              export PATH="$filtered_path"
            fi

            # Environment info
            active_python_version="$(python --version 2>&1)"
            echo ""
            echo "🐍 $active_python_version development environment"
            if [ "$active_python_version" != "$nix_python_version" ]; then
              echo "   Nix shell provides: $nix_python_version"
            fi
            echo ""
            echo "📦 Virtual environment:"
            if [ "$venv_status" = "usable" ]; then
              echo "   ✓ .venv activated"
              if [ "$active_python_version" != "$nix_python_version" ]; then
                echo "   ! Recreate .venv to move it onto the current Nix Python"
              fi
            elif [ "$venv_status" = "stale" ]; then
              echo "   ✗ .venv exists but its Python executable is unusable"
              echo "   ! Recreate it with: python -m venv --clear .venv"
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
        default = makePythonShell pkgsBase { };

        # GPU: system driver, PyTorch-bundled runtime, and CUDA compiler tools
        gpu = makePythonShell pkgsGpu { withGpu = true; };

        # Python version variants - default (no GPU)
        py312 = makePythonShell pkgsBase { python = pkgsBase.python312; };
        py313 = makePythonShell pkgsBase { python = pkgsBase.python313; };
        py314 = makePythonShell pkgsBase { python = pkgsBase.python314; };

        # Python version variants - GPU
        gpu-py312 = makePythonShell pkgsGpu {
          python = pkgsGpu.python312;
          withGpu = true;
        };
        gpu-py313 = makePythonShell pkgsGpu {
          python = pkgsGpu.python313;
          withGpu = true;
        };
        gpu-py314 = makePythonShell pkgsGpu {
          python = pkgsGpu.python314;
          withGpu = true;
        };
      };
    };
}
