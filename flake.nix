{
  description = "Python and PyTorch development environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.11";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
        };
      };

      # Library path for fixing dependencies
      pythonLdLibPath = pkgs.lib.makeLibraryPath (with pkgs; [
        zlib
        zstd
        stdenv.cc.cc
        curl
        openssl
        attr
        libssh
        bzip2
        libxml2
        acl
        libsodium
        util-linux
        xz
        systemd
      ]);

      # NVIDIA driver package
      nvidiaPackage = pkgs.linuxPackages.nvidiaPackages.stable;

      # Function to create a Python development shell
      makePythonShell = python: let
        pythonEnv = python.withPackages (ps: with ps; [
          pip
          virtualenv
        ]);
      in pkgs.mkShell {
        buildInputs = with pkgs; [
          pythonEnv

          # CUDA toolkit and related packages
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          nvidiaPackage

          # OpenGL and X11 libraries
          libGLU
          libGL
          xorg.libXi
          xorg.libXmu
          freeglut
          xorg.libXext
          xorg.libX11
          xorg.libXv
          xorg.libXrandr

          # Standard libraries
          zlib
          ncurses
          stdenv.cc
          binutils

          # Development tools
          git
          gnumake
          cmake
        ];

        shellHook = ''
          # Python dependencies
          export LD_LIBRARY_PATH="${pythonLdLibPath}:$LD_LIBRARY_PATH"
          
          # Compiler configuration
          export CC="${pkgs.gcc}/bin/gcc"
          export EXTRA_CCFLAGS="-I/usr/include"
          
          # CUDA configuration
          export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
          export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
          export LD_LIBRARY_PATH="${nvidiaPackage}/lib:${pkgs.ncurses}/lib:$LD_LIBRARY_PATH"
          export EXTRA_LDFLAGS="-L/lib -L${nvidiaPackage}/lib"
          
          # Triton-specific fixes for NixOS
          export TRITON_LIBCUDA_PATH="${nvidiaPackage}/lib/libcuda.so"
          export TRITON_PTXAS_PATH="${pkgs.cudaPackages.cudatoolkit}/bin/ptxas"
          
          echo "$(python --version) development environment ready."
        '';
      };

    in
    {
      devShells.${system} = {
        default = makePythonShell pkgs.python3;
        python312 = makePythonShell pkgs.python312;
        python313 = makePythonShell pkgs.python313;
        python314 = makePythonShell pkgs.python314;
      };
    };
}