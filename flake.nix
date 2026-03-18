{
  description = "hf-litmus: Continuous model compatibility testing for Tron";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        python = pkgs.python313;

        pythonEnv = python.withPackages (ps:
          with ps; [
            # Runtime
            huggingface-hub
            protobuf
            pyyaml
            filelock
            # Dev / test
            pytest
            pytest-cov
            pytest-timeout
            hypothesis
            setuptools
          ]);

        src = pkgs.lib.cleanSource ./.;

        # Shared helper: run a command in a derivation with the source tree
        mkCheck = name: script:
          pkgs.runCommand "hf-litmus-${name}" {
            inherit src;
            nativeBuildInputs = [pythonEnv pkgs.ruff pkgs.git];
          } ''
            cp -r $src src && chmod -R u+w src && cd src
            export HOME=$TMPDIR
            export PYTHONPATH=$PWD:$PYTHONPATH
            ${script}
            touch $out
          '';
      in {
        packages.default = python.pkgs.buildPythonPackage {
          pname = "hf-litmus";
          version = "0.1.0";
          inherit src;
          format = "pyproject";

          build-system = [python.pkgs.setuptools];
          dependencies = with python.pkgs; [
            huggingface-hub
            protobuf
            pyyaml
            filelock
          ];

          doCheck = false; # tests run via checks.tests
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.ruff
            pkgs.lefthook
            pkgs.git
          ];

          shellHook = ''
            export PYTHONPATH=$PWD:$PYTHONPATH
            lefthook install 2>/dev/null || true
          '';
        };

        checks = {
          # Lint: all ruff rules, zero tolerance
          lint = mkCheck "lint" ''
            ruff check .
          '';

          # Format: ruff format check
          format = mkCheck "format" ''
            ruff format --check .
          '';

          # Tests with coverage
          tests = mkCheck "tests" ''
            python -m pytest tests/ -x -q --timeout=30
          '';

          # Coverage threshold
          coverage = mkCheck "coverage" ''
            python -m pytest tests/ -q --timeout=30 \
              --cov=hf_litmus --cov-report=term-missing \
              --cov-fail-under=10
          '';

          # Build: verify the package builds cleanly
          build = self.packages.${system}.default;

          # Fuzz: hypothesis tests (included in test suite)
          fuzz = mkCheck "fuzz" ''
            python -m pytest tests/ -x -q --timeout=60 \
              -k "fuzz or hypothesis or property"
          '';
        };
      }
    );
}
