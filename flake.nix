{
  description = "Rust HAL for the Raspberry Pi RP2040 / RP235x microcontrollers";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";
    utils.url = "github:numtide/flake-utils";
    rust-overlay.url = "github:oxalica/rust-overlay";
    rust-overlay.inputs.nixpkgs.follows = "nixpkgs";
    git-hooks.url = "github:cachix/git-hooks.nix";
    git-hooks.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    {
      nixpkgs,
      utils,
      rust-overlay,
      git-hooks,
      ...
    }:
    utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ rust-overlay.overlays.default ];
        };

        # Keep this version in sync with RUST_VERSION in .github/workflows/*.yml.
        rust = pkgs.rust-bin.stable."1.99.1".default.override {
          extensions = [
            "rust-src"
            "clippy"
            "rust-analyzer"
          ];
          targets = [
            "thumbv6m-none-eabi"
            "thumbv8m.main-none-eabihf"
            "riscv32imac-unknown-none-elf"
          ];
        };

        pre-commit-check = git-hooks.lib.${system}.run {
          src = ./.;
          hooks = {
            # Rust. All workspaces use edition 2021.
            rustfmt = {
              enable = true;
              # The default entry runs cargo-fmt, which requires a Cargo.toml at
              # the repo root; this repo is several independent workspaces, so
              # format the staged files directly with the pinned rustfmt.
              entry = "${pkgs.lib.getExe' rust "rustfmt"}";
              package = rust;
              args = [ "--edition=2021" ];
            };

            # Nix.
            nixfmt.enable = true;
            deadnix.enable = true;
            statix.enable = true;
            nil.enable = true;

            # General.
            check-added-large-files.enable = true;
            check-merge-conflicts.enable = true;
            check-yaml.enable = true;
            detect-private-keys.enable = true;
            end-of-file-fixer.enable = true;
          };
        };
      in
      {
        checks.pre-commit-check = pre-commit-check;

        # Run all hooks over the repository with `nix fmt`.
        formatter =
          let
            inherit (pre-commit-check.config) package configFile;
          in
          pkgs.writeShellScriptBin "pre-commit-run" ''
            ${pkgs.lib.getExe package} run --all-files --config ${configFile}
          '';

        devShells.default = pkgs.mkShell {
          inherit (pre-commit-check) shellHook;
          buildInputs = [
            rust
            pkgs.flip-link
          ]
          ++ pre-commit-check.enabledPackages;
          propagatedBuildInputs = [
            pkgs.probe-rs-tools
            pkgs.elf2uf2-rs
          ];
        };
      }
    );
}
