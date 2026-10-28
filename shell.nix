let
  pkgs = import <nixpkgs> {
    config = {
      allowUnfree = true;
    };
  };
  python = pkgs.python3.override {
    self = python;
    packageOverrides = self: super: {
      # CUDA only for jax; everything else (torch, ...) stays CPU-only
      jax = (super.jax.override { cudaSupport = true; }).overridePythonAttrs (old: {
        doCheck = false;
      });
    };
  };
in
pkgs.mkShell {
  packages = [
    (python.withPackages (ps: [
      ps.jax
      ps.equinox
      ps.numpy
      ps.torch
      ps.torchvision
      ps.optax
      ps.matplotlib
      ps.typer
    ]))
    pkgs.basedpyright
    pkgs.black
  ];
}
