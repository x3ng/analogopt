{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    ngspice
    swig
  ];

  shellHook = ''
    echo "ngspice: $(ngspice -v 2>&1 | head -1)"
    echo "use 'uv run python' for Python"
  '';
}
