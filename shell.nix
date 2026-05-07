{ pkgs ? import <nixpkgs> { config.allowUnfree = true; } }:

pkgs.mkShell {
  buildInputs = with pkgs; [
    ngspice
    python312
    python312Packages.numpy
    python312Packages.scipy
    python312Packages.matplotlib
    python312Packages.pandas
  ];

  shellHook = ''
    echo "AnalogGym optimization environment"
    echo "  ngspice: $(ngspice -v 2>&1 | head -1)"
    echo "  python:  $(python --version)"
    echo ""
    echo "Additional pip packages needed:"
    echo "  pip install botorch gpytorch anthropic gymnasium tabulate torch-geometric pytest"
  '';
}
