{ pkgs, ... }:

{
  packages = with pkgs; [
    # ty
    git
  ];

  languages.python = {
    enable = true;
    version = "3.12";
    uv = {
      enable = true;
      sync.enable = true;
    };
    venv.enable = true;
  };

  processes.jupyter.exec = "jupyter notebook"; 
}
