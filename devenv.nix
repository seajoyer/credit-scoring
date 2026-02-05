{ pkgs, ... }:

{
  packages = with pkgs; [
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
}
