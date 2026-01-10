{ pkgs, ... }:

{
  packages = with pkgs; [
    git
    pyright
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

  # env.LD_LIBRARY_PATH = lib.makeLibraryPath [
  #   pkgs.stdenv.cc.cc.lib
  #   pkgs.zlib
  #   pkgs.glib
  # ];

  enterShell = ''
    cat <<EOF > pyrightconfig.json
    {
      "venvPath": "$DEVENV_ROOT/.devenv/state",
      "venv": "venv",
      "include": ["src", "notebooks"],
      "exclude": ["**/node_modules", "**/__pycache__"]
    }
    EOF
  '';
}
