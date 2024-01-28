import os

from . import cmd_args
from .paths_internal import models_path, script_path, data_path, sd_configs_path, sd_default_config, sd_model_file, default_sd_model_file, extensions_dir, extensions_builtin_dir  # noqa: F401

parser = cmd_args.parser

if os.environ.get('IGNORE_CMD_ARGS_ERRORS', None) is None:
    cmd_opts, _ = parser.parse_known_args()
else:
    cmd_opts, _ = parser.parse_known_args()

cmd_opts.webui_is_non_local = any([cmd_opts.share, cmd_opts.listen, cmd_opts.ngrok, cmd_opts.server_name])
cmd_opts.disable_extension_access = cmd_opts.webui_is_non_local and not cmd_opts.enable_insecure_extension_access
