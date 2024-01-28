import sys

from .shared_cmd_options import cmd_opts


def realesrgan_models_names():
    from . import realesrgan_model
    return [x.name for x in realesrgan_model.get_realesrgan_models(None)]


def postprocessing_scripts():
    from . import scripts

    return scripts.scripts_postproc.scripts


def sd_vae_items():
    from . import sd_vae

    return ["Automatic", "None"] + list(sd_vae.vae_dict)


def refresh_vae_list():
    from . import sd_vae

    sd_vae.refresh_vae_list()


def cross_attention_optimizations():
    from . import sd_hijack
    # import modules.sd_hijack

    return ["Automatic"] + [x.title() for x in sd_hijack.optimizers] + ["None"]


def sd_unet_items():
    # import modules.sd_unet
    from . import sd_unet

    return ["Automatic"] + [x.label for x in sd_unet.unet_options] + ["None"]


def refresh_unet_list():
    # import modules.sd_unet
    from . import sd_unet

    sd_unet.list_unets()


def list_checkpoint_tiles(use_short=False):
    # import modules.sd_models
    from . import sd_models
    return sd_models.checkpoint_tiles(use_short)


def refresh_checkpoints():
    # import modules.sd_models
    from . import sd_models
    return sd_models.list_models()


def list_samplers():
    from . import sd_samplers
    return sd_samplers.all_samplers


def reload_hypernetworks():
    from .hypernetworks import hypernetwork
    from . import shared

    shared.hypernetworks = hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir)


def get_infotext_names():
    from . import generation_parameters_copypaste, shared
    res = {}

    for info in shared.opts.data_labels.values():
        if info.infotext:
            res[info.infotext] = 1

    for tab_data in generation_parameters_copypaste.paste_fields.values():
        for _, name in tab_data.get("fields") or []:
            if isinstance(name, str):
                res[name] = 1

    return list(res)


ui_reorder_categories_builtin_items = [
    "prompt",
    "image",
    "inpaint",
    "sampler",
    "accordions",
    "checkboxes",
    "dimensions",
    "cfg",
    "denoising",
    "seed",
    "batch",
    "override_settings",
]


def ui_reorder_categories():
    from . import scripts

    yield from ui_reorder_categories_builtin_items

    sections = {}
    for script in scripts.scripts_txt2img.scripts + scripts.scripts_img2img.scripts:
        if isinstance(script.section, str) and script.section not in ui_reorder_categories_builtin_items:
            sections[script.section] = 1

    yield from sections

    yield "scripts"


class Shared(sys.modules[__name__].__class__):
    """
    this class is here to provide sd_model field as a property, so that it can be created and loaded on demand rather than
    at program startup.
    """

    sd_model_val = None

    @property
    def sd_model(self):
        from . import sd_models

        return sd_models.model_data.get_sd_model()

    @sd_model.setter
    def sd_model(self, value):
        from . import sd_models

        sd_models.model_data.set_sd_model(value)