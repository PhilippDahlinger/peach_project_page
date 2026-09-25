
from omegaconf import OmegaConf

from pc_mango.util.job_type_resolver import shortener



def conditional_resolver(condition, if_true: str, if_false: str):
    if condition:
        return if_true
    else:
        return if_false


def load_omega_conf_resolvers():
    OmegaConf.register_new_resolver("sub_dir_shortener", shortener)
    OmegaConf.register_new_resolver("format", lambda inpt, formatter: formatter.format(inpt))
    OmegaConf.register_new_resolver("conditional_resolver", conditional_resolver)
    OmegaConf.register_new_resolver("len", lambda x: len(x))
