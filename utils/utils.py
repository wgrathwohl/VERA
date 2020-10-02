"""
Miscellaneous utilities.
"""

import os

def set_bn_to_eval(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.eval()

def set_bn_to_train(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()

def makedirs(dirname):
    """
    Make directory only if it's not already there.
    """
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def make_logdirs(args):
    """
    Make directories for logging.
    """
    makedirs(args.save_dir)
    data_sgld_dir = "{}/{}".format(args.save_dir, "data_sgld")
    makedirs(data_sgld_dir)
    gen_sgld_dir = "{}/{}".format(args.save_dir, "generator_sgld")
    makedirs(gen_sgld_dir)
    z_sgld_dir = "{}/{}".format(args.save_dir, "z_only_sgld")
    makedirs(z_sgld_dir)

    data_sgld_chain_dir = "{}/{}".format(args.save_dir, "data_sgld_chain")
    makedirs(data_sgld_chain_dir)
    gen_sgld_chain_dir = "{}/{}".format(args.save_dir, "generator_sgld_chain")
    makedirs(gen_sgld_chain_dir)
    z_sgld_chain_dir = "{}/{}".format(args.save_dir, "z_only_sgld_chain")
    makedirs(z_sgld_chain_dir)

    save_model_dir = "{}/{}".format(args.save_dir, "save_model")
    makedirs(save_model_dir)
    return data_sgld_dir, gen_sgld_dir, z_sgld_dir, \
           data_sgld_chain_dir, gen_sgld_chain_dir, z_sgld_chain_dir, \
           save_model_dir


def print_log(print_str, args):
    """
    Print to stdout and flush output to file.
    """
    print(print_str)
    with open(os.path.join(args.save_dir, args.log_file), "a") as log_file:
        log_file.write(str(print_str) + "\n")
