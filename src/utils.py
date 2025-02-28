import logging
import os
import sys

import datasets
import transformers
import yaml


def load_config(file_name: str, config_path: str = "../training_config_files") -> dict:
    with open(os.path.join(config_path, file_name), "r") as f:
        config = yaml.safe_load(f)
    return config


def config_logs(logger: logging.Logger, train_conf: dict, peft_conf: dict):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = train_conf.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {train_conf.local_rank}, device: {train_conf.device}, n_gpu: {train_conf.n_gpu}"
        + f" distributed training: {bool(train_conf.local_rank != -1)}, 16-bits training: {train_conf.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_conf}")
    logger.info(f"PEFT parameters {peft_conf}")
