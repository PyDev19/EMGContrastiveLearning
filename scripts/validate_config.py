import argparse
import inspect

from src.config import DATASET_TO_PATINETS, TrainConfig, load_train_config
from src.utils.registers import DATASETS, LOSSES, MODELS, NORMALIZERS, SAMPLERS


def _model_arg_names(model_cls: type, exclude: set[str]) -> tuple[set[str], set[str]]:
    """Return (all accepted arg names, required arg names) for a model class.

    Walks the MRO because the DANN model takes **kwargs and forwards them to its parent.
    """
    accepted, required = set(), set()
    for cls in model_cls.__mro__:
        if cls is object or "__init__" not in vars(cls):
            continue
        for name, p in inspect.signature(cls.__init__).parameters.items():
            if name == "self" or p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            accepted.add(name)
            if p.default is inspect.Parameter.empty:
                required.add(name)
    return accepted - exclude, required - exclude


def check_train_config(config_path: str) -> TrainConfig:
    """Validate a training YAML.

    Checks, in order:
      1. the YAML matches the TrainConfig schema (missing fields, wrong types),
      2. the task / model / loss pairing,
      3. dataset, normalizer, model and loss names exist in their registries,
      4. model.args matches the model constructor (unknown and missing args),
      5. basic value sanity (positive epochs, divisibility of window by patch size, ...).

    Args:
        config_path (str): path to a YAML file.

    Returns:
        TrainConfig: the validated config.

    Raises:
        ValueError: with every problem found in checks 3-5 listed together.
        omegaconf.errors.OmegaConfBaseException: for schema failures (check 1).
    """
    config = load_train_config(config_path)  # checks 1 and 2

    errors: list[str] = []

    # 3. registry names
    for label, name, registry in [
        ("dataset", config.dataset.name, DATASETS),
        ("normalizer", config.normalizer, NORMALIZERS),
        ("model", config.model.name, MODELS),
        ("loss", config.loss.name, LOSSES),
    ]:
        if name not in registry:
            errors.append(
                f"unknown {label} '{name}'; available: {list(registry.keys())}"
            )
    if config.dataset.name not in DATASET_TO_PATINETS:
        errors.append(f"no patient list for dataset '{config.dataset.name}'")
    if config.dataset.name not in SAMPLERS and config.batch_size == "sampler":
        errors.append(
            f"batch_size='sampler' but no sampler for '{config.dataset.name}'"
        )

    # 4. model args vs constructor (num_subjects is injected by the builder for DANN)
    if config.model.name in MODELS:
        accepted, required = _model_arg_names(
            MODELS[config.model.name], exclude={"num_subjects"}
        )
        given = set(config.model.args)
        if unknown := sorted(given - accepted):
            errors.append(f"model.args has unknown keys: {unknown}")
        if missing := sorted(required - given):
            errors.append(f"model.args is missing required keys: {missing}")

    # 5. value sanity
    if config.num_epochs < 1:
        errors.append("num_epochs must be >= 1")
    if config.accumulation_steps < 1:
        errors.append("accumulation_steps must be >= 1")
    if config.learning_rate <= 0:
        errors.append("learning_rate must be > 0")
    if isinstance(config.batch_size, int) and config.batch_size < 1:
        errors.append("batch_size must be >= 1 or 'sampler'")

    window = config.dataset.args.window_opts
    args = config.model.args
    if window["size"] != args.get("time_steps", window["size"]):
        errors.append(
            f"window size {window['size']} != model time_steps {args['time_steps']}"
        )
    if "patch_size" in args and window["size"] % args["patch_size"] != 0:
        errors.append(
            f"window size {window['size']} is not divisible by patch_size {args['patch_size']}"
        )
    if (
        "embed_dim" in args
        and "num_heads" in args
        and args["embed_dim"] % args["num_heads"]
    ):
        errors.append("embed_dim must be divisible by num_heads")
    for key, value in args.items():
        if key.endswith("_prob") and not 0.0 <= value < 1.0:
            errors.append(f"model.args.{key}={value} must be in [0, 1)")

    if errors:
        raise ValueError(
            f"Invalid config '{config_path}':\n" + "\n".join(f"  - {e}" for e in errors)
        )

    print(f"Config OK: {config_path}")
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = check_train_config(args.config)
    print(f"\n{config}")


if __name__ == "__main__":
    main()
