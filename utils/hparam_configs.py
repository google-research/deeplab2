# coding=utf-8
# Copyright 2023 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A simple wrapper of hierarchical dictionary for hparams."""

import ast
import collections
import copy
from typing import Any, Dict, Text, List, Callable, Optional
import tensorflow as tf
import yaml

REQUIRED = '__required__'


def eval_str_fn(val: str) -> Any:
  if '|' in val:
    return [eval_str_fn(v) for v in val.split('|')]
  if val in {'true', 'false'}:
    return val == 'true'
  try:
    return ast.literal_eval(val)
  except (ValueError, SyntaxError):
    return val


# pylint: disable=protected-access
class Config(dict):
  """A config utility class."""

  def __init__(self, *args, **kwargs):
    super().__init__()
    input_config_dict = dict(*args, **kwargs)
    self.update(input_config_dict)

  def __len__(self):
    return len(self.__dict__)

  def __setattr__(self, k: str, v: Any) -> None:
    if isinstance(v, dict) and not isinstance(v, Config):
      self.__dict__[k] = Config(v)
    else:
      self.__dict__[k] = copy.deepcopy(v)

  def __getattr__(self, k: str) -> Any:
    return self.__dict__[k]

  def __setitem__(self, k: str, v: Any) -> None:
    self.__setattr__(k, v)

  def __getitem__(self, k: str) -> Any:
    return self.__dict__[k]

  def __contains__(self, k: str) -> bool:
    return self.__dict__.__contains__(k)

  def __iter__(self):
    for key in self.__dict__:
      yield key

  def items(self):
    for key, value in self.__dict__.items():
      yield key, value

  def replace(self, **kwargs) -> 'Config':
    """Deep copies and replaces some values."""
    cfg = copy.deepcopy(self)
    cfg.update(dict(**kwargs))
    return cfg

  def __repr__(self):
    return repr(self.as_dict())

  def __getstate__(self):
    return self.__dict__

  def __copy__(self):
    cls = self.__class__
    result = cls.__new__(cls)
    result.__dict__.update(self.__dict__)
    return result

  def __deepcopy__(self, memo: Any) -> 'Config':
    cls = self.__class__
    result = cls.__new__(cls)
    for k, v in self.__dict__.items():
      result[k] = v
    return result

  def __str__(self):
    try:
      return yaml.dump(self.as_dict(), indent=4)
    except TypeError:
      return str(self.as_dict())

  def _update(self, config_dict, allow_new_keys=True, skip_new_keys=False):
    """Recursively updates internal members."""
    if not config_dict:
      return

    for k, v in config_dict.items():
      if k not in self.__dict__:
        if allow_new_keys:
          self.__setattr__(k, v)
        elif skip_new_keys:
          pass
        else:
          raise KeyError('Key `{}` does not exist for overriding. '.format(k))
      else:
        if isinstance(self.__dict__[k], Config) and isinstance(v, dict):
          self.__dict__[k]._update(v, allow_new_keys)
        elif isinstance(self.__dict__[k], Config) and isinstance(v, Config):
          self.__dict__[k]._update(v.as_dict(), allow_new_keys)
        else:
          self.__setattr__(k, v)

  def get(self, k: str, default_value: Any = None) -> Any:
    return self.__dict__.get(k, default_value)

  def update(self, config_dict: Optional[Dict[Any, Any]] = None) -> None:
    """Updates members while allowing new keys."""
    if config_dict:
      self._update(config_dict, allow_new_keys=True)

  def update_dict(self, **kwargs):
    """Updates members while allowing new keys."""
    self._update(kwargs, allow_new_keys=True)

  def keys(self) -> Any:
    return self.__dict__.keys()

  def override_dict(self, skip_new_keys: bool = True, **kwargs) -> None:
    """Overrides members and skips new keys."""
    self._update(kwargs, allow_new_keys=False, skip_new_keys=skip_new_keys)

  def override(self, config_dict_or_str: ...,
               allow_new_keys: bool = False) -> None:
    """Updates members while disallowing new keys."""
    if not config_dict_or_str:
      return
    if isinstance(config_dict_or_str, str):
      if '=' in config_dict_or_str:
        config_dict = self.parse_from_str(config_dict_or_str)
      elif config_dict_or_str.endswith('.yaml'):
        config_dict = self.parse_from_yaml(config_dict_or_str)
      else:
        raise ValueError(
            'Invalid string {}, must end with .yaml or contains "=".'.format(
                config_dict_or_str))
    elif isinstance(config_dict_or_str, dict):
      config_dict = config_dict_or_str
    else:
      raise ValueError('Unknown value type: {}'.format(config_dict_or_str))

    self._update(config_dict, allow_new_keys)

  def parse_from_yaml(self, yaml_file_path: Text) -> Dict[Any, Any]:
    """Parses a yaml file and returns a dictionary."""
    with tf.io.gfile.GFile(yaml_file_path, 'r') as f:
      config_dict = yaml.load(f, Loader=yaml.FullLoader)
      return config_dict

  def save_to_yaml(self, yaml_file_path: str) -> None:
    """Writes a dictionary into a yaml file."""
    with tf.io.gfile.GFile(yaml_file_path, 'w') as f:
      yaml.dump(self.as_dict(), f, default_flow_style=False)

  def parse_from_str(self, config_str: Text) -> Dict[Any, Any]:
    """Parses a string like 'x.y=1,x.z=2' to nested dict {x: {y: 1, z: 2}}."""
    if not config_str:
      return {}
    config_dict = {}
    try:
      for kv_pair in config_str.split(','):
        # We skip the empty string here.
        if not kv_pair:
          continue
        key_str, value_str = kv_pair.split('=')
        key_str = key_str.strip()

        def add_kv_recursive(k, v):
          """Recursively parses x.y.z=tt to {x: {y: {z: tt}}}."""
          if '.' not in k:
            return {k: eval_str_fn(v)}
          pos = k.index('.')
          return {k[:pos]: add_kv_recursive(k[pos + 1:], v)}  # pylint: disable=cell-var-from-loop

        def merge_dict_recursive(target, src):
          """Recursively merges two nested dictionary."""
          for k in src.keys():
            if ((k in target and isinstance(target[k], dict) and
                 isinstance(src[k], collections.abc.Mapping))):
              merge_dict_recursive(target[k], src[k])  # pylint: disable=cell-var-from-loop
            else:
              target[k] = src[k]

        merge_dict_recursive(config_dict, add_kv_recursive(key_str, value_str))
      return config_dict
    except ValueError:
      raise ValueError(f'Invalid config_str: {config_str}') from None

  def as_dict(self) -> Dict[Any, Any]:
    """Returns a dict representation."""
    config_dict = {}
    for k, v in self.__dict__.items():
      if isinstance(v, Config):
        config_dict[k] = v.as_dict()
      elif isinstance(v, (list, tuple)):
        config_dict[k] = [
            i.as_dict() if isinstance(i, Config) else copy.deepcopy(i)
            for i in v
        ]
      else:
        config_dict[k] = copy.deepcopy(v)
    return config_dict

  def validate(self):
    required_keys = []
    for k, v in self.__dict__.items():
      if v == REQUIRED:
        required_keys.append(k)
    if required_keys:
      raise ValueError(f'Values are required for keys: {required_keys}')


class RegistryFactor():
  """A template for registry factory."""

  def __init__(self, prefix: str) -> None:
    self.registry_map = {}
    self.prefix = prefix

  def register(self, name: Optional[str] = None) -> Callable[[Any], Any]:
    """Registers a function, mainly for config here."""

    def decorator(cls):
      key = self.prefix + (name or cls.__name__.lower())
      if key in self.registry_map:
        raise ValueError(f'{key} is already registered')
      self.registry_map[key] = cls
      return cls

    return decorator

  def lookup(self, name: str) -> Config:
    """Looks up a class based on class name."""
    key = self.prefix + name.lower()
    if key not in self.registry_map:
      raise KeyError(f'{key} is not in {self.registry_map.keys()}')
    return self.registry_map[key]

  def keys(self, prefix: str = '') -> List[Config]:
    return [
        k[len(prefix):]
        for k in self.registry_map.keys()
        if k.startswith(prefix)
    ]


def create_config_from_dict(config_dict: Dict[str, Any],
                            required_keys: List[str],
                            optional_keys: Dict[str, Any]) -> Config:
  """Creates hparam config from dictionary."""
  config = Config()
  for key in required_keys:
    if key not in config_dict:
      raise ValueError('Required key %s missed in config dict.' % key)
    config[key] = config_dict[key]
  for key in optional_keys:
    if key not in config_dict:
      config[key] = optional_keys[key]
    else:
      config[key] = config_dict[key]
  return config
