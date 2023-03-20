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

"""Tests for hparam_configs."""

import os
import tempfile
from absl import logging
import tensorflow.compat.v1 as tf
import yaml
from deeplab2.utils import hparam_configs


class HparamConfigsTest(tf.test.TestCase):

  def test_config_override(self):
    c = hparam_configs.Config({'a': 1, 'b': 2})
    self.assertEqual(c.as_dict(), {'a': 1, 'b': 2})

    c.update({'a': 10})
    self.assertEqual(c.as_dict(), {'a': 10, 'b': 2})

    c.b = 20
    self.assertEqual(c.as_dict(), {'a': 10, 'b': 20})

    c.override('a=true,b=ss')
    self.assertEqual(c.as_dict(), {'a': True, 'b': 'ss'})

    c.override('a=100,,,b=2.3,')  # Extra ',' is fine.
    self.assertEqual(c.as_dict(), {'a': 100, 'b': 2.3})

    c.override('a=2x3,b=50')  # a is a special format for image size.
    self.assertEqual(c.as_dict(), {'a': '2x3', 'b': 50})

    # Overrriding string must be in the format of xx=yy.
    with self.assertRaises(ValueError):
      c.override('a=true,invalid_string')

  def test_config_yaml(self):
    tmpdir = tempfile.gettempdir()
    yaml_file_path = os.path.join(tmpdir, 'x.yaml')
    with open(yaml_file_path, 'w') as f:
      f.write("""
        x: 2
        y:
          z: 'test'
      """)
    c = hparam_configs.Config(dict(x=234, y=2342))
    c.override(yaml_file_path)
    self.assertEqual(c.as_dict(), {'x': 2, 'y': {'z': 'test'}})

    yaml_file_path2 = os.path.join(tmpdir, 'y.yaml')
    c.save_to_yaml(yaml_file_path2)
    with open(yaml_file_path2, 'r') as f:
      config_dict = yaml.load(f, Loader=yaml.FullLoader)
    self.assertEqual(config_dict, {'x': 2, 'y': {'z': 'test'}})

  def test_config_override_recursive(self):
    c = hparam_configs.Config({'x': 1})
    self.assertEqual(c.as_dict(), {'x': 1})
    c.override('y.y0=2,y.y1=3', allow_new_keys=True)
    self.assertEqual(c.as_dict(), {'x': 1, 'y': {'y0': 2, 'y1': 3}})
    c.update({'y': {'y0': 5, 'y1': {'y11': 100}}})
    self.assertEqual(c.as_dict(), {'x': 1, 'y': {'y0': 5, 'y1': {'y11': 100}}})
    self.assertEqual(c.y.y1.y11, 100)

  def test_config_override_list(self):
    c = hparam_configs.Config({'x': [1.0, 2.0]})
    self.assertEqual(c.as_dict(), {'x': [1.0, 2.0]})
    c.override('x=3.0|4.0|5.0')
    self.assertEqual(c.as_dict(), {'x': [3.0, 4.0, 5.0]})

  def test_registry_factory(self):
    registry = hparam_configs.RegistryFactor(prefix='test:')

    @registry.register()  # Use class name as key in default.
    class A:
      pass

    @registry.register(name='special_b')  # Use name as key if name is not None.
    class B:
      pass

    self.assertEqual(registry.lookup('A'), A)
    self.assertEqual(registry.lookup('special_b'), B)
    with self.assertRaises(KeyError):
      registry.lookup('B')


if __name__ == '__main__':
  logging.set_verbosity(logging.WARNING)
  tf.test.main()
