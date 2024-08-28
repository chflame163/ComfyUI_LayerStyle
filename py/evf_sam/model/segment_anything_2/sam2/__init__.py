# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from hydra import initialize_config_module

# initialize_config_module("model/segment_anything_2/sam2_configs", version_base="1.2")
initialize_config_module("model/segment_anything_2/sam2_configs")
