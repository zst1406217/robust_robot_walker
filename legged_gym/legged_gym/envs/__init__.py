# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from .base.robot_field_noisy import RobotFieldNoisy
from legged_gym.utils.task_registry import task_registry

from .a1.a1_remote_goal_config import A1RemoteGoalCfg, A1RemoteGoalCfgPPO
task_registry.register( "a1_remotegoal", RobotFieldNoisy, A1RemoteGoalCfg(), A1RemoteGoalCfgPPO() )
from .a1.a1_mix_goal_stage1_config import A1MixStage1GoalCfg, A1MixStage1GoalCfgPPO
task_registry.register( "a1_mixgoalstage1", RobotFieldNoisy, A1MixStage1GoalCfg(), A1MixStage1GoalCfgPPO() )
from .a1.a1_mix_goal_stage2_config import A1MixStage2GoalCfg, A1MixStage2GoalCfgPPO
task_registry.register( "a1_mixgoalstage2", RobotFieldNoisy, A1MixStage2GoalCfg(), A1MixStage2GoalCfgPPO() )

from .a1.a1_bar_track_config import A1BarTrackCfg, A1BarTrackCfgPPO
task_registry.register( "a1_bartrack", RobotFieldNoisy, A1BarTrackCfg(), A1BarTrackCfgPPO() )