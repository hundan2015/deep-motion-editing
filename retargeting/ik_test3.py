import os
from datasets.bvh_parser import BVH_file
from datasets.bvh_writer import BVH_writer
from models.IK import fix_foot_contact
from os.path import join as pjoin
from models.skeleton import build_edge_topology

import sys
import torch
from models.Kinematics import InverseKinematics
from datasets.bvh_parser import BVH_file
from tqdm import tqdm

sys.path.append("../utils")

import BVH as BVH
import Animation as Animation
from Quaternions_old import Quaternions

import hmrutils.matrix as matrix
from hmrutils.ik.ccd_ik import CCD_IK
from pytorch3d.transforms import (
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    quaternion_to_matrix,
    matrix_to_quaternion,
)
from einops import einsum
from hmrutils.geo.quaternion import qbetween, qslerp, qinv, qmul, qrot
import numpy as np


def main():
    fake_pos = torch.from_numpy(np.load("./fake_pos.npy"))
    fake_res_denorm = torch.from_numpy(np.load("./fake_res_denorm.npy"))
    offset = torch.from_numpy(np.load("./offset.npy"))
    topology = torch.from_numpy(np.load("./topology.npy"))

    raw = fake_res_denorm
    position = raw[:, -3:, :]
    rotation = raw[:, :-3, :]

    rotation = rotation.reshape((rotation.shape[0], -1, 4, rotation.shape[-1]))
    # region supported fk
    identity = torch.tensor((1, 0, 0, 0), dtype=torch.float, device=raw.device)
    identity = identity.reshape((1, 1, -1, 1))
    new_shape = list(rotation.shape)
    new_shape[1] += 1
    new_shape[2] = 1
    rotation_final = identity.repeat(new_shape)
    count = rotation_final.shape[1]
    for i in range(1, count):
        rotation_final[:, i, :, :] = rotation[:, i - 1, :, :]

    rotation_final = rotation_final.permute(0, 3, 1, 2)
    position = position.permute(0, 2, 1)
    result = torch.empty(rotation_final.shape[:-1] + (3,), device=position.device)

    norm = torch.norm(rotation_final, dim=-1, keepdim=True)
    # norm[norm < 1e-10] = 1
    rotation_final = rotation_final / norm
    transform = quaternion_to_matrix(rotation_final)

    offset = offset.reshape((-1, 1, offset.shape[-2], offset.shape[-1], 1))

    result[..., 0, :] = position
    for i, pi in enumerate(topology):
        if pi == -1:
            assert i == 0
            continue

        transform[..., i, :, :] = torch.matmul(
            transform[..., pi, :, :].clone(), transform[..., i, :, :].clone()
        )
        result[..., i, :] = torch.matmul(
            transform[..., pi, :, :], offset[..., i, :, :]
        ).squeeze()
        result[..., i, :] += result[..., pi, :]
    # print(result - fake_pos)
    # endregion

    # region matrix library
    rotation_mat = quaternion_to_matrix(rotation_final)
    temp_offset = offset.repeat(1, 108, 1, 1, 1).squeeze(-1)
    local_mat = matrix.get_TRS(rotation_mat, temp_offset)
    fk_mat = matrix.forward_kinematics(local_mat, topology)
    matrix_positions = matrix.get_position(fk_mat)
    print(result - matrix_positions)
    # endregion
    # region my forward
    for index, par in enumerate(topology):
        if index == 0:
            continue
        local_mat[..., index, :, :] = torch.matmul(
            local_mat[..., par, :, :], local_mat[..., index, :, :]
        )
    my_positions = matrix.get_position(local_mat)
    # endregion
    print(matrix_positions - my_positions)


if __name__ == "__main__":
    main()
