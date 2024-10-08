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


def get_parent_chain(parents: list, index: int, stop: int) -> list:
    current = index
    res = []
    while current != stop:
        res.append(current)
        current = parents[current]
    res.append(stop)
    res.reverse()
    return res


def get_id_from_joint_names(name_list: list, names: list) -> list:
    res = []
    for name in name_list:
        res.append(names.index(name))
    return res


def transform_from_quaternion(quater: torch.Tensor):
    qw = quater[..., 0]
    qx = quater[..., 1]
    qy = quater[..., 2]
    qz = quater[..., 3]

    x2 = qx + qx
    y2 = qy + qy
    z2 = qz + qz
    xx = qx * x2
    yy = qy * y2
    wx = qw * x2
    xy = qx * y2
    yz = qy * z2
    wy = qw * y2
    xz = qx * z2
    zz = qz * z2
    wz = qw * z2

    m = torch.empty(quater.shape[:-1] + (3, 3), device=quater.device)
    m[..., 0, 0] = 1.0 - (yy + zz)
    m[..., 0, 1] = xy - wz
    m[..., 0, 2] = xz + wy
    m[..., 1, 0] = xy + wz
    m[..., 1, 1] = 1.0 - (xx + zz)
    m[..., 1, 2] = yz - wx
    m[..., 2, 0] = xz - wy
    m[..., 2, 1] = yz + wx
    m[..., 2, 2] = 1.0 - (xx + yy)

    return m


def forward(
    topology,
    rotation: torch.Tensor,
    position: torch.Tensor,
    offset: torch.Tensor,
    order="xyz",
    quater=True,
    world=True,
):
    """
    rotation should have shape batch_size * Joint_num * (3/4) * Time
    position should have shape batch_size * 3 * Time
    offset should have shape batch_size * Joint_num * 3
    output have shape batch_size * Time * Joint_num * 3
    """
    if not quater and rotation.shape[-2] != 3:
        raise Exception("Unexpected shape of rotation")
    if quater and rotation.shape[-2] != 4:
        raise Exception("Unexpected shape of rotation")
    rotation = rotation.permute(0, 3, 1, 2)
    position = position.permute(0, 2, 1)
    result = torch.empty(rotation.shape[:-1] + (3,), device=position.device)

    norm = torch.norm(rotation, dim=-1, keepdim=True)
    # norm[norm < 1e-10] = 1
    rotation = rotation / norm

    if quater:
        transform = transform_from_quaternion(rotation)

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
            transform[..., i, :, :], offset[..., i, :, :]
        ).squeeze()
        if world:
            result[..., i, :] += result[..., pi, :]
    return result


def example(
    input_file_dir: str,
    output_path: str,
    target_points: torch.tensor = None,
    is_test: bool = True,
):
    anim, name, ftime = BVH.load(input_file_dir)
    clip_len, n_joints, _ = anim.rotations.qs.shape

    # region forward test
    temp_result = forward(
        anim.parents,
        torch.from_numpy(anim.rotations.qs)
        .permute(1, 2, 0)
        .unsqueeze(0)
        .to(torch.float32),
        torch.from_numpy(anim.positions)
        .permute(1, 2, 0)[0]
        .unsqueeze(0)
        .to(torch.float32),
        torch.from_numpy(anim.offsets).unsqueeze(0).to(torch.float32),
    )
    # endregion

    count = 1

    # BVH anim.rotations.qs + offset to local mat. Need a fk like procedure.
    rotation_mat = quaternion_to_matrix(
        torch.from_numpy(anim.rotations.qs)
    )  # (324,29,3,3)
    offset = (
        torch.from_numpy(anim.offsets).unsqueeze(0).repeat(clip_len, 1, 1)
    )  # (324,29,3)
    # rotation_mat = rotation_mat.transpose(-1,-2)
    local_mat = matrix.get_TRS(rotation_mat, offset)  # (324,29,4,4)

    fk_mat = matrix.forward_kinematics(local_mat, anim.parents)  # (324,29,4,4)
    global_rot = matrix.get_rotation(fk_mat).clone()  # (324,29,3,3)

    if target_points == None:
        # target_points = anim.positions[:, get_id_from_joint_names(["Head"], name)[0], :]
        target_points = matrix.get_position(fk_mat)[
            ..., get_id_from_joint_names(["Head"], name)[0], :
        ].to(torch.float32)
        # target_points = torch.from_numpy(target_points).to(torch.float32)

    # Get chain from end point and parent.
    chain = get_parent_chain(
        anim.parents,
        get_id_from_joint_names(["LeftHand"], name)[0],
        get_id_from_joint_names(["LeftShoulder"], name)[0],
    )  # PASS
    target_ind = (list(range(len(chain))))[-count:]  # PASS

    IK_solver = CCD_IK(
        local_mat=local_mat,
        parent=anim.parents,
        target_ind=target_ind,
        target_pos=target_points,  # local postion
        target_rot=global_rot[..., chain[-count:], :, :],
        kinematic_chain=chain,
        max_iter=2,
    )

    chain_local_mat = IK_solver.solve()
    chain_rotmat = matrix.get_rotation(chain_local_mat)
    local_mat[..., chain[1:], :-1, :-1] = chain_rotmat[..., 1:, :, :]  # (B, L, J, 3, 3)

    rotation_mat = matrix.get_rotation(local_mat)
    rotation = matrix_to_quaternion(rotation_mat)

    new_fk_mat = matrix.forward_kinematics(local_mat, anim.parents)
    position = matrix.get_position(new_fk_mat)

    anim.rotations = Quaternions(rotation.numpy())
    anim.positions = position.numpy()

    BVH.save(output_path, anim, name, ftime)


if __name__ == "__main__":
    example(
        "./datasets/Mixamo/Aj/Disappointed.bvh",
        "./examples/intra_structure/fuck.bvh",
    )
    print("Finished!")
