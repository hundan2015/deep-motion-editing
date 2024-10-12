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


def get_parent_chain(parents: list, index: int, stop: int) -> list:
    current = index
    res = []
    while current != stop:
        res.append(current)
        current = parents[current]
    res.append(stop)
    res.reverse()
    return res


def get_parent_chain_count(parents: list, index: int, count: int) -> list:
    current = index
    res = []
    counter = 0
    while counter != count:
        res.append(current)
        current = parents[current]
        counter += 1
    res.reverse()
    return res


def get_id_from_joint_names(name_list: list, names: list) -> list:
    res = []
    for name in name_list:
        res.append(names.index(name))
    return res


def get_obj(positions, name, index=0):
    import trimesh

    def make_cube(size, position):
        cube = trimesh.creation.box(extents=[size, size, size])
        target_position = position  # 指定位置
        cube.apply_translation(target_position)
        return cube

    cube0 = make_cube(1, positions[..., 0, :][index])
    for i in range(1, positions.shape[-2]):
        cube0 += make_cube(1, positions[..., i, :][index])
    cube0.export(name)


def get_batched_ik(
    rotation: torch.tensor,
    offset: torch.tensor,
    parents: list,
    chain: torch.tensor,
    target_position: torch.tensor,
    ik_position: int = 1,
    lr=0.001,
    epoch=1000,
    is_log=False,
):
    """
    Args:
        rotation: torch.tensor (...., n_joint, 4), In Quatanion.
        offset: torch.tensor (..., n_joint, 3)
        chain: list(n_joint)
        target_position: torch.tensor (..., 3)
    Returns:
        new_rotation: torch.tensor (...., n_joint, 4), In Quatanion.
    """
    rotation_mat = quaternion_to_matrix(rotation)  # (324,29,3,3),PASSED
    local_mat = matrix.get_TRS(rotation_mat, offset)  # (324,29,4,4),PASSED

    fk_mat = matrix.forward_kinematics(local_mat, parents)  # (324,29,4,4),PASSED

    local_mat_short = local_mat[..., chain, :, :].clone()
    local_mat_short[..., 0, :, :] = fk_mat[..., chain[0], :, :].clone()
    parent_short = [i - 1 for i in range(len(chain))]

    rotations_short = matrix.get_rotation(local_mat_short)
    quat = matrix_to_quaternion(rotations_short[..., ik_position, :, :])

    fk_mat_short = matrix.forward_kinematics(local_mat_short, parent_short)
    positions_short = matrix.get_position(fk_mat_short)

    # region traditional
    end_position = positions_short[..., -1, :]
    current_position = positions_short[..., ik_position, :]
    current_vector = torch.nn.functional.normalize(
        end_position - current_position, dim=-1
    )
    target_vector = torch.nn.functional.normalize(
        target_position - current_position, dim=-1
    )

    quat = qmul(qbetween(current_vector, target_vector), quat)
    # endregion

    # region Torch optimize
    loss_fn = torch.nn.MSELoss()  # 使用均方误差损失
    for step in range(epoch):
        var = torch.nn.Parameter(quat.clone().detach().requires_grad_())
        optimizer = torch.optim.Adamax([var], lr=lr)
        optimizer.zero_grad()  # 清除前一轮的梯度
        local_mat_temp = local_mat_short.clone()
        local_mat_temp[..., ik_position, :3, :3] = quaternion_to_matrix(var)
        positions_temp = matrix.get_position(
            matrix.forward_kinematics(local_mat_temp, parent_short)
        )
        end_pos_temp = positions_temp[..., -1, :]
        start_pos_temp = positions_temp[..., ik_position, :]

        loss = loss_fn(
            torch.nn.functional.normalize(end_pos_temp - start_pos_temp),
            torch.nn.functional.normalize(target_position - start_pos_temp),
        )
        loss.backward()
        optimizer.step()
        quat = var.clone().detach()
        if is_log and step % 10 == 0:
            print(f"Step {step}, Loss: {loss.item()}")
    # endregion

    chain_rotmat = matrix.get_rotation(local_mat_short)
    chain_rotmat[..., ik_position, :, :] = quaternion_to_matrix(quat)
    local_mat[..., chain[1:], :-1, :-1] = chain_rotmat[..., 1:, :, :]  # (B, L, J, 3, 3)
    rotation_mat = matrix.get_rotation(local_mat)
    new_rotation = matrix_to_quaternion(rotation_mat)
    return new_rotation


def example(
    input_file_dir: str,
    output_path: str,
    target_points: torch.tensor = None,
    is_test: bool = True,
):
    anim, name, ftime = BVH.load(input_file_dir)
    clip_len, n_joints, _ = anim.rotations.qs.shape

    count = 1

    # BVH anim.rotations.qs + offset to local mat. Need a fk like procedure.
    rotation_mat = quaternion_to_matrix(
        torch.from_numpy(anim.rotations.qs)
    )  # (324,29,3,3),PASSED
    offset = (
        torch.from_numpy(anim.offsets).unsqueeze(0).repeat(clip_len, 1, 1)
    )  # (324,29,3),PASSED

    # region prepare target points
    local_mat = matrix.get_TRS(rotation_mat, offset)  # (324,29,4,4),PASSED
    fk_mat = matrix.forward_kinematics(local_mat, anim.parents)  # (324,29,4,4),PASSED

    if target_points == None:
        # target_points = anim.positions[:, get_id_from_joint_names(["Head"], name)[0], :]
        target_points = matrix.get_position(fk_mat)[
            ..., get_id_from_joint_names(["Head"], name)[0], :
        ].to(torch.float32)
    # endregion

    chain = get_parent_chain(
        anim.parents,
        get_id_from_joint_names(["LeftHand"], name)[0],
        get_id_from_joint_names(["Spine2"], name)[0],
    )  # PASS

    rotation = get_batched_ik(
        torch.from_numpy(anim.rotations.qs),
        torch.from_numpy(anim.offsets).unsqueeze(0).repeat(clip_len, 1, 1),
        anim.parents,
        chain,
        target_points,
        3,
    )

    anim.rotations = Quaternions(rotation.numpy())
    # anim.positions = position.numpy()

    BVH.save(output_path, anim, name, ftime)


if __name__ == "__main__":
    example(
        "./datasets/Mixamo/Aj/Disappointed.bvh",
        "./examples/intra_structure/fuck32.bvh",
    )
    print("Finished!")
