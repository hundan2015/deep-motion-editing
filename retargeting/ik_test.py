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
    local_mat = matrix.get_TRS(rotation_mat, offset)  # (324,29,4,4),PASSED

    fk_mat = matrix.forward_kinematics(local_mat, anim.parents)  # (324,29,4,4),PASSED
    global_rot = matrix.get_rotation(fk_mat).clone()  # (324,29,3,3),CONFERMED
    positions = matrix.get_position(fk_mat).clone()
    # region Trimesh visiualize
    # positions = matrix.get_position(fk_mat)
    # hip_points = matrix.get_position(fk_mat)[
    #     ..., get_id_from_joint_names(["Hips"], name)[0], :
    # ].to(torch.float32)

    # get_obj(positions, "fuckme1.obj")
    # endregion

    if target_points == None:
        # target_points = anim.positions[:, get_id_from_joint_names(["Head"], name)[0], :]
        target_points = matrix.get_position(fk_mat)[
            ..., get_id_from_joint_names(["Head"], name)[0], :
        ].to(torch.float32)
        # target_points = target_points.unsqueeze(-1)
        # target_points = torch.zeros_like(target_points)

    # region my ik test
    chain = get_parent_chain(
        anim.parents,
        get_id_from_joint_names(["LeftHand"], name)[0],
        get_id_from_joint_names(["Spine2"], name)[0],
    )  # PASS
    local_mat_short = local_mat[..., chain, :, :].clone()
    local_mat_short[..., 0, :, :] = fk_mat[..., chain[0], :, :].clone()
    parent_short = [i - 1 for i in range(len(chain))]

    fk_mat_short = matrix.forward_kinematics(local_mat_short, parent_short)
    positions_short = matrix.get_position(fk_mat_short)
    rotations_short = matrix.get_rotation(local_mat_short)
    root_position = positions_short[:, 0, :]
    original_position = positions_short[:, 1, :]
    target_position = target_points.clone()
    quat = matrix_to_quaternion(rotations_short[:, 1])

    # region Torch optimize
    # epoch = 100
    # target = target_position
    # loss_fn = torch.nn.MSELoss()  # 使用均方误差损失
    # for step in range(epoch):
    #     var = torch.nn.Parameter(quat.clone().detach().requires_grad_())
    #     optimizer = torch.optim.Adamax([var], lr=0.01)
    #     optimizer.zero_grad()  # 清除前一轮的梯度
    #     local_mat_temp = local_mat_short.clone()
    #     local_mat_temp[:, 1, :3, :3] = quaternion_to_matrix(var)
    #     end_pos_temp = matrix.get_position(
    #         matrix.forward_kinematics(local_mat_temp, parent_short)
    #     )[:, -1]
    #     loss = loss_fn(end_pos_temp, target)
    #     loss.backward()
    #     optimizer.step()
    #     quat = var.clone().detach()
    #     if step % 10 == 0:
    #         print(f"Step {step}, Loss: {loss.item()}")

    # chain_rotmat = matrix.get_rotation(local_mat_short)
    # chain_rotmat[:, 1] = quaternion_to_matrix(quat)
    # local_mat[..., chain[1:], :-1, :-1] = chain_rotmat[..., 1:, :, :]  # (B, L, J, 3, 3)
    # endregion
    # region traditional
    solved_pos_target_quat = qmul(
        qbetween(
            original_position - root_position, target_position - original_position
        ),
        quat,
    )

    x_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
    x_vec[..., 0] = 1.0
    x_vec_sum = torch.zeros_like(x_vec)
    y_vec = torch.zeros((quat.shape[:-1] + (3,)), device=quat.device)
    y_vec[..., 1] = 1.0
    y_vec_sum = torch.zeros_like(y_vec)
    x_vec_sum += qrot(solved_pos_target_quat, x_vec)
    y_vec_sum += qrot(solved_pos_target_quat, y_vec)

    x_vec_avg = matrix.normalize(x_vec_sum / count)
    y_vec_avg = matrix.normalize(y_vec_sum / count)
    z_vec_avg = torch.cross(x_vec_avg, y_vec_avg, dim=-1)
    solved_rot = torch.stack([x_vec_avg, y_vec_avg, z_vec_avg], dim=-1)  # column
    parent_rot = matrix.get_rotation(fk_mat_short)[..., 0, :, :]
    solved_local_rot = matrix.get_mat_BtoA(parent_rot, solved_rot)

    local_mat_short[..., 1, :-1, :-1] = solved_local_rot
    new_fk_mat_short = matrix.forward_kinematics(local_mat_short, parent_short)
    new_position_short = matrix.get_position(new_fk_mat_short)

    print(
        torch.norm(
            torch.nn.functional.normalize(
                new_position_short[:, -1, :] - new_position_short[:, 1, :]
            )
            - torch.nn.functional.normalize(target_points - new_position_short[:, 1, :])
        )
    )
    # endregion
    # endregion

    # region Original CCD IK part
    # # Get chain from end point and parent.
    # chain = get_parent_chain(
    #     anim.parents,
    #     get_id_from_joint_names(["LeftHand"], name)[0],
    #     get_id_from_joint_names(["Spine2"], name)[0],
    # )  # PASS
    # target_ind = (list(range(len(chain))))[-count:]  # PASS

    # IK_solver = CCD_IK(
    #     local_mat=local_mat,
    #     parent=anim.parents,
    #     target_ind=target_ind,
    #     target_pos=target_points,  # local postion
    #     target_rot=global_rot[..., chain[-count:], :, :],
    #     # target_rot=None,
    #     kinematic_chain=chain,
    #     max_iter=10,
    # )

    # chain_local_mat = IK_solver.solve()
    # chain_rotmat = matrix.get_rotation(chain_local_mat)
    # local_mat[..., chain[1:], :-1, :-1] = chain_rotmat[..., 1:, :, :]  # (B, L, J, 3, 3)
    # endregion

    # region Eye test
    # local_mat[..., get_id_from_joint_names(["LeftForeArm"], name)[0], :3, :3] = (
    #     torch.eye(3, device=local_mat.device)
    # )
    # endregion

    rotation_mat = matrix.get_rotation(local_mat)
    rotation = matrix_to_quaternion(rotation_mat)

    new_fk_mat = matrix.forward_kinematics(local_mat, anim.parents)
    position = matrix.get_position(new_fk_mat)
    get_obj(position, "fuckme2.obj", 77)

    anim.rotations = Quaternions(rotation.numpy())
    # anim.positions = position.numpy()

    BVH.save(output_path, anim, name, ftime)


if __name__ == "__main__":
    example(
        "./datasets/Mixamo/Aj/Disappointed.bvh",
        "./examples/intra_structure/fuck3.bvh",
    )
    print("Finished!")
