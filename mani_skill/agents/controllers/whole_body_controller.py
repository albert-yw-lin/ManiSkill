"""
Whole Body Controller for mobile manipulation robots like Fetch.
Implements a quadratic programming-based controller that coordinates base motion and arm motion
to achieve desired end-effector poses while respecting joint limits and avoiding collisions.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any
from gymnasium import spaces

from qpsolvers import solve_qp
QP_AVAILABLE = True

from mani_skill.agents.controllers.base_controller import BaseController, ControllerConfig
from mani_skill.agents.controllers.utils.kinematics import Kinematics
from mani_skill.utils.geometry.rotation_conversions import (
    quaternion_to_matrix,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_apply,
    quaternion_to_axis_angle,
)
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, DriveMode
from mani_skill.utils import common, sapien_utils


class WholeBodyController(BaseController):
    """
    Whole Body Controller that coordinates base motion and arm motion
    for mobile manipulation robots like Fetch.
    
    Controls 11 DOF total:
    - Mobile base (4 DOF): x, y, theta (yaw), torso_lift
    - Arm (7 DOF): shoulder_pan, shoulder_lift, upperarm_roll, elbow_flex, 
                   forearm_roll, wrist_flex, wrist_roll
    
    Excludes head joints (head_pan, head_tilt) and gripper joints.
    """
    
    config: "WholeBodyControllerConfig"
    sets_target_qpos = True
    sets_target_qvel = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize controller parameters
        self._initialize_controller_params()
        
        # Initialize kinematics
        self._initialize_kinematics()
        
        # Initialize joints
        self._initialize_joints()
        
        # Initialize target pose
        self._target_pose = None
        
    def _initialize_controller_params(self):
        """Initialize controller parameters"""
        # Robot configuration
        self.num_base_dof = 4  # x, y, theta, torso_lift for mobile base
        self.num_arm_dof = len(self.config.arm_joint_names)
        self.total_dof = self.num_base_dof + self.num_arm_dof  # Should be 11 total
        
        # Weights for objective function
        self.W_ee = self.config.ee_tracking_weight
        self.W_posture = self.config.posture_regularization_weight  
        self.W_damping = self.config.base_damping_weight
        
        # Solver parameters
        self.max_iterations = self.config.max_iterations
        self.convergence_threshold = self.config.convergence_threshold
        self.timestep = 1.0 / self._control_freq
        
        # Neutral posture for regularization (extract only WBC DOF)
        # NOTE: While we extract all 11 WBC DOF, only torso_lift and arm joints are regularized.
        # Mobile base position (x, y, theta) is excluded from posture regularization.
        if self.config.neutral_posture is not None:
            # Extract only the 11 DOF we control: base(4) + arm(7)
            # Actual Fetch active joint order: [base_x, base_y, base_theta, torso_lift, head_pan, shoulder_pan, head_tilt, arm_joints...]
            # We control: [base_x, base_y, base_theta, torso_lift, shoulder_pan, shoulder_lift, arm_joints...] (skipping head joints)
            full_posture = self.config.neutral_posture
            if len(full_posture) == 15:  # Full Fetch posture
                # Map from full 15-DOF posture to our 11 WBC DOF
                # Based on actual active joint indices: [0,1,2,3,5,7,8,9,10,11,12]
                wbc_indices = [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 12]  # Skip head joints at indices 4 and 6
                self.q_retract = full_posture[wbc_indices] if hasattr(full_posture, '__getitem__') else full_posture
            else:
                self.q_retract = self.config.neutral_posture
        else:
            self.q_retract = None
        
        # Collision parameters (optional)
        self.collision_margin = self.config.collision_margin
        self.collision_detection_range = self.config.collision_detection_range
        
    def _initialize_kinematics(self):
        """Initialize kinematics solver"""
        # Get arm joint indices
        arm_joint_names = self.config.arm_joint_names
        self.arm_joints = []
        self.arm_joint_indices = []
        
        for i, joint in enumerate(self.articulation.get_active_joints()):
            if joint.name in arm_joint_names:
                self.arm_joints.append(joint)
                self.arm_joint_indices.append(i)
                
        self.arm_joint_indices = torch.tensor(self.arm_joint_indices, device=self.device)
        
        # Initialize kinematics solver for arm
        self.kinematics = Kinematics(
            self.config.urdf_path,
            self.config.ee_link_name,
            self.articulation,
            self.arm_joint_indices,
        )
        
        # Find end-effector link
        self.ee_link = self.kinematics.end_link
        
        # Find base link
        if self.config.base_link_name:
            self.base_link = sapien_utils.get_obj_by_name(
                self.articulation.get_links(), self.config.base_link_name
            )
        else:
            self.base_link = self.articulation.root

    def _initialize_joints(self):
        """Initialize joints controlled by this controller"""
        # We control base_joint_names + arm_joint_names (11 total DOF)
        #NOTE: why do we need this or?
        joint_names = (self.config.base_joint_names or []) + self.config.arm_joint_names
        self.joints = []
        joint_indices = []
        
        for i, joint in enumerate(self.articulation.get_active_joints()):
            if joint.name in joint_names:
                self.joints.append(joint)
                joint_indices.append(i)
                
        self.active_joint_indices = torch.tensor(joint_indices, device=self.device, dtype=torch.int32)
        
        # Identify non-controlled joints (like head joints, gripper joints)
        all_active_joints = self.articulation.get_active_joints()
        all_joint_indices = list(range(len(all_active_joints)))
        self.non_controlled_joint_indices = [idx for idx in all_joint_indices if idx not in joint_indices]
        self.non_controlled_joints = [all_active_joints[idx] for idx in self.non_controlled_joint_indices]
        self.non_controlled_joint_indices = torch.tensor(self.non_controlled_joint_indices, device=self.device, dtype=torch.int32)
        
        # Ensure we have the expected number of joints
        #NOTE: why do we need to make sure there is base joints but not assume there are?
        expected_joints = len(self.config.arm_joint_names) + (len(self.config.base_joint_names) if self.config.base_joint_names else 0)
        if len(self.joints) != expected_joints:
            print(f"Warning: Expected {expected_joints} joints but found {len(self.joints)}")
            print(f"Expected joint names: {joint_names}")
            print(f"Found joint names: {[j.name for j in self.joints]}")
        
        # Print information about non-controlled joints for debugging
        if len(self.non_controlled_joints) > 0:
            print(f"WholeBodyController: Found {len(self.non_controlled_joints)} non-controlled joints that will be held in place:")
            for joint in self.non_controlled_joints:
                print(f"  - {joint.name}")

    def _initialize_action_space(self):
        """Initialize action space for desired end-effector pose"""
        # Action space is [x, y, z, qw, qx, qy, qz] for pose
        # low = np.array([-np.inf] * 7, dtype=np.float32)
        # high = np.array([np.inf] * 7, dtype=np.float32)
        # Set reasonable default position limits if not specified
        if self.config.pos_limits is not None:
        #     low[:3] = self.config.pos_limits[0]
        #     high[:3] = self.config.pos_limits[1]
            
        # # Quaternion bounds
        # low[3:] = -1.0
        # high[3:] = 1.0    
            pos_low, pos_high = self.config.pos_limits
        else:
            # Default workspace limits for mobile manipulation (in meters)
            pos_low = [-2.0, -2.0, 0.0]  # Can move 2m in x,y and above ground
            pos_high = [2.0, 2.0, 2.0]   # Can reach up to 2m high
        
        low = np.array(pos_low + [-1.0, -1.0, -1.0, -1.0], dtype=np.float32)
        high = np.array(pos_high + [1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        
        self.single_action_space = spaces.Box(low, high, dtype=np.float32)

    def set_drive_property(self):
        """Set drive properties for joints"""
        # Set drive properties for controlled joints
        for joint in self.joints:
            joint.set_drive_properties(
                stiffness=self.config.stiffness,
                damping=self.config.damping,
                force_limit=self.config.force_limit,
                mode=self.config.drive_mode
            )
        
        # Set drive properties for non-controlled joints to hold them in place
        # Use moderate stiffness and damping to prevent drifting without being too stiff
        if len(self.non_controlled_joints) > 0:
            hold_stiffness = self.config.stiffness * 0.5  # Use half the stiffness for holding
            hold_damping = self.config.damping * 0.8     # Use slightly less damping for stability
            
            for joint in self.non_controlled_joints:
                joint.set_drive_properties(
                    stiffness=hold_stiffness,
                    damping=hold_damping,
                    force_limit=self.config.force_limit,
                    mode=self.config.drive_mode
                )

    def reset(self):
        """Reset controller state"""
        self._target_pose = None
        
        # Hold non-controlled joints at their current positions after reset
        # This ensures they don't drift even after environment resets
        if hasattr(self, 'non_controlled_joints') and len(self.non_controlled_joints) > 0:
            current_qpos = self.articulation.get_qpos()
            non_controlled_current_qpos = current_qpos[..., self.non_controlled_joint_indices]
            
            self.articulation.set_joint_drive_targets(
                non_controlled_current_qpos, self.non_controlled_joints, self.non_controlled_joint_indices
            )

    def compute_end_effector_pose(self, q):
        """Forward kinematics to get current end-effector pose"""
        # Use kinematics solver for forward kinematics
        # Extract arm joint positions for kinematics computation
        q_arm = q[..., self.num_base_dof:]  # Last 7 DOF are arm joints
        
        # Compute forward kinematics using the kinematics solver
        # This gives us the pose relative to the root link (torso_lift_link)
        ee_pose_relative = self.kinematics.compute_forward_kinematics(q_arm)
        
        # Get base transformation from current base joint positions
        # Base DOF: [x, y, theta, torso_lift]
        base_x = q[..., 0]
        base_y = q[..., 1] 
        base_theta = q[..., 2]
        torso_lift = q[..., 3]
        
        # Create base transformation matrix
        batch_size = q.shape[0]
        T_base = torch.zeros(batch_size, 4, 4, device=self.device)
        
        # Rotation about z-axis (yaw)
        cos_theta = torch.cos(base_theta)
        sin_theta = torch.sin(base_theta)
        
        T_base[:, 0, 0] = cos_theta
        T_base[:, 0, 1] = -sin_theta
        T_base[:, 1, 0] = sin_theta
        T_base[:, 1, 1] = cos_theta
        T_base[:, 2, 2] = 1.0
        T_base[:, 3, 3] = 1.0
        
        # Translation (x, y, z) 
        # The torso_lift_joint origin is at xyz="-0.086875 0 0.37743" relative to base_link
        # So we need to add this offset plus the current torso_lift position
        base_z_offset = 0.37743  # Base offset from URDF torso_lift_joint origin
        T_base[:, 0, 3] = base_x
        T_base[:, 1, 3] = base_y
        T_base[:, 2, 3] = base_z_offset + torso_lift
        
        # Combine base transformation with arm end-effector pose
        # ee_pose_relative is relative to torso_lift_link
        T_ee_relative = ee_pose_relative.to_transformation_matrix()
        T_ee_world = torch.bmm(T_base, T_ee_relative)
        
        # Convert back to Pose by extracting position and quaternion
        pos = T_ee_world[:, :3, 3]  # Extract position
        rot_mat = T_ee_world[:, :3, :3]  # Extract rotation matrix
        
        # Convert rotation matrix to quaternion
        from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion
        quat = matrix_to_quaternion(rot_mat)
        
        # Create Pose object
        ee_pose_world = Pose.create_from_pq(pos, quat)
        
        return ee_pose_world
        
    def compute_pose_error(self, x_ee_current: Pose, x_ee_desired: Pose):
        """Compute 6D pose error between current and desired pose using SE(3) logarithm map"""
        # TODO: let's first assume this is correct, but needs to be verified.
        # Compute relative transformation: T_rel = T_desired * T_current^(-1)
        # This gives us the transformation from current to desired pose
        
        # Method 1: Using transformation matrices (more numerically stable)
        T_current = x_ee_current.to_transformation_matrix()  # (batch, 4, 4)
        T_desired = x_ee_desired.to_transformation_matrix()  # (batch, 4, 4)
        
        # T_rel = T_desired * T_current^(-1)
        T_current_inv = torch.zeros_like(T_current)
        R_current = T_current[..., :3, :3]
        p_current = T_current[..., :3, 3]
        
        # Compute inverse: T^(-1) = [R^T, -R^T * p; 0, 1]
        T_current_inv[..., :3, :3] = R_current.transpose(-2, -1)
        T_current_inv[..., :3, 3] = -torch.bmm(
            R_current.transpose(-2, -1), 
            p_current.unsqueeze(-1)
        ).squeeze(-1)
        T_current_inv[..., 3, 3] = 1.0
        
        T_rel = torch.bmm(T_desired, T_current_inv)
        
        # Extract translation part (linear velocity)
        v_linear = T_rel[..., :3, 3]
        
        # Extract rotation part and compute logarithm (angular velocity)
        R_rel = T_rel[..., :3, :3]
        
        # Convert rotation matrix to axis-angle representation (SO(3) logarithm)
        # This is equivalent to the rotational part of the SE(3) logarithm
        from mani_skill.utils.geometry.rotation_conversions import matrix_to_quaternion, quaternion_to_axis_angle
        q_rel = matrix_to_quaternion(R_rel)
        v_angular = quaternion_to_axis_angle(q_rel)
        
        # Combine into 6D twist: [v_linear, v_angular]
        # This represents the body frame twist (se(3) coordinates)
        pose_error = torch.cat([v_linear, v_angular], dim=-1)
        
        return pose_error
        
    def compute_jacobian(self, q):
        """Compute geometric Jacobian mapping joint velocities to end-effector twist"""
        # Get current pose
        ee_pose = self.ee_link.pose
        base_pose = self.base_link.pose
        
        # Compute Jacobian using finite differences
        eps = 1e-6
        batch_size = q.shape[0]
        
        # Base Jacobian (4 DOF: x, y, theta, torso_lift)
        J_base = torch.zeros(batch_size, 6, 4, device=self.device)
        
        # Translation in x and y
        J_base[:, 0, 0] = 1.0  # dx/dx = 1
        J_base[:, 1, 1] = 1.0  # dy/dy = 1
        
        # Rotation about z NOTE: this is from rigid body kinematics.
        ee_pos_rel = ee_pose.p - base_pose.p
        J_base[:, 0, 2] = -ee_pos_rel[:, 1]  # dx/dtheta = -y
        J_base[:, 1, 2] = ee_pos_rel[:, 0]   # dy/dtheta = x
        J_base[:, 5, 2] = 1.0  # dtheta/dtheta = 1
        
        # Torso lift (z translation)
        J_base[:, 2, 3] = 1.0  # dz/d(torso_lift) = 1
        
        # Arm Jacobian using kinematics (only for the 7 arm joints)
        # The kinematics solver expects full articulation joint positions, not just WBC subset
        q_full = self.articulation.get_qpos()
        J_arm = self.kinematics.compute_jacobian(q_full)
        # TODO: the formulation is correct, just make sure the parameters and the kinematic chain are correct.
            
        # Combine base and arm Jacobians
        J_full = torch.cat([J_base, J_arm], dim=-1)
        return J_full

    # def _compute_arm_jacobian_numerical(self, q):
    #     """Compute arm Jacobian using numerical differentiation"""
    #     eps = 1e-6
    #     batch_size = q.shape[0]
    #     num_arm_joints = len(self.arm_joint_indices)
        
    #     J_arm = torch.zeros(batch_size, 6, num_arm_joints, device=self.device)
        
    #     # Get current end-effector pose
    #     ee_pose_current = self.ee_link.pose
        
    #     for i in range(num_arm_joints):
    #         # Perturb joint i
    #         q_plus = q.clone()
    #         q_plus[:, self.arm_joint_indices[i]] += eps
            
    #         # Set joint positions and compute forward kinematics
    #         self.articulation.set_qpos(q_plus)
    #         ee_pose_plus = self.ee_link.pose
            
    #         # Compute finite difference
    #         pos_diff = (ee_pose_plus.p - ee_pose_current.p) / eps
            
    #         # Orientation difference (simplified)
    #         q_diff = quaternion_multiply(ee_pose_plus.q, 
    #                                    torch.cat([ee_pose_current.q[..., :1], 
    #                                               -ee_pose_current.q[..., 1:]], dim=-1))
    #         rot_diff = 2.0 * q_diff[..., 1:] / eps
            
    #         J_arm[:, :3, i] = pos_diff
    #         J_arm[:, 3:, i] = rot_diff
            
    #     # Restore original joint positions
    #     self.articulation.set_qpos(q)
    #     return J_arm

    def solve_ik_step(self, q_current, x_ee_desired: Pose):
        """Solve one iteration of the IK optimization using QP"""
        batch_size = q_current.shape[0]
        
        # Get current end-effector pose
        x_ee_current = self.compute_end_effector_pose(q_current)
        
        # Compute pose error
        e_ee = self.compute_pose_error(x_ee_current, x_ee_desired)
        
        # Compute Jacobian
        J_ee = self.compute_jacobian(q_current)
        
        # For now, handle batch size 1 only (single environment)
        if batch_size != 1:
            raise NotImplementedError("Batch size > 1 not supported yet")
            
        q_dot = self._solve_qp_single(J_ee, e_ee, q_current)
        return q_dot

    def _solve_qp_single(self, J_ee, e_ee, q_current):
        """Solve QP for a single sample"""
        batch_size = q_current.shape[0]
        if batch_size > 1:
            raise NotImplementedError("Batched QP solving not implemented yet")
        
        # Convert to numpy for qpsolvers
        J_np = J_ee[0].detach().cpu().numpy()
        e_np = e_ee[0].detach().cpu().numpy()
        q_np = q_current[0].detach().cpu().numpy()
        
        # Validate inputs for NaN/inf values
        if np.any(np.isnan(J_np)) or np.any(np.isinf(J_np)):
            print("Warning: Invalid Jacobian values detected, using zero velocities")
            return torch.zeros((batch_size, self.total_dof), device=q_current.device, dtype=q_current.dtype)
        
        if np.any(np.isnan(e_np)) or np.any(np.isinf(e_np)):
            print("Warning: Invalid pose error detected, using zero velocities")
            return torch.zeros((batch_size, self.total_dof), device=q_current.device, dtype=q_current.dtype)
        
        n_vars = J_np.shape[1]  # Number of DOFs
        
        # Construct QP matrices
        # Objective: min ||J * q_dot - e_ee||^2 + regularization terms
        
        # Primary tracking term: ||J * q_dot - e_ee||^2
        H_track = self.W_ee * (J_np.T @ J_np)
        g_track = -self.W_ee * (J_np.T @ e_np)
        
        # Posture regularization: ||q_current + q_dot * dt - q_retract||^2
        # NOTE: Only apply to torso_lift and arm joints, NOT to mobile base position (x, y, theta)
        if self.q_retract is not None:
            q_retract_np = self.q_retract.detach().cpu().numpy() if torch.is_tensor(self.q_retract) else self.q_retract
            if len(q_retract_np) == len(q_np):
                # Create posture regularization matrix that excludes mobile base position (first 3 DOF)
                H_posture = np.zeros((n_vars, n_vars))
                g_posture = np.zeros(n_vars)
                
                # Only regularize torso_lift (index 3) and arm joints (indices 4-10)
                # Skip mobile base position: base_x (0), base_y (1), base_theta (2)
                posture_indices = list(range(3, n_vars))  # torso_lift + arm joints
                
                if len(posture_indices) > 0:
                    q_diff = q_np - q_retract_np
                    # Apply regularization only to non-base DOF
                    for i in posture_indices:
                        H_posture[i, i] = self.W_posture
                        g_posture[i] = self.W_posture * self.timestep * q_diff[i]
            else:
                H_posture = np.zeros((n_vars, n_vars))
                g_posture = np.zeros(n_vars)
        else:
            H_posture = np.zeros((n_vars, n_vars))
            g_posture = np.zeros(n_vars)
        
        # Base damping: ||q_dot_base||^2
        H_damping = np.zeros((n_vars, n_vars))
        H_damping[:4, :4] = self.W_damping * np.eye(4)  # First 4 DOFs are base (x,y,theta,torso)
        g_damping = np.zeros(n_vars)
        
        # Combine terms
        H = H_track + H_posture + H_damping
        g = g_track + g_posture + g_damping
        
        # Ensure H is positive definite
        H += 1e-3 * np.eye(n_vars)
        
        # Validate QP matrices
        if np.any(np.isnan(H)) or np.any(np.isinf(H)) or np.any(np.isnan(g)) or np.any(np.isinf(g)):
            print("Warning: Invalid QP matrices detected, using zero velocities")
            return torch.zeros((batch_size, self.total_dof), device=q_current.device, dtype=q_current.dtype)

        # Constraints
        # Joint velocity limits
        if hasattr(self.config, 'velocity_limits') and self.config.velocity_limits is not None:
            v_min, v_max = self.config.velocity_limits
            lb = np.full(n_vars, v_min)
            ub = np.full(n_vars, v_max)
        else:
            lb = np.full(n_vars, -1.0)  # Default limits
            ub = np.full(n_vars, 1.0)
            
        # Joint position limits (via velocity integration)
        if hasattr(self.config, 'position_limits') and self.config.position_limits is not None:
            pos_min, pos_max = self.config.position_limits
            if len(pos_min) == len(q_np):
                lb_pos = (pos_min - q_np) / self.timestep
                ub_pos = (pos_max - q_np) / self.timestep
                lb = np.maximum(lb, lb_pos)
                ub = np.minimum(ub, ub_pos)
        
        # Solve QP
        try:
            from qpsolvers import solve_qp
            q_dot_solution = solve_qp(
                P=H, q=g, lb=lb, ub=ub, 
                solver='osqp', verbose=False
            )
            
            if q_dot_solution is None:
                # Fallback: least squares solution
                q_dot_solution = np.linalg.lstsq(J_np, e_np, rcond=None)[0]
                
            return torch.tensor(q_dot_solution, device=q_current.device, dtype=torch.float32).unsqueeze(0)
            
        except Exception as e:
            print(f"QP solver error: {e}")
            return torch.zeros((batch_size, self.total_dof), device=q_current.device, dtype=q_current.dtype)
        
    def execute(self, x_ee_desired: Pose, q_current):
        """Main WBC execution: map desired end-effector pose to joint commands"""
        # Iterative IK solving
        q_command = q_current.clone()
        
        for iteration in range(self.max_iterations):
            # Solve for joint velocities
            q_dot = self.solve_ik_step(q_command, x_ee_desired)
            
            # Integrate velocities
            q_command = q_command + q_dot * self.timestep
            
            # Check convergence
            x_ee_current = self.compute_end_effector_pose(q_command)
            pose_error = self.compute_pose_error(x_ee_current, x_ee_desired)
            
            if torch.norm(pose_error, dim=-1).max() < self.convergence_threshold:
                break
                
        return q_command

    def set_action(self, action: Array):
        """Set the target end-effector pose from action"""
        action = self._preprocess_action(action)
        
        # Extract pose from action [x, y, z, qw, qx, qy, qz]
        pos = action[..., :3]
        quat = action[..., 3:]
        
        # Normalize quaternion
        quat = quat / torch.norm(quat, dim=-1, keepdim=True)
        
        # Create target pose
        self._target_pose = Pose.create_from_pq(pos, quat)
        
        # Get current joint positions for WBC DOF only
        q_full = self.articulation.get_qpos()
        q_current = q_full[..., self.active_joint_indices]  # Extract only WBC joints
        
        # Execute whole body control
        q_target = self.execute(self._target_pose, q_current)
        
        # Set drive targets (q_target is already the right size for WBC joints)
        self.set_drive_targets(q_target)

    def set_drive_targets(self, targets):
        """Set drive targets for controlled joints and hold non-controlled joints at current positions"""
        # Set targets for controlled joints
        self.articulation.set_joint_drive_targets(
            targets, self.joints, self.active_joint_indices
        )
        
        # Hold non-controlled joints (like head joints) at their current positions
        # This prevents them from drifting due to inertia when the robot moves
        if len(self.non_controlled_joints) > 0:
            # Get current positions of non-controlled joints
            current_qpos = self.articulation.get_qpos()
            non_controlled_current_qpos = current_qpos[..., self.non_controlled_joint_indices]
            
            # Set drive targets for non-controlled joints to their current positions
            self.articulation.set_joint_drive_targets(
                non_controlled_current_qpos, self.non_controlled_joints, self.non_controlled_joint_indices
            )

    def get_state(self) -> dict:
        """Get controller state"""
        if self._target_pose is not None:
            return {"target_pose": self._target_pose.raw_pose}
        return {}

    def set_state(self, state: dict):
        """Set controller state"""
        if "target_pose" in state:
            target_pose = state["target_pose"]
            self._target_pose = Pose.create_from_pq(
                target_pose[..., :3], target_pose[..., 3:]
            )

    def interpolate_to_target(self, x_ee_start: Pose, x_ee_target: Pose, alpha: float):
        """Interpolate between start and target poses for smooth motion"""
        # Linear interpolation for position
        pos_interp = (1 - alpha) * x_ee_start.p + alpha * x_ee_target.p
        
        # SLERP for orientation (quaternion)
        q_start = x_ee_start.q
        q_target = x_ee_target.q
        
        # Ensure quaternions are in same hemisphere
        dot_product = torch.sum(q_start * q_target, dim=-1, keepdim=True)
        q_target = torch.where(dot_product < 0, -q_target, q_target)
        
        # SLERP
        q_interp = self._slerp(q_start, q_target, alpha)
        
        return Pose.create_from_pq(pos_interp, q_interp)

    def _slerp(self, q1, q2, t):
        """Spherical linear interpolation for quaternions"""
        dot = torch.sum(q1 * q2, dim=-1, keepdim=True)

        # If quaternions are very close, use linear interpolation
        if torch.any(torch.abs(dot) > 0.9995):
            result = q1 + t * (q2 - q1)
            return result / torch.norm(result, dim=-1, keepdim=True)
        
        # Calculate angle between quaternions
        theta_0 = torch.acos(torch.abs(dot))
        theta = theta_0 * t
        
        # Orthogonal quaternion
        q2_perp = q2 - q1 * dot
        q2_perp = q2_perp / torch.norm(q2_perp, dim=-1, keepdim=True)
        
        # SLERP result
        result = q1 * torch.cos(theta) + q2_perp * torch.sin(theta)
        return result


@dataclass
class WholeBodyControllerConfig(ControllerConfig):
    """Configuration for WholeBodyController"""
    
    # Joint names
    arm_joint_names: List[str]
    """Names of arm joints to control"""
    base_joint_names: List[str] = None
    """Names of base joints (x, y, theta, torso_lift)"""
    
    # Kinematics
    urdf_path: str = None
    """Path to URDF file for kinematics"""
    ee_link_name: str = "gripper_link"
    """Name of end-effector link"""
    base_link_name: str = None
    """Name of base link"""
    
    # # Control parameters
    stiffness: Union[float, List[float]] = 1000.0
    damping: Union[float, List[float]] = 100.0
    force_limit: Union[float, List[float]] = 100.0
    
    # WBC weights
    ee_tracking_weight: float = 1.0  # Reduced from 10.0 to be less aggressive
    """Weight for end-effector tracking objective"""
    posture_regularization_weight: float = 2e-2  # Increased from 1.0 for stability
    """Weight for posture regularization"""
    base_damping_weight: float = 1.5  # Increased from 1.5 to reduce oscillations
    """Weight for base damping"""
    
    # Solver parameters
    max_iterations: int = 20  # Reduced for smaller per-step changes
    """Maximum iterations for iterative IK"""
    convergence_threshold: float = 1e-4  # Tighter convergence threshold
    """Convergence threshold for IK"""
    
    # Limits
    pos_limits: Optional[tuple] = None
    """Position limits for end-effector (min, max)"""
    velocity_limits: Optional[tuple] = (-0.5, 0.5)  # More conservative velocity limits
    """Velocity limits for joints (min, max)"""
    position_limits: Optional[tuple] = None
    """Position limits for joints (min, max)"""
    
    # Neutral posture
    neutral_posture: Optional[Array] = None
    """Neutral posture for regularization"""
    
    # Collision avoidance
    collision_margin: float = 0.02
    """Safety margin for collision avoidance (2cm)"""
    collision_detection_range: float = 0.10
    """Range for collision detection (10cm)"""
    
    # Control mode
    normalize_action: bool = False
    drive_mode: DriveMode = "force"
    
    controller_cls = WholeBodyController