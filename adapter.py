"""HEFT PMG input assembly from named SDK state and reference packets."""
import numpy as np
from scipy.spatial.transform import Rotation as R
from motion_challenge.adapters.onnx import ONNXController


class HEFT(ONNXController):
    raw_action_history=True

    def reset(self,state,reference,seed):
        super().reset(state,reference,seed)
        self.boot=25
        self.lags=[0,1,2,3,4,8,12,16,20]
        self.buffers=[np.zeros((21,n),np.float32) for n in [3,3,29,29]]
        self.actions=np.zeros((8,29),np.float32)
        self.select=[self.runtime['reference_steps'].index(i) for i in [0,1,2,3,4,5,6,-1,-2,-4,-8,-12,-16]]

    def observations(self,state,reference):
        q,qd,w,g,actual=self.proprio(state)
        g=g/(np.linalg.norm(g)+1e-8)
        self.buffers=[np.concatenate([np.asarray(v,np.float32)[None],h[:-1]]) for h,v in zip(self.buffers,[w,g,q,qd])]
        self.actions=np.concatenate([self.last[None],self.actions[:-1]])
        self.boot=max(0,self.boot-1)
        refs=[reference[i] for i in self.select]
        positions=np.array([f.root_position_world_m for f in refs])
        rotations=R.from_quat([f.root_orientation_xyzw for f in refs])
        joints=np.array([f.joint_position_rad for f in refs])[:,self.order]
        displacement=rotations[0].inv().apply(positions[1:]-positions[0])
        # HEFT uses two columns in column-major order, unlike several other releases.
        rot6=(actual.inv()*rotations).as_matrix()[:,:,:2].transpose(0,2,1)
        down=rotations.inv().apply(np.tile([0,0,-1],(len(refs),1)))
        return {'context':np.array([[self.boot/25.]]),
                'motion_command':np.r_[displacement.reshape(-1),rot6.reshape(-1)][None],
                'target_motion':np.r_[joints.reshape(-1),(joints-q).reshape(-1),positions[:,2],down.reshape(-1)][None],
                'proprioception':np.concatenate([h[self.lags].reshape(-1) for h in self.buffers]+[self.actions.reshape(-1)])[None]}


def create(root,runtime):return HEFT(root,runtime)
