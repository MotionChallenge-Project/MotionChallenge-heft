"""Compare HEFT observations, reset, network and control with the pinned author.

Supply the author source checkout, original embedded ONNX and packaged weights.
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np
import onnxruntime as ort
import yaml
from scipy.spatial.transform import Rotation as R

from adapter import HEFT
from motion_challenge.sdk import ReferenceFrame, StatePacket


def definitions(path, scope, names, methods):
    nodes = []
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.ClassDef) and node.name in names:
            node.body = [item for item in node.body if isinstance(item, ast.Assign)
                         or isinstance(item, ast.FunctionDef) and item.name in methods]
            nodes.append(node)
    module = ast.Module(ast.parse('from __future__ import annotations').body + nodes, [])
    exec(compile(module, str(path), 'exec'), scope)


def main():
    parser = argparse.ArgumentParser()
    for name in ('source', 'weights', 'author-weights', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pins = json.loads((root / 'native-sources.json').read_text())
    for name, expected in pins['files'].items():
        assert hashlib.sha256((args.source / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256(args.author_weights.read_bytes()).hexdigest() == pins['embedded_author_onnx_sha256']
    runtime = json.loads((root / 'runtime.json').read_text())
    runtime['weights'] = str(args.weights.resolve())
    policy = HEFT(root, runtime)
    src = args.source / 'sim2real/src'
    sys.path.insert(0, str(src))
    from common.joint_mapper import JointMapper
    scope = {'np': np}
    definitions(src / 'runtime/policy.py', scope, ['Policy', 'TrackingPolicyRaw', 'ONNXModule'],
                {'_build_obs_modules', '_reset_obs_modules', 'reset', 'update_obs', 'compute_action', '__call__'})
    definitions(src / 'deploy.py', scope, ['Controller'], {'_apply_action'})
    ctrl_cfg = yaml.safe_load((args.source / 'sim2real/config/g1/controller.yaml').read_text())
    cfg = yaml.safe_load((args.source / 'sim2real/config/g1/tracking.yaml').read_text())
    assert policy.names == cfg['action_joint_names'] == ctrl_cfg['policy_joint_names']
    for actual, expected in ((policy.default, ctrl_cfg['default_qpos']), (policy.scale, cfg['action_scale']),
                             (policy.kp, ctrl_cfg['kps']), (policy.kd, ctrl_cfg['kds'])):
        np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-7)
    assert policy.cfg['action_clip'] == cfg['action_clip']
    assert sorted(cfg['future_steps']) == runtime['reference_steps']
    np.testing.assert_allclose(runtime['reference_offsets_s'], np.array(runtime['reference_steps']) / 50, atol=1e-12)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.author_weights), sess_options=options, providers=['CPUExecutionProvider'])
    model = scope['ONNXModule'].__new__(scope['ONNXModule'])
    model.ort_session = session
    model.in_keys = ['policy']
    model.out_keys = [node.name for node in session.get_outputs()]
    errors = {key: 0. for key in ('context', 'motion_command', 'target_motion', 'proprioception',
              'network', 'action', 'motor_target', 'synthetic_action_history', 'synthetic_motor_target')}
    def compare(key, actual, expected):
        errors[key] = max(errors[key], float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)))))
    rng = np.random.default_rng(241024)
    names = tuple(np.array(policy.names)[rng.permutation(29)])
    order = [names.index(name) for name in policy.names]
    max_action = 0.
    widths = (1, 114, 806, 808)
    assert sum(widths) == 1729
    for reset in range(4):
        ctrl = scope['Controller'].__new__(scope['Controller'])
        ctrl.config = NS(policy_joint_names=policy.names)
        ctrl.dof_size = 29
        ctrl.default_qpos = np.array(ctrl_cfg['default_qpos'], np.float32)
        ctrl.kps, ctrl.kds = np.array(ctrl_cfg['kps'], np.float32), np.array(ctrl_cfg['kds'], np.float32)
        for field in ('cmd_q', 'cmd_qd', 'cmd_kp', 'cmd_kd'):
            setattr(ctrl, field, np.zeros(29, np.float32))
        native = scope['TrackingPolicyRaw'].__new__(scope['TrackingPolicyRaw'])
        native.controller, native.config = ctrl, NS(**cfg)
        native.n_joints, native.use_compliance_flag_obs = 29, False
        native.action_clip, native.action_scale = cfg['action_clip'], np.array(cfg['action_scale'], np.float32)
        native.last_action, native.applied_action = np.zeros(29, np.float32), np.zeros(29, np.float32)
        native.mapper_action = JointMapper(policy.names, policy.names)
        native.module = model
        native._build_obs_modules()
        assert native.num_obs == 1729
        native.reset()
        for step in range(40):
            actual = R.random(random_state=rng)
            q, qd, omega = rng.normal(0, .35, 29), rng.normal(0, .5, 29), rng.normal(0, .3, 3)
            state = StatePacket(step * .02, names, tuple(q), tuple(qd), (0., 0., .8), tuple(actual.as_quat()),
                                (0., 0., 0.), tuple(actual.apply(omega)))
            frames = [ReferenceFrame(i * .02, i * .02, tuple(rng.normal(0, .35, 29)),
                       tuple(rng.normal(0, .3, 3)), tuple(R.random(random_state=rng).as_quat()),
                       (0.,) * 29, (0.,) * 3, (0.,) * 3) for i in range(40)]
            refs = tuple(frames[np.clip(step + offset, 0, 39)] for offset in runtime['reference_steps'])
            native.ref_idx, native.ref_len = step - 1, len(frames)
            native.ref_joint_pos = np.array([f.joint_position_rad for f in frames], np.float32)[:, order]
            native.ref_root_pos = np.array([f.root_position_world_m for f in frames], np.float32)
            native.ref_root_quat = np.array([f.root_orientation_xyzw for f in frames])[:, [3, 0, 1, 2]]
            ctrl.qj, ctrl.dqj = q[order].astype(np.float32), qd[order].astype(np.float32)
            ctrl.quat, ctrl.gyro = actual.as_quat()[[3, 0, 1, 2]], omega.astype(np.float32)
            if step == 0:
                policy.reset(state, refs, reset)
            native.update_obs()
            inputs = policy.observations(state, refs)
            segments = dict(zip(('context', 'motion_command', 'target_motion', 'proprioception'),
                           np.split(native.policy_input['policy'], np.cumsum(widths)[:-1], axis=1)))
            for key, value in segments.items():
                compare(key, inputs[key], value)
            expected = model(native.policy_input)['action']
            max_action = max(max_action, float(np.max(np.abs(expected))))
            compare('network', policy.infer(segments)['action'], expected)
            ctrl._apply_action(native.compute_action())
            observation_fn = policy.observations
            policy.observations = lambda *_: inputs
            out = policy.act(state, refs)
            policy.observations = observation_fn
            compare('action', policy.last, native.last_action)
            compare('motor_target', np.array(out.joint_position_target_rad)[order], ctrl.cmd_q)
        synthetic = np.linspace(-20, 20, 29, dtype=np.float32)[None]
        native.module = lambda _: {'action': synthetic}
        ctrl._apply_action(native.compute_action())
        infer_fn = policy.infer
        policy.infer = lambda _: {'action': synthetic}
        out = policy.act(state, refs)
        policy.infer = infer_fn
        compare('synthetic_motor_target', np.array(out.joint_position_target_rad)[order], ctrl.cmd_q)
        compare('synthetic_action_history', policy.last, native.last_action)
        native.update_obs()
        next_inputs = policy.observations(state, refs)
        compare('synthetic_action_history', next_inputs['proprioception'][0, -232:-203],
                native.policy_input['policy'][0, -232:-203])
    limits = {key: 2e-4 for key in errors}
    limits['network'] = 0.
    report = {'passed': all(errors[key] <= limits[key] for key in errors), 'cases': 160, 'resets': 4,
              'max_abs_error': errors, 'limits': limits, 'max_abs_native_action': max_action,
              'checkpoint_sha256': runtime['weights_sha256'], 'source_commit': pins['source_commit'],
              'scope': 'Pinned author observation modules, reset/boot, reference-index update, original ONNX, '
                       'clipped action history and final controller target. SDK joint order shuffled; '
                       'start/end reference clamping and synthetic out-of-range action covered. '
                       'Transport and hardware imports omitted.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    assert report['passed']


if __name__ == '__main__':
    main()
