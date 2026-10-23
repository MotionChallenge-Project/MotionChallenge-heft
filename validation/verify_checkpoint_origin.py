# /// script
# requires-python = ">=3.12"
# dependencies = ["onnx==1.20.1", "onnxruntime==1.30.0", "numpy>=2,<3"]
# ///
"""Verify the four-input wrapper against the pinned author's ONNX.

uv run validation/verify_checkpoint_origin.py AUTHOR.onnx AUTHOR.onnx.data WRAPPER.onnx
Supply files linked in provenance.json; this never changes an evaluation checkpoint.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
import onnx
import onnxruntime as ort


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('author',type=Path);parser.add_argument('data',type=Path);parser.add_argument('wrapper',type=Path)
    args=parser.parse_args()
    record=json.loads(Path(__file__).with_name('checkpoint-origin.json').read_text())
    for path,key in [(args.author,'author_graph_sha256'),(args.data,'author_data_sha256'),(args.wrapper,'packaged_sha256')]:
        assert hashlib.sha256(path.read_bytes()).hexdigest()==record[key],str(path)+' checksum mismatch'
    native=onnx.load(args.author,load_external_data=False)
    # Only the recorded data file is loaded, regardless of the supplied filename.
    with tempfile.TemporaryDirectory() as temporary:
        folder=Path(temporary);(folder/'policy.onnx.data').write_bytes(args.data.read_bytes())
        onnx.external_data_helper.load_external_data_for_model(native,str(folder))
    onnx.external_data_helper.convert_model_from_external_data(native)
    wrapped=onnx.load(args.wrapper)
    nt={t.name:onnx.numpy_helper.to_array(t) for t in native.graph.initializer}
    wt={t.name:onnx.numpy_helper.to_array(t) for t in wrapped.graph.initializer}
    assert nt.keys()==wt.keys() and all(np.array_equal(nt[k],wt[k]) for k in nt)
    assert [n.SerializeToString() for n in native.graph.node]==[n.SerializeToString() for n in wrapped.graph.node[1:]]
    concat=wrapped.graph.node[0]
    assert concat.op_type=='Concat' and list(concat.input)==['context','motion_command','target_motion','proprioception'] and list(concat.output)==['policy']
    assert len(concat.attribute)==1 and concat.attribute[0].name=='axis' and concat.attribute[0].i==1
    opts=ort.SessionOptions();opts.intra_op_num_threads=1
    a=ort.InferenceSession(native.SerializeToString(),sess_options=opts,providers=['CPUExecutionProvider'])
    b=ort.InferenceSession(str(args.wrapper),sess_options=opts,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(record['random_seed']);maximum=0.
    for _ in range(record['cases']):
        feed={x.name:rng.standard_normal(x.shape).astype(np.float32) for x in b.get_inputs()}
        vector=np.concatenate([feed[k] for k in concat.input],axis=1)
        maximum=max(maximum,float(np.max(np.abs(a.run(['action'],{'policy':vector})[0]-b.run(['action'],feed)[0]))))
    assert maximum<=1e-6,maximum
    print(json.dumps({'cases':record['cases'],'action_max_abs_error':maximum,'initializer_count':len(nt),'native_nodes_preserved_exactly':True}))

if __name__=='__main__':main()
