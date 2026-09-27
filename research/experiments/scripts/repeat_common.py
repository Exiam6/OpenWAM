"""Read-only diagnostic helpers; no policy or RNG mutation."""
import hashlib,json
import numpy as np

def payload_hash(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def fresh_payload(payload):
    """Each wire request is decoded anew; native preprocessing mutates its dict."""
    return json.loads(json.dumps(payload,allow_nan=False))

def array_hash(value):
    a=np.ascontiguousarray(value)
    return hashlib.sha256(str(a.dtype).encode()+str(a.shape).encode()+a.tobytes()).hexdigest()

def traced_generate(generate, sink):
    def traced(*args,**kwargs):
        result=generate(*args,**kwargs)
        actions=result['actions']
        a=actions.detach().cpu().numpy() if hasattr(actions,'detach') else np.asarray(actions)
        sink.append(a.copy())
        return result
    return traced

def difference(a,b):
    a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape and a.shape[-1]==20
    assert np.isfinite(a).all() and np.isfinite(b).all()
    d=a.astype(np.float64)-b.astype(np.float64)
    out={'byte_equal':array_hash(a)==array_hash(b),'max_abs':float(np.abs(d).max()),'rmse':float(np.sqrt(np.mean(d*d))),'mean_abs':float(np.abs(d).mean())}
    groups={'xyz':[0,1,2,10,11,12],'rot6d':list(range(3,9))+list(range(13,19)),'gripper':[9,19]}
    out['groups']={k:{'max_abs':float(np.abs(d[...,v]).max()),'rmse':float(np.sqrt(np.mean(d[...,v]**2)))}for k,v in groups.items()}
    return out
