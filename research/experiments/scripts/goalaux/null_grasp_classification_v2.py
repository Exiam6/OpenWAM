"""Native basket grasp null-candidate sentinel at all three source call sites."""
import re

def is_native_no_grasp(record):
 if record.get('task')!='place_object_basket' or record.get('accepted') or record.get('outcome')!='infrastructure_failure':return False
 if record.get('error')!="AssertionError('target_pose cannot be None for move action.')" or record.get('exit_code',0)!=0:return False
 trace=record.get('traceback','')
 return all(re.search(pattern,trace) is not None for pattern in [r'place_object_basket\.py", line (?:73|112|122), in play_once',r'_base_task\.py", line 1205, in grasp_actor\n\s+Action\(arm_tag, "move", target_pose=pre_grasp_pose\)',r'utils/action\.py", line 76, in __init__'])

def annotate(record):
 if not is_native_no_grasp(record):return dict(record)
 d=dict(record);d.update(original_outcome=record['outcome'],outcome='expert_infeasible',classification_amendment='native basket grasp returned no candidate at an AST-enumerated call site; exact pre-score source/traceback class; original result retained; no retry');return d
