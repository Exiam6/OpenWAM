"""Permit only the observed exception caught by native expert candidate filtering."""
import re
HEADER='[eval_policy_wrapper] exception in place_object_basket.play_once'
EXPECTED='AssertionError: target_pose cannot be None for move action.'
PATTERN=re.compile(re.escape(HEADER)+r'\nTraceback \(most recent call last\):\n(?:(?!\[eval_policy_wrapper\]|error occurs !).)*?'+re.escape(EXPECTED)+r'\nerror occurs !\n?',re.S)
FATAL=['CUDA/Curobo prewarm failed:','planner fallback engaged:','error occurs !','Traceback (most recent call last):']
def classify(text, final=False):
 blocks=[]
 def remove(m):
  block=m.group(0)
  if any(marker in block for marker in ['CUDA/Curobo prewarm failed:', 'planner fallback engaged:']):return block
  if 'in grasp_actor' not in block or 'in __init__' not in block or 'in play_once' not in block or block.count('Traceback (most recent call last):')!=1:return block
  exceptions=[x for x in block.splitlines() if re.match(r'^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt):',x)]
  if exceptions!=[EXPECTED]:return block
  blocks.append(block);return ''
 clean=PATTERN.sub(remove,text);normalized=clean
 # Logs can be observed between native traceback writes. A bounded trailing
 # block may remain pending until its exception/catch marker arrives.
 if not final and HEADER in clean:
  prefix,tail=clean.rsplit(HEADER,1);lines=tail.splitlines()
  exceptions=[x for x in lines if re.match(r'^[A-Za-z_][\w.]*(?:Error|Exception|Interrupt):',x)]
  if len(lines)<=35 and 'error occurs !' not in tail and all(x==EXPECTED for x in exceptions):clean=prefix
 return dict(rejected_candidates=len(blocks),fatal=[x for x in FATAL if x in clean],pending=HEADER in text and clean!=normalized if not final else False)
