"""Directory routing only; frozen representations, scores and criteria unchanged."""
import sys
import score_confirmation as original
original.O=original.B/'confirmation-v2';original.F=original.B/'fresh-v3'
if __name__=='__main__':
 if sys.argv[1]=='extract':original.extract()
 elif sys.argv[1]=='score':original.score()
 else:raise ValueError(sys.argv[1])
