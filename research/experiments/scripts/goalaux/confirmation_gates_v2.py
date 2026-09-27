"""Directory routing only; original numerical/data gates unchanged."""
import sys
import confirmation_gates as original
original.O=original.B/'confirmation-v2';original.F=original.B/'fresh-v3'
if __name__=='__main__':
 if sys.argv[1]=='identity':original.identity()
 elif sys.argv[1]=='parity':original.parity()
 else:raise ValueError(sys.argv[1])
