def pci(s):
 a,b,c=s.split(':');d,e=c.split('.');return tuple(int(x,16) for x in [a,b,d,e])
