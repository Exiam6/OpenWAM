import unittest
from probe import system_display


class ProcessTests(unittest.TestCase):
    def test_only_tiny_system_xorg_is_background(self):
        item=dict(type='G',user='gdm',name='/usr/lib/xorg/Xorg',memory_mib=4)
        self.assertTrue(system_display(item))
        for change in [dict(user='zifanz4'),dict(user='another-user'),dict(type='C+G'),dict(memory_mib=17),dict(name='blender')]:
            self.assertFalse(system_display(dict(item,**change)))


if __name__=='__main__':unittest.main()
