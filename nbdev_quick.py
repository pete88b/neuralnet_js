__all__=['nbdev_quick']

import re
from IPython.core.magic import Magics, magics_class, cell_magic
from pathlib import Path

_re_class_func_def = re.compile(r"""
# Catches any 0-indented function or class definition with its name in group 1
(?:async\sdef|def|class)  # Non-catching group for def or class
\s+                     # One whitespace or more
([^\(\s]+)            # Catching group with any character except an opening parenthesis or a whitespace (name)
\s*                 # Any number of whitespace
(?:\(|:)          # Non-catching group with either an opening parenthesis or a : (classes don't need ())
""", re.VERBOSE)

@magics_class
class NbdevQuick(Magics):
    
    def _header(self):
        return '# Created by NbdevQuick\n'
        # TODO: add link to project etc

    def init(self,module_name='quick_module',auto_import='wildcard'):
        """set the `module_name`, create the module .py (if it doesn't exist) and otionally import it"""
        self.file=Path(f'{module_name}.py')
        if not self.file.exists():
            with open(self.file, 'w') as f: f.write(self._header())
        if auto_import=='wildcard':
            self.shell.run_cell(f'from {module_name} import *')
        elif auto_import=='module':
            self.shell.run_cell(f'import {module_name}')

    def _cell_name(self,lines):
        """Extract the name of the cell from the first line. The name will be:
        a class name, a function name or the first piece of text on the line"""
        line=lines[0]
        match=_re_class_func_def.match(line)
        if match is not None: return match.group(1)
        return re.split('=| ', line)[0]
        
    def _read_file(self,start_tag,end_tag):
        """Returns module source, skipping over the section between start and end tags"""
        if not self.file.exists(): return [self._header()]
        lines,in_section=[],False
        with open(self.file) as f: 
            for l in f:
                if start_tag==l: in_section=True; continue
                if end_tag==l: in_section=False; continue
                if not in_section: lines.append(l)
        if not lines[-1].endswith('\n'): lines.append('\n')
        return lines

    def _write_to_file(self,cell):
        """write the source of the cell to the module file"""
        # tform things like %magic and !system commands
        # TODO: this is prety clever but ... might it be better to encourage plain py only?
        cell=self.shell.input_transformer_manager.transform_cell(cell)
        lines=re.split('(\n)',cell)
        cell_name=self._cell_name(lines)
        start_tag=f'#NbdevQuick:start({cell_name})\n'
        end_tag=f'#NbdevQuick:end({cell_name})\n'
        file_lines=self._read_file(start_tag,end_tag)
        file_lines.extend([start_tag]+lines+[end_tag])
        with open(self.file, 'w') as f: f.write(''.join(file_lines))
        
    @cell_magic
    def nbdev_export(self,line,cell):
        """NbdevQuick: Put an `%%nbdev_export` magic on each cell you want exported"""    
        if not self.shell.run_cell(cell).success: return
        # if `run_cell` fails, we don't want to change the module
        #TODO: read module from line?
        self._write_to_file(cell)

nbdev_quick=NbdevQuick(get_ipython())
nbdev_quick.shell.register_magics(nbdev_quick)