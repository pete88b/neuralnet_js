# Created by NbdevQuick
#NbdevQuick:start(from)
from pathlib import Path
import glob, re, nbconvert, nbformat
# TODO: describe conda env / dependencies
# TODO: add this to Dockerfile and use py notebook and command line from there
#NbdevQuick:end(from)
#NbdevQuick:start(read_nb)
def read_nb(fname):
    "Read the notebook in `fname`."
    with open(Path(fname),'r', encoding='utf8') as f: return nbformat.reads(f.read(), as_version=4)
#NbdevQuick:end(read_nb)
#NbdevQuick:start(check_re)
def check_re(cell, pat, code_only=True):
    "Check if `cell` contains a line with regex `pat`"
    if code_only and cell['cell_type'] != 'code': return
    if isinstance(pat, str): pat = re.compile(pat, re.IGNORECASE | re.MULTILINE)
    return pat.search(cell['source'])
#NbdevQuick:end(check_re)
#NbdevQuick:start(_find_default_exp)
def _find_default_exp(nb):
    m=re.search(r'^//default_exp (\w+)', nb['cells'][0]['source'])
    if m is None: return None
    return m.group(1)
#NbdevQuick:end(_find_default_exp)
#NbdevQuick:start(_notebook2script)
def _notebook2script(fname, target):
    fname,target,nb = Path(fname),Path(target),read_nb(fname)
    default_exp=_find_default_exp(nb)
    if default_exp is None: return
    target.mkdir(parents=True,exist_ok=True)
    target_file=target/f'{default_exp}.js'
    print('Converting',fname,'to',target_file)
    with open(target_file, 'w') as f:
        for cell in nb['cells']:
            if check_re(cell, r'(^/\*\*)|(^export )') is not None:
                f.write(cell['source'])
                f.write('\n\n')
#NbdevQuick:end(_notebook2script)
#NbdevQuick:start(notebook2script)
def notebook2script(fname=None, target='src'):
    for f in glob.glob('./[!_]*.ipynb' if fname is None else fname):
        _notebook2script(f, target)
#NbdevQuick:end(notebook2script)
#NbdevQuick:start(notebook2md)
def notebook2md(fname='index.ipynb'):
    "Convert a notebook to README.md in the current working directory"
    print('Converting',fname,'to README.md')
    converter=nbconvert.MarkdownExporter()
    md,resources=converter.from_filename(fname)
    with open('README.md','w') as f: f.write(md)
#NbdevQuick:end(notebook2md)
#NbdevQuick:start(try:)
try: IN_NOTEBOOK = 'google.colab' in str(get_ipython()) or 'ZMQInteractiveShell' in str(get_ipython())
except: IN_NOTEBOOK = False
#NbdevQuick:end(try:)
#NbdevQuick:start(import)
import argparse
if __name__ == '__main__' and not IN_NOTEBOOK:
    parser = argparse.ArgumentParser(description='Convert notebooks to js scripts.')
    parser.add_argument('fname', default=None, nargs='?', help='Pathname to glob.')
    parser.add_argument('target', default='src', nargs='?',
                        help='Name of directory to write js scripts to.')
    args = parser.parse_args()
    notebook2script(args.fname, args.target)
    if Path('index.ipynb').is_file(): notebook2md()
#NbdevQuick:end(import)
