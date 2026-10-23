"""Check the submission layout and metadata; add --weights to check local bytes."""
import argparse,ast,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parent
MODELS=('ijepa-vit-h14-pretrained','ijepa-vit-h14-adapted','mobilenetv5-300m','dinov3-vit-l16')

def read(path):return json.loads(path.read_text())
def sha256(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()

def verify(weights=False):
    checked={};split=None
    for name in MODELS:
        folder=ROOT/'models'/name
        expected={'README.md','config.json','data.json','results.json','requirements.txt','run.py','code','licenses','weights','tests'}
        actual={p.name for p in folder.iterdir() if p.name not in {'assets','__pycache__','.pytest_cache','runs'}}
        assert actual==expected,(name,actual)
        config=read(folder/'config.json');data=read(folder/'data.json');result=read(folder/'results.json')
        manifest=read(folder/'weights/manifest.json')
        assert manifest['model_id']==name
        assert manifest['format']=='tig-model-weights-v1'
        assert set(manifest['artifacts'])=={'best_encoder','best_classifier'}
        inference=manifest['inference']
        assert inference['source_model_config_sha256']==sha256(folder/'config.json')
        assert inference['threshold']==result['threshold']==config['classifier']['threshold']==0.5
        assert inference['assessment_used_to_select'] is False
        assert result['status']=='completed'
        assert split is None or data==split,'Data splits differ'
        split=data
        assert result['validation']['frames']==data['roles']['selection']['frames']
        assert result['assessment']['frames']==data['roles']['assessment']['frames']
        for role,item in manifest['artifacts'].items():
            path=folder/'weights'/item['path']
            assert Path(item['path']).parts[0]=='best'
            assert '..' not in Path(item['path']).parts
            assert re.fullmatch('[0-9a-f]{64}',item['sha256'])
            assert sum(part['bytes'] for part in item['parts'])==item['bytes']
            if weights:
                assert path.is_file() and not path.is_symlink(),path
                assert path.stat().st_size==item['bytes'] and sha256(path)==item['sha256'],path
        assert result['weights']['classifier_sha256']==manifest['artifacts']['best_classifier']['sha256']
        assert {p.name for p in (folder/'weights').iterdir() if p.name!='__pycache__'}=={'best','manifest.json','README.md'}
        assert {p.name for p in (folder/'weights/best').iterdir()} <= {'encoder.pt','encoder.safetensors','classifier.pt'}
        for path in (folder/'code').rglob('*.py'):ast.parse(path.read_text())
        if inference['family']=='ijepa':
            source=folder/'code/ijepa/vendor'
            for item in read(source/'SOURCE_MANIFEST.json')['files']:
                assert sha256(source/item['file'])==item['local_sha256']
        required = (['ijepa/train.py','ijepa/classifier.py','ijepa/evaluate.py','ijepa/data.py','ijepa/model.py']
                    if inference['family']=='ijepa' else
                    ['tig_benchmark/pipeline.py','tig_benchmark/models.py','tig_benchmark/data.py','transfer_core/classifier.py'])
        if inference['family']=='mobilenetv5':required += ['tig_benchmark/finetune.py']
        for relative in required+['evaluation.py','inference.py','weights.py']:
            assert (folder/'code'/relative).is_file(), (name, 'Missing required source', relative)
        assert (folder/'weights/README.md').is_file()
        assert (folder/'tests').is_dir()
        checked[name]={'best_checkpoint':True,'weight_bytes_verified':weights}
    for path in ROOT.rglob('*.md'):
        if any(x in path.parts for x in ('assets','.git')):continue
        for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
            if re.match(r'\w+://',link) or link.startswith('#'):continue
            assert (path.parent/link.split('#')[0]).exists(),(path,link)
    return {'passed':True,'same_data_split':True,'models':checked}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights',action='store_true')
    args=parser.parse_args()
    print(json.dumps(verify(args.weights),indent=2))
