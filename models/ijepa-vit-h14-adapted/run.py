"""Submission commands for this model."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'code'))
from weights import download,read_manifest,sha256

parser=argparse.ArgumentParser(description='Download, verify and run the best checkpoint.')
commands=parser.add_subparsers(dest='command',required=True)
commands.add_parser('fetch-weights')
commands.add_parser('verify-weights')
p=commands.add_parser('fetch-source');p.add_argument('--output')
p=commands.add_parser('predict');p.add_argument('--image',required=True);p.add_argument('--device',default='cpu');p.add_argument('--assets')
args=parser.parse_args()
manifest=ROOT/'weights/manifest.json'
if args.command in ('fetch-weights','verify-weights'):
    result=download(manifest,verify_only=args.command=='verify-weights')
elif args.command=='fetch-source':
    inventory=read_manifest(ROOT)
    if inventory['inference']['family']=='ijepa':
        folder=ROOT/'code/vendor';pin=json.loads((folder/'SOURCE_MANIFEST.json').read_text())
        if not all(sha256(folder/item['file'])==item['local_sha256'] for item in pin['files']):
            raise ValueError('Bundled encoder source changed')
        result={'source':'bundled','verified':True}
    else:
        from native import fetch_source
        result=fetch_source(ROOT,args.output)
else:
    from inference import predict
    result=predict(ROOT,args.image,args.device,args.assets)
print(json.dumps(result,indent=2))
