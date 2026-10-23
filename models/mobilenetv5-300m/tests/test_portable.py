"""Portable loading must respect shared encoders and the requested head."""
import hashlib
import json
from pathlib import Path
from PIL import Image
import pytest
import torch
from torch import nn
from tig_benchmark import inference


def spec(path, root):
    return {'path':path.relative_to(root).as_posix(), 'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


@pytest.fixture
def package(tmp_path, monkeypatch):
    model = json.loads((Path(__file__).resolve().parents[1]/'model.json').read_text())['runtime_model']
    mobile = model == 'mobilenetv5'
    weights = tmp_path/'weights'; (weights/'best').mkdir(parents=True); (weights/'final').mkdir()
    (tmp_path/'config.json').write_text('{}')
    base = weights/('pretrained/encoder.safetensors' if mobile else 'best/encoder.safetensors')
    base.parent.mkdir(exist_ok=True);base.write_bytes(b'released encoder fixture')
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.value=nn.Parameter(torch.tensor(3.))
        def forward(self, image):
            return self.value.expand(len(image),2048)
    monkeypatch.setattr(inference,'load_assets',lambda *args:{'model':model,'checkpoint_sha256':spec(base,weights)['sha256']})
    monkeypatch.setattr(inference,'build',lambda *args:(Tiny(),lambda image:torch.zeros(3,2,2)))
    for kind,value in [('best',7.),('final',9.)]:
        classifier=weights/kind/'classifier.pt';w=torch.zeros(1,2048);w[0,0]=1.
        torch.save({'head':{'weight':w,'bias':torch.tensor([float(kind=='final')])},'mean':torch.zeros(2048),'std':torch.ones(2048)},classifier)
        encoder=base
        if mobile:
            encoder=weights/kind/'encoder.pt';torch.save({'encoder':{'value':torch.tensor(value)}},encoder)
        desc={'format':'tig-checkpoint-v1','kind':kind,'mode':'supervised' if mobile else 'frozen',
              'geometry':'native','encoder_format':'supervised_torch' if mobile else 'native_safetensors',
              'source_model_config_sha256':hashlib.sha256(b'{}').hexdigest(),
              'encoder':spec(encoder,weights),'classifier':spec(classifier,weights)}
        if mobile:desc['pretrained_encoder']=spec(base,weights)
        (weights/kind/'model.json').write_text(json.dumps(desc))
    image=tmp_path/'fixture.png';Image.new('RGB',(2,2)).save(image)
    return tmp_path,image,mobile


def test_requested_checkpoint_loads_its_exact_encoder_and_head(package):
    folder,image,mobile=package
    best=inference.predict_local(folder,image,checkpoint='best')
    final=inference.predict_local(folder,image,checkpoint='final')
    assert best['logit']==(7. if mobile else 3.)
    assert final['logit']==(10. if mobile else 4.)


def test_modified_classifier_is_rejected(package):
    folder,image,_=package
    (folder/'weights/final/classifier.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        inference.predict_local(folder,image,checkpoint='final')


def test_descriptor_cannot_escape_the_weights_directory(tmp_path):
    with pytest.raises(ValueError,match='inside'):
        inference.checkpoint_file(tmp_path,{'path':'../elsewhere','sha256':'unused'})
