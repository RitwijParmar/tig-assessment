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
    weights = tmp_path/'weights'; (weights/'best').mkdir(parents=True)
    (tmp_path/'config.json').write_text('{}')
    base = weights/'best/encoder.safetensors'
    if not mobile:base.write_bytes(b'released encoder fixture')
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.value=nn.Parameter(torch.tensor(3.))
        def forward(self, image):
            return self.value.expand(len(image),2048)
    def load(*args, **kwargs):
        assert kwargs == {'require_checkpoint':False}
        return {'model':model,'checkpoint':None}
    def build(*args, **kwargs):
        assert kwargs == {'load_pretrained':False}
        return Tiny(),lambda image:torch.zeros(3,2,2)
    monkeypatch.setattr(inference,'load_assets',load)
    monkeypatch.setattr(inference,'build',build)
    for kind,value in [('best',7.)]:
        classifier=weights/kind/'classifier.pt';w=torch.zeros(1,2048);w[0,0]=1.
        torch.save({'head':{'weight':w,'bias':torch.tensor([float(kind=='final')])},'mean':torch.zeros(2048),'std':torch.ones(2048)},classifier)
        encoder=base
        if mobile:
            encoder=weights/kind/'encoder.pt';torch.save({'encoder':{'value':torch.tensor(value)}},encoder)
        desc={'format':'tig-checkpoint-v1','kind':kind,'mode':'supervised' if mobile else 'frozen',
              'geometry':'native','encoder_format':'supervised_torch' if mobile else 'native_safetensors',
              'source_model_config_sha256':hashlib.sha256(b'{}').hexdigest(),
              'encoder':spec(encoder,weights),'classifier':spec(classifier,weights)}
        (weights/kind/'model.json').write_text(json.dumps(desc))
    image=tmp_path/'fixture.png';Image.new('RGB',(2,2)).save(image)
    return tmp_path,image,mobile


def test_requested_checkpoint_loads_its_exact_encoder_and_head(package):
    folder,image,mobile=package
    best=inference.predict_local(folder,image,checkpoint='best')
    assert best['logit']==(7. if mobile else 3.)
    with pytest.raises(ValueError,match='Only the best'):
        inference.predict_local(folder,image,checkpoint='final')


def test_modified_classifier_is_rejected(package):
    folder,image,_=package
    (folder/'weights/best/classifier.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='checksum'):
        inference.predict_local(folder,image,checkpoint='best')


def test_descriptor_cannot_escape_the_weights_directory(tmp_path):
    with pytest.raises(ValueError,match='inside'):
        inference.checkpoint_file(tmp_path,{'path':'../elsewhere','sha256':'unused'})


def test_source_only_assets_cannot_initialize_training(tmp_path, monkeypatch):
    from tig_benchmark import assets
    from tig_benchmark.shared import digest
    config = assets.model_config('mobilenetv5')
    monkeypatch.setattr(assets,'verify_source',lambda *args:'pinned-source')
    lock={'format':'tig-model-assets-v1','model':'mobilenetv5','source':'verified',
          'source_revision':'pinned-source','checkpoint':None,
          'checkpoint_sha256':config['checkpoint_sha256'],
          'model_config_sha256':digest(config)}
    path=tmp_path/'assets.local.json';path.write_text(json.dumps(lock))
    assert assets.load_assets('mobilenetv5',path,require_checkpoint=False)==lock
    with pytest.raises(ValueError,match='Training requires'):
        assets.load_assets('mobilenetv5',path)
