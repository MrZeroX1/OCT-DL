import torch
from src.model import MaskedBCEWithLogitsLoss, build_model
def test_masked_loss_ignores_unknown_labels():
 logits=torch.tensor([[0.,0.]],requires_grad=True); targets=torch.tensor([[1.,0.]]);mask=torch.tensor([[True,False]])
 loss=MaskedBCEWithLogitsLoss()(logits,targets,mask);loss.backward();assert logits.grad[0,1].item()==0
def test_model_output_shape_without_weights():
 model=build_model("resnet18",6,pretrained=False);assert model(torch.randn(2,3,224,224)).shape==(2,6)
