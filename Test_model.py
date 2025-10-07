import torch
from network.VDLNet import VDLNet


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_model():
    torch.manual_seed(42)
    model = VDLNet(visual_encoder_name='convnext_base').to(device)

    batch_size = 2
    img_size = 256
    rgb_img = torch.randn(batch_size, 3, img_size, img_size).to(device)  
    depth_img = torch.randn(batch_size, 1, img_size, img_size).to(device)
    texts = [
        "a red car in the foreground",  
        "a person standing near the door"
    ]

    with torch.no_grad():
        saliency_map = model(rgb_img, depth_img, texts)

    print("Saliency Map Shape:", saliency_map.shape)

if __name__ == "__main__":
    print("Model Test...")
    test_model()
    print("Test Done !")
