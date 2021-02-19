from torchvision import transforms


def build_transforms(cfg):
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg.DATA.GLOBAL_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    ])