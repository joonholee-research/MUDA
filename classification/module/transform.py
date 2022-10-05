from torchvision import transforms

## Tiny: mnist, usps
transform_n0 = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(28),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_w0 = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(32),
    transforms.RandomCrop(size=(28,28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## Mid: mnist
transform_n1 = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_w1 = transforms.Compose([
    transforms.Grayscale(3),
    transforms.Resize(36),
    transforms.RandomCrop(size=(32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## Mid: svhn
transform_n2 = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_w2 = transforms.Compose([
    transforms.Resize(36),
    transforms.RandomCrop(size=(32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## R50: office-31, office-home
transform_n3 = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_w3 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(size=(224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## R101: VisDA
transform_n4 = transform_n3

transform_w4 = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(size=224, scale=(0.8, 1.0), ratio=(0.75, 1.3333333333333333)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

## Big: synsign, gtsrb
transform_n5 = transforms.Compose([
    transforms.Resize((40,40)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_w5 = transforms.Compose([
    transforms.Resize(44),
    transforms.RandomCrop(size=(40,40)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

## R18: miniDomainNet
transform_n6 = transforms.Compose([
    transforms.Resize(104),
    transforms.CenterCrop(size=(96,96)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_w6 = transforms.Compose([
    transforms.Resize(104),
    transforms.RandomCrop(size=(96,96)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
