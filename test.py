import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score
from transformers import AutoTokenizer, AutoModel
from PIL import Image
import clip
import matplotlib.pyplot as plt
import os
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# 加载图像模型（CLIP ViT-B/16）
clip_model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
clip_model.visual.float()
for name, param in clip_model.visual.named_parameters():
    if any(layer in name for layer in ["ln_post", "proj", "transformer.resblocks.10", "transformer.resblocks.11"]):
        param.requires_grad = True
    else:
        param.requires_grad = False
clip_model.eval()

# 加载文本模型
text_model_name = "emilyalsentzer/Bio_ClinicalBERT"
#text_model_name = "bert-base-uncased"  # 修改为普通BERT
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
text_model = AutoModel.from_pretrained(text_model_name).to(device)

# 数据集类
class MedDataset(Dataset):
    def __init__(self, csv_path, use="training"):
        df = pd.read_csv(csv_path)
        df = df[df["use"] == use]
        self.image_paths = df["filepath"].tolist()
        self.reports = df["gpt4_summary"].tolist()
        self.labels = df["glaucoma"].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        report = self.reports[idx][:512]
        label = self.labels[idx]
        return image, report, label

def custom_collate_fn(batch):
    images, reports, labels = zip(*batch)
    return list(images), list(reports), torch.tensor(labels, dtype=torch.float32)

# 图文融合分类器 + 对比损失输出
class CLIPFusionClassifier(nn.Module):
    def __init__(self):
        super(CLIPFusionClassifier, self).__init__()
        self.text_model = text_model
        self.image_encoder = clip_model.visual

        self.image_proj = nn.Linear(512, 256)
        self.text_proj = nn.Linear(768, 256)

        self.attn_pool = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 2),
            nn.Softmax(dim=1)
        )

        # 新增文本注意力池化层
        self.text_attention = nn.Sequential(
            nn.Linear(768, 128),  # 将768维特征映射到128维
            nn.Tanh(),
            nn.Linear(128, 1),    # 输出每个token的注意力权重
            nn.Softmax(dim=1)     # 在序列长度维度做归一化
        )
        '''
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1)
        )
        '''
        self.mlp_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.Linear(256, 1)
        )
        '''
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
            )
        '''

    def forward(self, images, reports):
        image_inputs = torch.stack([preprocess(img) for img in images]).to(device)
        image_features = self.image_encoder(image_inputs)
        image_features = self.image_proj(image_features)

        text_inputs = text_tokenizer(reports, padding=True, truncation=True,
                                     return_tensors='pt', max_length=512).to(device)
        text_outputs = self.text_model(**text_inputs)
        
         # 获取所有token的特征 [batch_size, seq_len, 768]
        token_features = text_outputs.last_hidden_state  
        
        # 计算每个token的注意力权重 [batch_size, seq_len, 1]
        attention_weights = self.text_attention(token_features)  
        
        # 加权求和得到文本特征 [batch_size, 768]
        text_features = (attention_weights * token_features).sum(dim=1)  
        
        # 后续投影和融合保持不变
        text_features = self.text_proj(text_features)
        #text_features = text_outputs.last_hidden_state.mean(dim=1)
        #text_features = self.text_proj(text_features)

        combined = torch.cat([image_features, text_features], dim=1)
        weights = self.attn_pool(combined)

        image_weighted = image_features * weights[:, 0].unsqueeze(1)
        text_weighted = text_features * weights[:, 1].unsqueeze(1)
        fused = image_weighted + text_weighted

        #fused = fused + image_features  # residual
        #fused = fused + text_features  # residual
        logits = self.mlp_head(fused)
        return logits, image_features, text_features

# 对比损失函数
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, image_feat, text_feat):
        image_feat = nn.functional.normalize(image_feat, dim=-1)
        text_feat = nn.functional.normalize(text_feat, dim=-1)
        logits = torch.matmul(image_feat, text_feat.T) / self.temperature
        labels = torch.arange(len(image_feat)).to(device)
        loss_i2t = nn.CrossEntropyLoss()(logits, labels)
        loss_t2i = nn.CrossEntropyLoss()(logits.T, labels)
        return (loss_i2t + loss_t2i) / 2

# alpha调度函数（cosine decay）
#def get_alpha(epoch, total_epochs, max_alpha=0.1):
    #return max_alpha * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

def get_alpha(epoch, total_epochs=None, max_alpha=0.1, decay_rate=0.98):
    return max_alpha * (decay_rate ** epoch)

# 评估函数
@torch.no_grad()
def compute_auc_and_accuracy(model, dataloader):
    model.eval()
    all_labels, all_probs = [], []
    for images, reports, labels in dataloader:
        labels = labels.to(device).unsqueeze(1)
        logits, _, _ = model(images, reports)
        probs = torch.sigmoid(logits)
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    all_preds = (torch.tensor(all_probs) > 0.5).int().numpy()
    auc = roc_auc_score(all_labels, all_probs)
    acc = accuracy_score(all_labels, all_preds)
    return auc, acc

# 训练函数（带图文对比损失 + 动态 alpha）
def train(model, train_loader, val_loader, optimizer, cls_criterion, contrast_criterion, epochs=10, patience=10):
    best_val_auc = 0
    no_improve_epochs = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        alpha = get_alpha(epoch, epochs)
        for images, reports, labels in train_loader:
            labels = labels.to(device).unsqueeze(1)
            logits, img_feat, txt_feat = model(images, reports)
            cls_loss = cls_criterion(logits, labels)
            contrast_loss = contrast_criterion(img_feat, txt_feat)
            scaled_contrast_loss = contrast_loss / contrast_loss.detach().mean()
            loss = cls_loss + alpha * scaled_contrast_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        train_auc, train_acc = compute_auc_and_accuracy(model, train_loader)
        val_auc, val_acc = compute_auc_and_accuracy(model, val_loader)

        print(f"[Epoch {epoch+1}/{epochs}] Loss: {avg_train_loss:.4f}  Train AUC: {train_auc:.4f}  Val AUC: {val_auc:.4f}  Alpha: {alpha:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), "best_model.pth")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= patience:
                print("⏹️ Early stopping.")
                break

# 参数设置
csv_path = "/home/cds/MedCLIP-main/eye1.csv"
#csv_path = "/home/cds/MedCLIP-main/eye2.csv"
batch_size = 32
epochs = 100

# 数据加载
train_dataset = MedDataset(csv_path, use="training")
val_dataset = MedDataset(csv_path, use="validation")
test_dataset = MedDataset(csv_path, use="test")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

model = CLIPFusionClassifier().to(device)
vision_params = [p for n, p in model.image_encoder.named_parameters() if p.requires_grad]
other_params = [p for n, p in model.named_parameters() if "image_encoder" not in n and p.requires_grad]
optimizer = optim.AdamW([
    {"params": vision_params, "lr": 1e-7},
    {"params": other_params, "lr": 1e-6}
])
cls_criterion = nn.BCEWithLogitsLoss()
contrast_criterion = ContrastiveLoss()

train(model, train_loader, val_loader, optimizer, cls_criterion, contrast_criterion, epochs=epochs, patience=5)

model.load_state_dict(torch.load("best_model.pth"))
test_auc, test_acc = compute_auc_and_accuracy(model, test_loader)
print(f"✅ Best Model Test AUC: {test_auc:.4f}")
print(f"✅ Best Model Test Acc: {test_acc:.4f}")
save_dir = f"qgy-best_AUC_{test_auc:.4f}_ACC_{test_acc:.4f}"
os.makedirs(save_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))