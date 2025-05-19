import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig
import os

# 설정
BERT = "beomi/KcELECTRA-base-v2022"
HIDDEN_SIZE = 256
OUTPUT_SIZE = 6
MAX_LEN = 512
LABELS = ["분노", "기쁨", "불안", "당황", "슬픔", "상처"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "KcELECTRA-base-v2022_probs_43.pth")

# 데이터셋 클래스
class TestDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenized = tokenizer.batch_encode_plus(
            texts,
            padding='max_length',
            add_special_tokens=True,
            truncation=True,
            max_length=MAX_LEN,
            return_tensors='pt'
        )

    def __len__(self):
        return self.tokenized['input_ids'].shape[0]

    def __getitem__(self, idx):
        return self.tokenized['input_ids'][idx], self.tokenized['attention_mask'][idx]

# 모델 클래스
class GlobalMaxPool1D(nn.Module):
    def forward(self, x):
        return x.max(dim=1).values

class EmotionModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(EmotionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(BERT)
        self.config = AutoConfig.from_pretrained(BERT)
        self.pooling = GlobalMaxPool1D()
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = output[0]
        pooled = self.pooling(x)
        logits = self.fc(self.dropout(pooled))
        return logits

# 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(BERT)
model = EmotionModel(HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
model.eval()

# ✅ 감정 분석 함수 (확률 리스트 반환)
def analyze_emotion(text: str) -> list:
    dataset = TestDataset([text], tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    for input_ids, attention_mask in dataloader:
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probs = F.softmax(output, dim=1).cpu().numpy()[0]  # shape: (6,)
            return [round(float(prob * 100), 1) for prob in probs]