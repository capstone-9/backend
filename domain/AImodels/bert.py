import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoTokenizer, AutoModel, AutoConfig

# 설정
BERT = "xlm-roberta-large"
HIDDEN_SIZE = 256
OUTPUT_SIZE = 6
MAX_LEN = 400
LABELS = ["분노", "기쁨", "불안", "당황", "슬픔", "상처"]

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
class EmotionModel(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(EmotionModel, self).__init__()
        self.bert = AutoModel.from_pretrained(BERT)
        self.config = AutoConfig.from_pretrained(BERT)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.config.hidden_size, output_size)

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = output.last_hidden_state[:, 0, :]  # [CLS] 토큰
        return self.fc(self.dropout(cls_output))

# 모델 로딩
tokenizer = AutoTokenizer.from_pretrained(BERT)
model = EmotionModel(HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load(
    "./xlm-roberta-large_42_fold1.pth",
    map_location=torch.device('cpu')
))
model.eval()

# ✅ 감정 분석 함수
def analyze_emotion(text: str) -> str:
    dataset = TestDataset([text], tokenizer)
    dataloader = DataLoader(dataset, batch_size=1)

    for input_ids, attention_mask in dataloader:
        with torch.no_grad():
            output = model(input_ids, attention_mask)
            probs = output.detach().numpy()
            pred = np.argmax(probs, axis=1)[0]
            return LABELS[pred]
