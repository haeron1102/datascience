import os
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm # 사용할 Library를 선언함.

class Train_Dataset(Dataset): #학습에 사용할 여러가지 음원 파일을 하나의 dataset으로 묶는 class임.
    def __init__(self, root_dir, seg_len=16384, split='train'): # segment 길이와, 위치 등을 입력받음.
        self.seg_len = seg_len
        self.mix_segments = []
        self.bass_segments = []
        files = os.listdir(root_dir)
        mixture_files = [f for f in files if f.endswith('_mixture.wav')]
        mixture_files = sorted(mixture_files)[0:10] #folder의 file 중 상위 10개를 이용함.
        for mix_file in mixture_files:
            base = mix_file[:-12] #음원의 앞 이름만 가져옴.
            bass_file = base + '_bass.wav'
            mix_path = os.path.join(root_dir, mix_file) #mixture file의 path를 설정함.
            bass_path = os.path.join(root_dir, bass_file) #bass file의 path를 설정함.
            mix, _ = librosa.load(mix_path, sr=44100, mono=True) #librosa를 활용하여 mixture data를 load함.
            bass, _ = librosa.load(bass_path, sr=44100, mono=True) #librosa를 활용하여 bass data를 load함.
            n_seg = (len(mix) - seg_len) // seg_len #3/4을 train으로 사용하기 위해 전체 segment수를 계산함.
            cut = int(n_seg * 0.75)
            indexes = range(0, cut) if split=='train' else range(cut, n_seg) #train인 경우 0 ~ cut, val인 경우 cut ~ n_seg까지
            for i in indexes:
                s = i * seg_len
                self.mix_segments.append(mix[s:s+seg_len])
                self.bass_segments.append(bass[s:s+seg_len]) #여기까지 하여서 mixture와 bass의 segment를 리스트로 정리함.

    def __len__(self):
        return len(self.mix_segments) #dataset의 필수적인 함수 선언. 
    
    def __getitem__(self, i):
        m = self.mix_segments[i]
        b = self.bass_segments[i]
        m = m / np.max(np.abs(m)) if np.max(np.abs(m))>0 else m #m data를 0과 1 사이(0도 포함)로 정규화함.
        b = b / np.max(np.abs(b)) if np.max(np.abs(b))>0 else b #b data를 0과 1 사이(0도 포함)로 정규화함.
        return torch.FloatTensor(m), torch.FloatTensor(b) #dataset의 필수적인 함수 선언.


class UNet(torch.nn.Module): #강의에 나온 UNet 구조를 이요한 class를 선언함.
    def __init__(self):
        super().__init__()
        self.enc1 = torch.nn.Conv1d(1, 32, 15, 1, 7)
        self.enc2 = torch.nn.Conv1d(32, 64, 15, 2, 7)
        self.enc3 = torch.nn.Conv1d(64, 128, 15, 2, 7)
        self.dec3 = torch.nn.ConvTranspose1d(128, 64, 16, 2, 7)
        self.dec2 = torch.nn.ConvTranspose1d(64, 32, 16, 2, 7)
        self.out = torch.nn.Conv1d(32, 1, 15, 1, 7) #encoder, decoder layer을 설정하였음.
        self.act = torch.nn.Tanh() #출력값이 0~1 사이가 되도록 하이퍼볼릭 탄젠트를 사용함.
    def forward(self, x):
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))
        e3 = torch.relu(self.enc3(e2))
        d3 = torch.relu(self.dec3(e3))
        d2 = torch.relu(self.dec2(d3))
        out = self.act(self.out(d2)) #forward method를 선언함.
        return out

def train():
    root_dir = './outputs' #train data path를 설정함.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #학습을 진행할 device 변수를 지정함.
    train_set = Train_Dataset(root_dir, split='train') #실제 학습에 사용할 train_set을 Trian_Dataset class를 이용하여서 제작함.
    val_set = Train_Dataset(root_dir, split='val') #val을 측정할 때 사용할 val_set을 Train_Dataset class를 이용하여서 제작함.
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True) 
    val_loader = DataLoader(val_set, batch_size=16) #data loader를 이용하여서 data 제공에 대해 지정하고 수월하게 함.
    model = UNet().to(device) #cuda(우리의 경우)로 model을 옮김.
    loss_fn = torch.nn.MSELoss() #MSE로 loss func을 정의함.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001) #optimizer model을 정의함.

    
    model_path = 'bass_model_multi.pth' #train에 사용할 model의 path를 지정함.
    model.load_state_dict(torch.load(model_path, map_location=device)) #path에서의 model을 가져옴.
    
    train_losses, val_losses = [], [] #loss값을 저장할 list를 선언함.
    num_epochs = 300
    for epoch in tqdm(range(num_epochs), desc='Epochs'): #epoch만큼 반복함. 진행률을 보기 위해서 tqdm library를 사용함.
        model.train() #train mode로 설정함.
        t_loss = 0
        for m, b in tqdm(train_loader, desc=f"Train {epoch+1}/{num_epochs}", leave=False): #train을 실시하기 위해 반복문을 사용함.
            m, b = m.unsqueeze(1).to(device), b.unsqueeze(1).to(device) #1d data를 학습을 위해 2d data로 바뀌주고, device로 옮김.
            optimizer.zero_grad() #초기 값들을 initializing 함. 
            out = model(m) #결과값을 변수에 지정함.
            loss = loss_fn(out, b) #predict값과 b의 mse로 loss를 지정함.
            loss.backward() #바탕으로 역전파를 진행함. 
            optimizer.step() #weight update함.
            t_loss += loss.item() #loss값을 t_loss에 추가함.
        t_loss /= len(train_loader) #평균 loss값을 알기위해 나눔.
        model.eval() #evaluate mode로 설정함.
        v_loss = 0
        with torch.no_grad(): #val loss를 측정함.
            for m, b in tqdm(val_loader, desc=f"Val {epoch+1}/{num_epochs}", leave=False):
                m, b = m.unsqueeze(1).to(device), b.unsqueeze(1).to(device)
                out = model(m)
                v_loss += loss_fn(out, b).item()
        v_loss /= len(val_loader) #평균 val loss값을 알기위해 나눔.
        train_losses.append(t_loss) #추가!
        val_losses.append(v_loss) #추가!
        if (epoch+1)%100==0 or epoch==0:
            tqdm.write(f"{epoch+1} epoch: train {t_loss:.5f}, val {v_loss:.5f}") #loss값을 출력함.
    plt.plot(train_losses, label='train') #for문이 끝난후 loss값의 추세를 알기위해서 그래프를 그림.
    plt.plot(val_losses, label='val') 
    plt.legend(); plt.savefig('loss_multi.png'); plt.close() #그래플 저장하고, 닫음.
    torch.save(model.state_dict(), 'bass_model_multi.pth') #train된 model을 저장함.
    print('학습 완료!')

if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else 'cpu') #현재 사용하는 device가 뭔지 알기위하여 출력함.
    train() # train을 진행함.