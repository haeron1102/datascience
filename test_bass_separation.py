import torch
import librosa
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from bass_separation_model import UNet #필요한 Library, Module을 선언함.

def separate_bass(model, audio_path, output_path, device='cpu'): #mixture에서 bass만 분리해주는 function을 설정함.
    model.eval() #결과만 나오는 mode.
    y, sr = librosa.load(audio_path, sr=44100) #audio_path의 것을 가져와서 대입함.
    seg_len = 16384 #segment 길이를 설정함.
    segments = []
    for i in range(0, len(y), seg_len): 
        seg = y[i:i+seg_len]
        if len(seg) < seg_len:
            seg = np.pad(seg, (0, seg_len - len(seg)))
        segments.append(seg) #segment별로 list에 추가함.
    separated = [] 
    with torch.no_grad():
        for seg in segments:
            seg_tensor = torch.FloatTensor(seg).unsqueeze(0).unsqueeze(0).to(device) #segment를 torchfloattensor로 바꾸고 차원 바꿔서, device(우리의 경우 cuda)로 보냄.
            out = model(seg_tensor) #seg_tensor을 넣은 결과값을 변수에 넣음.
            separated.append(out.squeeze().cpu().numpy()) #분리된 결과값을 추가함.
    separated_audio = np.concatenate(separated)[:len(y)] #지금까지 저장한 결과값 list를 하나로 합침.
    sf.write(output_path, separated_audio, sr) #separated_audio를 output_path에 저장함.

def test_bass_separation(): #separate와 비교까지 해주는 function을 선언함.
    model = UNet() #model은 bass_separation_model의U UNet으로, 객체를 선언함.
    model.load_state_dict(torch.load('bass_model_multi.pth', map_location='cpu')) #bass_model_multi.pth의 가중치들을 가져와서 대입함.
    device = 'cuda' if torch.cuda.is_available() else 'cpu' #device를 설정함. 우리의 경우 cuda를 사용함.(학교컴ㅎㅎ 4080)
    model = model.to(device)

    test_audio_path = "E:\coding\swai(e)\outputs\Titanium - Haunted Age_mixture.wav" #test할 audio file의 path를 지정함.
    output_path = "./separated_bass_test_multi.wav" #결과물 저장할 path
    label_audio_path = "E:\coding\swai(e)\outputs\Titanium - Haunted Age_bass.wav" #실제 bass 음원과의 비교를 위한 path.

    separate_bass(model, test_audio_path, output_path, device) #분리(separate)를 실행함.
    y_original, sr = librosa.load(label_audio_path, sr=44100) #mixture 음원을 가져옴.
    y_separated, _ = librosa.load(output_path, sr=44100) #분리한 음원을 가져옴.

    plt.figure(figsize=(15, 8)) #그래프 사이즈를 설정함.
    plt.subplot(2, 1, 1) #subplot을 설정함.
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_original)), ref=np.max) #short time 퓨리에 변환을 사용하여서 데이터를 가공하고 이를 절대값해서 db로 변환함.
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log') #spectrum을 보임.
    plt.colorbar(format='%+2.0f dB') #db을 알려주는 colorbar를 보임.
    plt.title('Original Bass') #제목을 설정함.
    plt.subplot(2, 1, 2) 
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y_separated)), ref=np.max) #마찬가지임.
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Separated Bass')
    plt.tight_layout()
    plt.savefig('bass_separation_spec_multi.png', dpi=200) #그린것을 저장함.
    plt.show()

    plt.figure(figsize=(12, 4))
    t = np.linspace(0, len(y_original)/sr, len(y_original)) #분석을 진행할 time을 sr로 나눠서 설정함.
    plt.plot(t[:sr*10], y_original[:sr*10], label='Original Bass', alpha=0.7) #y_original의 10초까지의 data를 그림.
    plt.plot(t[:sr*10], y_separated[:sr*10], label='Separated Bass', alpha=0.7) #y_separated의 10초까지의 data를 그림.
    plt.xlabel('Time (s)') #축제목을 설정함.
    plt.ylabel('Amplitude')
    plt.legend() #범례를 설정함.
    plt.title('Waveform Comparison (First 10 seconds)') #제목을 설정함.
    plt.tight_layout()
    plt.savefig('bass_separation_wave_multi.png', dpi=200) #저장함.
    plt.show()

    print(f"테스트 완료! 결과가 {output_path}에 저장되었습니다.")
    print("스펙트로그램: bass_separation_spec_multi.png, 파형: bass_separation_wave_multi.png")

if __name__ == "__main__":
    test_bass_separation() #실제 비교 function 실행.