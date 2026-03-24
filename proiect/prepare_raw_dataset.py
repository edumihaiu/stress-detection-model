import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = "../datasets"
OUT_DIR = "../dataset_curat"

#  0 = relaxat, 1 = neutru, 2 = stresat
FER_MAP = {'happy': 0, 'neutral': 1, 'surprise': 1, 'sad': 2, 'angry': 2, 'fear': 2, 'disgust': 2}
AFFECTNET_MAP = {'Happiness': 0, 'Neutral': 1, 'Surprise': 1, 'Sadness': 2, 'Anger': 2, 'Fear': 2, 'Disgust': 2, 'Contempt': 2}
CASME_MAP = {'happiness': 0, 'surprise': 1, 'sadness': 2, 'fear': 2, 'disgust': 2, 'repression': 2, 'contempt': 2, 'others': 1}
RAFDB_MAP = {4: 0, 1: 1, 7: 1, 2: 2, 3: 2, 5: 2, 6: 2}

def create_dirs():
    for split in ['train', 'test']:
        for cls in ['0', '1', '2']:
            os.makedirs(os.path.join(OUT_DIR, split, cls), exist_ok=True)

def copy_fer_or_affectnet(dataset_name, map_dict):
    for split in ['train', 'test']:  
        real_split = "Train" if (dataset_name == "affectnet" and split == "train") else split
        real_split = "Test" if (dataset_name == "affectnet" and split == "test") else real_split
        
        split_dir = os.path.join(BASE_DIR, dataset_name, real_split)
        if not os.path.exists(split_dir):
            continue
            
        for emotion_folder in os.listdir(split_dir):
            if emotion_folder not in map_dict:
                continue
                
            target_class = str(map_dict[emotion_folder])
            src_folder = os.path.join(split_dir, emotion_folder)
            dest_folder = os.path.join(OUT_DIR, split.lower(), target_class)
            
            for img in os.listdir(src_folder):
                if img.endswith(('.jpg', '.png')):
                    shutil.copy2(os.path.join(src_folder, img), os.path.join(dest_folder, f"{dataset_name}_{img}"))

def process_casme():
    casme_dir = os.path.join(BASE_DIR, "casme2")
    if not os.path.exists(casme_dir):
        return
        
    for emotion_folder in os.listdir(casme_dir):
        if emotion_folder not in CASME_MAP:
            continue
            
        target_class = str(CASME_MAP[emotion_folder])
        src_folder = os.path.join(casme_dir, emotion_folder)
        
        images = [f for f in os.listdir(src_folder) if f.endswith('.jpg')]
        if len(images) == 0:
            continue
            
        # 80% train, 20% test
        train_imgs, test_imgs = train_test_split(images, test_size=0.2, random_state=42)
        
        for img in train_imgs:
            shutil.copy2(os.path.join(src_folder, img), os.path.join(OUT_DIR, "train", target_class, f"casme_{img}"))
        for img in test_imgs:
            shutil.copy2(os.path.join(src_folder, img), os.path.join(OUT_DIR, "test", target_class, f"casme_{img}"))

def process_rafdb():
    raf_dir = os.path.join(BASE_DIR, "rafdb")
    
    # train, test
    for split, label_file in [('train', 'train_labels.csv'), ('test', 'test_labels.csv')]:
        csv_path = os.path.join(raf_dir, label_file)
        if not os.path.exists(csv_path):
            continue
            
        df = pd.read_csv(csv_path, names=['image', 'label'], header=0) 
        
        for index, row in df.iterrows():
            img_name = row['image']
            label = int(row['label'])
            
            if label not in RAFDB_MAP:
                continue
                
            target_class = str(RAFDB_MAP[label])
            

            src_img = os.path.join(raf_dir, split, str(label), img_name) 
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, os.path.join(OUT_DIR, split, target_class, f"rafdb_{img_name}"))
            else:
                print(f"[-] lipsește poza: {src_img}")

if __name__ == "__main__":
    print("start")
    create_dirs()
    copy_fer_or_affectnet("fer2013", FER_MAP)
    copy_fer_or_affectnet("affectnet", AFFECTNET_MAP)
    process_casme()
    process_rafdb()
    print("dataset_clean created")
