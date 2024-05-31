import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import json
import time
import albumentations as A


class KoreanHandWritingDataset(Dataset):
    def __init__(self, root_dir, transform=None, is_train=True, is_sanity_check=None) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        self.is_sanity_check = is_sanity_check
        self.all_image_path_list, self.all_label_list, self.real_image_path_list, self.real_label_list = self.read_paths()

    def __len__(self):
        return len(self.real_image_path_list)

    # TODO: check if the model consumes all the data really
    def __getitem__(self, idx):
        anchor_label = self.real_label_list[idx]
        label_arr = np.array(self.all_label_list)
        label_person, label_sentence, label_real, label_idx_among_all = anchor_label

        # positive labels        
        indices_person = np.where(label_arr[:, 0] == label_person)[0]
        indices_real = np.where(label_arr[:, 2].astype(int) == label_real)[0]

        # 같은 사람, 같은 문장, 자필
        indices_sentence = np.where(label_arr[:, 1] == label_sentence)[0]

        pos_indices = np.intersect1d(indices_person, indices_sentence)
        pos_indices = np.intersect1d(pos_indices, indices_real)
        pos_indices = np.delete(pos_indices, np.where(pos_indices == label_idx_among_all))

        # 확인용 assertion
        assert len(pos_indices) > 0, f"{len(pos_indices) = }"
        idx_check = np.random.choice(pos_indices)
        check_person, check_sentence, check_real = label_arr[idx_check]

        # 같은 사람, 자필
        assert label_person == check_person and label_sentence == check_sentence and label_real == int(check_real) and label_real == 1, f"{label_person = }, {check_person = }, {label_sentence = }, {check_sentence = }, {label_real = }, {check_real = }, "

        # negative labels
        # 다른 사람, 같은 문장
        neg_indices_person = np.where(label_arr[:, 0] != label_person)[0]
        neg_indices_sentence = np.where(label_arr[:, 1] == label_sentence)[0]

        neg_indices = np.intersect1d(neg_indices_person, neg_indices_sentence)

        # 확인용 assertsion
        assert len(neg_indices) > 0, f"{len(neg_indices) = }"
        idx_check = np.random.choice(neg_indices)
        check_person, check_sentence, check_real = label_arr[idx_check]

        # 다른 사람, 같은 문장
        assert label_person != check_person and label_sentence == check_sentence
    
        pos_idx = np.random.choice(pos_indices)
        neg_idx = np.random.choice(neg_indices) 
        
        # get images
        anchor_img = self.read_image(idx, is_anchor=True)
        pos_img = self.read_image(pos_idx, is_anchor=False)
        neg_img = self.read_image(neg_idx, is_anchor=False)
            
        # augmentation
        if self.transform:
            anchor_img = self.transform(image=anchor_img)["image"]
            pos_img = self.transform(image=pos_img)["image"]
            neg_img = self.transform(image=neg_img)["image"]

        anchor_img = self.convert_to_tensor(anchor_img)
        pos_img = self.convert_to_tensor(pos_img)
        neg_img = self.convert_to_tensor(neg_img)

        return anchor_img, pos_img, neg_img
        
    def read_paths(self):
        p = Path(self.root_dir) / f"{'Training' if self.is_train else 'Validation'}"

        image_p = p / "01.원천데이터" / f"{'T' if self.is_train else 'V'}S_images"
        label_p = p / "02.라벨링데이터" / f"{'T' if self.is_train else 'V'}L_labels"
        
        # set the return lists
        all_label_list = []
        all_image_path_list = []
        real_image_path_list = []
        real_label_list = []

        start = time.time()
        time_for_1000 = time.time()
        for i, (image_path, label_path) in enumerate(zip(image_p.glob("*/*/*/*_0150_x*"), label_p.glob("*/*/*/*"))):

            # image, label이 가진 경로가 다르면 assertion
            assert image_path.parts[-4:-2] == label_path.parts[-4:-2], f"{image_path.parts[-4:-2] = }, {label_path.parts[-4:-2] = }"
            with open(label_path, "r", encoding="utf-8-sig") as f:
                j = json.load(f)

            label_person = j["person"]["id"]
            # image_file_name = j["image"][-1]["image_info"]["file_name"]
            label_real = 1 if "자필" in label_path.parts[-3] else 0
            label_sentence = j["image"][-1]["annotations"]["property"]["category_id"]

            all_label_list.append([label_person, label_sentence, label_real])            
            str_image_path = str(image_path)
            if image_path.exists():
                all_image_path_list.append(str_image_path)
            else:
                raise FileNotFoundError(f"image path [{str_image_path}] doesn't exist")
            
            # anchor image를 위한 자필 데이터만 담기
            if label_real == 1:
                real_image_path_list.append(str_image_path)
                real_label_list.append([label_person, label_sentence, label_real, i])
            
            # 모델 확인을 위한 소규모 데이터셋 가져오기
            if self.is_sanity_check is True:
                if self.is_train:
                    if len(real_label_list) == 200:
                        break
                else:
                    if len(real_label_list) == 20:
                        break           
            if (i + 1) % 1000 == 0:
                print(f"[{i + 1}]th iteration... it took [{time.time() - time_for_1000 :.2f}] seconds")
                time_for_1000 = time.time()
        end = time.time()
        assert len(all_image_path_list) == len(all_label_list), f"{len(all_image_path_list) = }, {len(all_label_list) = }"
        print(f"read paths done! it took [{end - start :.2f}] seconds, got [{len(all_label_list)}] labels and real images [{len(real_label_list)}]")
        return all_image_path_list, all_label_list, real_image_path_list, real_label_list

    def read_image(self, idx, is_anchor):
        if is_anchor:
            image_path = self.real_image_path_list[idx]
        else:
            image_path = self.all_image_path_list[idx]
        io_result = False

        io_idx = -1        
        while io_result is False:
            try:
                image = Image.open(image_path)  # there was a case when 150th image was not valid
                io_result = True
            except:
                # get an image file one frame before the last one and on and on
                io_idx -= 1

                # get the new image path
                p = Path(image_path)
                folder = p.parent
                files = list(folder.glob("*"))
                image_path = files[io_idx]
                image_path = str(image_path)

        image = np.array(image)[:, :, :3]  # image has alpha channel -> 4 channels
        return image
    
    def convert_to_tensor(self, image):
        # convert to torch.tensor
        image = image.transpose((2, 0, 1))
        image = torch.tensor(image, dtype=torch.float)
        return image


def get_dataloader(root_dir, is_train, is_sanity_check, batch_size, shuffle):

    # transform
    transform = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ]
    )

    # make dataset
    ds = KoreanHandWritingDataset(root_dir, transform, is_train, is_sanity_check)
    dl = DataLoader(ds, batch_size, shuffle)
    return dl