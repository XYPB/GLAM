from typing import Iterable
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import ImageFilter, Image, ImageDraw
import random
from textaugment import EDA, Word2vec, Translate
import nltk
from randaugment import RandAugment

from .argo_translator import TranslateBackAugment


def otsu_mask(img):
    median = np.median(img)
    _, thresh = cv2.threshold(img, median, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def right_orient_mammogram(image):
    convert = False
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        convert = True
    left_nonzero = cv2.countNonZero(image[:, 0 : int(0.5 * image.shape[1])])
    right_nonzero = cv2.countNonZero(image[:, int(0.5 * image.shape[1]) :])

    is_flipped = left_nonzero < right_nonzero
    if is_flipped:
        image = cv2.flip(image, 1)

    if convert:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image, is_flipped


def remove_text_label(image):
    # Convert the image to a NumPy array if it's a PIL image
    if isinstance(image, Image.Image):
        image = np.array(image)
    convert = False
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        convert = True
    if image.max() <= 1:
        image = (image * 255).astype(np.uint8)  # Convert to 8-bit if not already

    # Binarize the image using a naive non-zero thresholding
    binary_image = (image > 5).astype(np.uint8) * 255

    # Apply Gaussian blur to the binarized image
    blurred_image = cv2.GaussianBlur(binary_image, (5, 5), 2.0)
    # Binarize the blurred image again
    binary_image = (blurred_image > 0).astype(np.uint8) * 255
    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_image, connectivity=8
    )

    # Create an output image to store the result
    output_image = image.copy()

    # Remove small connected components
    for i in range(1, num_labels):  # Start from 1 to skip the background
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 1e4 and area < np.max(
            stats[:, cv2.CC_STAT_AREA]
        ):  # Threshold for small areas, adjust as needed
            x, y, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            output_image[y : y + h, x : x + w] = 0  # Set the region to black
    # if image is set to pure black, return the original image
    if np.all(output_image == 0):
        output_image = image
    if convert:
        output_image = cv2.cvtColor(output_image, cv2.COLOR_GRAY2RGB)
    return output_image


class CutoutPIL(object):
    def __init__(self, cutout_factor=0.5):
        self.cutout_factor = cutout_factor

    def __call__(self, x):
        img_draw = ImageDraw.Draw(x)
        h, w = x.size[0], x.size[1]  # HWC
        h_cutout = int(self.cutout_factor * h + 0.5)
        w_cutout = int(self.cutout_factor * w + 0.5)
        y_c = np.random.randint(h)
        x_c = np.random.randint(w)

        y1 = np.clip(y_c - h_cutout // 2, 0, h)
        y2 = np.clip(y_c + h_cutout // 2, 0, h)
        x1 = np.clip(x_c - w_cutout // 2, 0, w)
        x2 = np.clip(x_c + w_cutout // 2, 0, w)
        fill_color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
        )
        img_draw.rectangle([x1, y1, x2, y2], fill=fill_color)

        return x


class OtsuCut(object):

    def __init__(self, align_orientation: bool = False, remove_text: bool = False):
        super().__init__()
        self.algn_orientation = align_orientation
        self.remove_text = remove_text

    def __process__(self, x):
        if isinstance(x, Image.Image):
            x = np.array(x)

        if self.algn_orientation:
            x, _ = right_orient_mammogram(x)
        if self.remove_text:
            x = remove_text_label(x)

        mask = otsu_mask(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY))
        # Convert to NumPy array if not already

        # Check if the matrix is empty or has no '1's
        if mask.size == 0 or not np.any(mask):
            return Image.fromarray(x)

        # Find the rows and columns where '1' appears
        rows = np.any(mask == 255, axis=1)
        cols = np.any(mask == 255, axis=0)

        # Find the indices of the rows and columns
        min_row, max_row = np.where(rows)[0][[0, -1]]
        min_col, max_col = np.where(cols)[0][[0, -1]]

        # Crop and return the submatrix
        x = x[min_row : max_row + 1, min_col : max_col + 1]

        img = Image.fromarray(x)
        return img

    def __call__(self, x):
        if isinstance(x, Iterable):
            return [self.__process__(im) for im in x]
        else:
            return self.__process__(x)


class DataTransforms(object):
    def __init__(
        self,
        is_train: bool = True,
        img_size: int = 256,
        crop_size: int = 224,
        align_orientation: bool = False,
        remove_text: bool = False,
    ):
        if is_train:
            data_transforms = [
                OtsuCut(align_orientation, remove_text),
                transforms.Resize((img_size, img_size)),
                transforms.RandomResizedCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        else:
            data_transforms = [
                OtsuCut(align_orientation, remove_text),
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class DataFixedTransforms(object):
    def __init__(
        self,
        is_train: bool = True,
        img_size: int = 256,
        crop_size: int = 224,
        align_orientation: bool = True,
        remove_text: bool = True,
    ):
        if is_train:
            data_transforms = [
                OtsuCut(align_orientation, remove_text),
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        else:
            data_transforms = [
                OtsuCut(align_orientation, remove_text),
                transforms.Resize((img_size, img_size)),
                transforms.CenterCrop(crop_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class DetectionDataTransforms(object):
    def __init__(
        self,
        is_train: bool = True,
        image_size: int = 224,
        crop_size: int = 224,
        jitter_strength: float = 1.0,
    ):
        if is_train:
            self.color_jitter = transforms.ColorJitter(
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.8 * jitter_strength,
                0.2 * jitter_strength,
            )

            kernel_size = int(0.1 * 224)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        else:
            data_transforms = [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]

        self.data_transforms = transforms.Compose(data_transforms)

    def __call__(self, image):
        return self.data_transforms(image)


class GaussianBlur:
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(
                sample, (self.kernel_size, self.kernel_size), sigma
            )

        return sample


class SimCLRTransform(object):
    def __init__(
        self,
        is_train: bool = True,
        img_size: int = 256,
        crop_size: int = 224,
        align_orientation: bool = False,
        remove_text: bool = False,
    ):
        if is_train:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomResizedCrop(size=crop_size, scale=(0.25, 1.0)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply(
                        [transforms.GaussianBlur((7, 7), [0.1, 2.0])], p=0.4
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)


class SimCLRFixedTransform(object):
    def __init__(
        self,
        is_train: bool = True,
        img_size: int = 256,
        crop_size: int = 224,
        align_orientation: bool = False,
        remove_text: bool = False,
    ):
        if is_train:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(size=crop_size),
                    transforms.RandomAffine(
                        degrees=10,
                        scale=(0.9, 1.1),
                        translate=(0.05, 0.05),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8
                    ),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomApply(
                        [transforms.GaussianBlur((7, 7), [0.1, 2.0])], p=0.4
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)


class Moco2Transform(object):
    def __init__(
        self,
        is_train: bool = True,
        img_size: int = 256,
        crop_size: int = 224,
        align_orientation: bool = False,
        remove_text: bool = False,
    ) -> None:
        if is_train:
            # This setting follows SimCLR
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomCrop(crop_size),
                    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2)], p=0.4),
                    transforms.RandomApply(
                        [transforms.GaussianBlur((7, 7), [0.1, 2.0])], p=0.4
                    ),
                    transforms.RandomAffine(
                        degrees=10,
                        scale=(0.8, 1.1),
                        translate=(0.0625, 0.0625),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    # output image must be gray scale
                    transforms.RandomGrayscale(p=1.0),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)


class ChestXRayTransform(object):
    def __init__(
        self,
        is_train: bool = True,
        img_size: int = 256,
        crop_size: int = 224,
        align_orientation: bool = False,
        remove_text: bool = False,
    ) -> None:
        self.img_size = img_size
        self.crop_size = crop_size
        if is_train:
            self.data_transforms = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    CutoutPIL(cutout_factor=0.2),
                    # No more cropping
                    # transforms.RandomResizedCrop(size=crop_size, scale=(0.1, 1.0)),
                    transforms.RandomAffine(
                        degrees=10, scale=(0.8, 1.1), translate=(0.0625, 0.0625)
                    ),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomApply(
                        [transforms.ColorJitter(0.5, 0.1, 0, 0)], p=0.5
                    ),
                    transforms.RandomGrayscale(p=0.5),
                    transforms.RandomApply(
                        [transforms.GaussianBlur((5, 5), [0.1, 2.0])], p=0.4
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    transforms.Resize((img_size, img_size)),
                    transforms.CenterCrop(crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)


class RSNAMammoTransform(object):
    def __init__(
        self,
        is_train: bool = True,
        img_size: int = 256,
        crop_size: int = 224,
        align_orientation: bool = False,
        remove_text: bool = False,
    ) -> None:
        if is_train:
            # This setting follows SimCLR
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.RandomApply([transforms.ColorJitter(0.2, 0.2)], p=0.4),
                    transforms.RandomApply(
                        [transforms.GaussianBlur((5, 5), [0.1, 2.0])], p=0.4
                    ),
                    transforms.RandomAffine(
                        degrees=10,
                        scale=(0.9, 1.1),
                        translate=(0.05, 0.05),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    # output image must be gray scale
                    transforms.RandomGrayscale(p=1.0),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.ToTensor(),
                ]
            )
        else:
            self.data_transforms = transforms.Compose(
                [
                    OtsuCut(align_orientation, remove_text),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(),
                ]
            )

    def __call__(self, img):
        return self.data_transforms(img)


class BackTranslation(object):

    def __init__(self, use_google: bool = False):
        if use_google:
            self.t_list = [
                Translate(src="en", to="es"),
                Translate(src="en", to="fr"),
                Translate(src="en", to="de"),
                Translate(src="en", to="it"),
            ]
        else:
            self.t_list = [
                TranslateBackAugment(src="en", to="es"),
                TranslateBackAugment(src="en", to="fr"),
                TranslateBackAugment(src="en", to="de"),
                TranslateBackAugment(src="en", to="it"),
            ]

    def __call__(self, sentence):
        t = random.choice(self.t_list)
        try:
            augmented_text = t.augment(sentence)
        except Exception as e:
            # use the original text if there is an error
            print(f"Text Translation Error During Augmentation: {e}")
            augmented_text = sentence
        return augmented_text


class SentenceSwap(object):

    def __init__(self, stop_token, n=1):
        assert isinstance(stop_token, str)
        self.stop_token = stop_token
        self.n = n

    def __call__(self, text):
        text = text.split(self.stop_token)
        if len(text) <= 1:
            return self.stop_token.join(text)
        for _ in range(self.n):
            idx2swap = random.sample(range(len(text)), 2)
            sent1 = text[idx2swap[0]]
            sent2 = text[idx2swap[1]]
            text[idx2swap[0]] = sent2
            text[idx2swap[1]] = sent1
        return self.stop_token.join(text)


class RandomWordDeletion(object):

    def __init__(
        self, p=0.1, max_deletion=10, punctuations="!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    ):
        self.p = p
        self.max_deletion = max_deletion
        self.punctuations = punctuations

    def __call__(self, text):
        assert isinstance(text, str)
        words = text.split()
        if len(words) <= self.max_deletion:
            return text
        new_words = []
        cnt = 0
        for word in words:
            if any(p in word for p in self.punctuations):
                new_words.append(word)
            elif random.random() > self.p and cnt < self.max_deletion:
                new_words.append(word)
            else:
                cnt += 1
        output = " ".join(new_words)
        return " ".join(new_words)


class TextTransform(object):

    # default 0.2 prob not to use any aug
    def __init__(
        self,
        is_train: bool = True,
        bos_token: str = None,
        eos_token: str = None,
        stop_token: str = None,
        remove_stop_word_prob: float = 0.4,
        synonym_replacement_prob: float = 0.2,
        random_swap_prob: float = 0.0,
        random_deletion_prob: float = 0.0,
        random_sent_swap_prob: float = 0.1,
        random_back_translation_prob: float = 0.2,
    ):
        self.is_train = is_train
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.stop_token = stop_token
        self.stop_words = set(nltk.corpus.stopwords.words("english"))
        remove_stop_word = lambda x: " ".join(
            [word for word in x.split() if word.lower() not in self.stop_words]
        )
        eda = EDA()
        # This format only uses 1 augmenter at a time, replace it with a list of augmenters
        self.preprocess_func = (
            lambda x: x.replace(self.bos_token, "")
            .replace(self.eos_token, "")
            .replace(self.stop_token, ".")
        )
        self.postprocess_func = (
            lambda x: f"{self.bos_token} {x} {self.eos_token}".replace(
                ".", self.stop_token
            )
        )

        self.transform_list = [
            remove_stop_word,
            eda.synonym_replacement,
            eda.random_swap,
            RandomWordDeletion(),
            SentenceSwap("."),
            BackTranslation(use_google=True),
        ]
        self.prob_list = [
            remove_stop_word_prob,
            synonym_replacement_prob,
            random_swap_prob,
            random_deletion_prob,
            random_sent_swap_prob,
            random_back_translation_prob,
        ]

    def __call__(self, text):
        if self.is_train:
            try:
                assert self.stop_token in text
            except AssertionError:
                print(f"Stop token not found in text: {text}, try resolving this issue")
                text = text.replace("   ", " " + self.stop_token + " ")
                assert self.stop_token in text
            text = self.preprocess_func(text)
            for transform, prob in zip(self.transform_list, self.prob_list):
                if random.random() < prob:
                    text = transform(text)
            text = self.postprocess_func(text)
            return text
        else:
            return text


# class GaussianBlur:
#     """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709."""

#     def __init__(self, sigma=(0.1, 2.0)):
#         self.sigma = sigma

#     def __call__(self, x):
#         sigma = random.uniform(self.sigma[0], self.sigma[1])
#         x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
#         return x
